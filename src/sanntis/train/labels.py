#!/usr/bin/env python3

# Copyright 2026 EMBL - European Bioinformatics Institute
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""NPClassifier labels from the compounds of MIBiG-format entries (MIBiG and external data alike).

Per entry: ``compounds[].structure`` (SMILES) → RDKit normalisation (largest fragment, uncharged, canonical
tautomer, isotopes cleared) → InChIKey; one SMILES per skeleton (first InChIKey block, lowest InChIKey) is sent to
the NPClassifier API (``npclassifier.gnps2.org``). Returned terms are mapped onto the frozen NPClassifier vocabulary
shipped in ``sanntis/data/npc`` (terms outside it are counted and dropped); a BGC's labels are the union over its
compounds.

API answers are cached verbatim under ``$SANNTIS_CACHE/npclassifier`` (one file per SMILES), so a re-run makes no
request. :func:`seed_cache` fills the cache from a published snapshot of answers (a directory, or a version of the
training-data dataset on the Hub) before the first run. BGC ``npc_status``: ``classified`` (at least one compound
with a term), ``no_npc_term``, ``no_structure`` (no parsable structure), ``unlabelled`` (no JSON entry).
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path

import pandas as pd

from sanntis.utils import cache_dir, http_request, sha256_file, write_json

log = logging.getLogger(__name__)

API = "https://npclassifier.gnps2.org"
ONTOLOGY_VERSION = "npclassifier_index_v1_250fd53c"
LEVELS = (("pathway", "pathway_results"), ("superclass", "superclass_results"), ("class", "class_results"))
TERMS, EDGES = "np_terms.tsv", "np_term_edges.tsv"


class LabelError(RuntimeError):
    pass


def _data(name: str) -> Path:
    return Path(str(files("sanntis") / "data" / "npc" / name))


def vocabulary() -> set[tuple[str, str]]:
    """(level, name) of every term in the frozen vocabulary."""
    t = pd.read_csv(_data(TERMS), sep="\t", dtype=str)
    return set(zip(t["level"], t["name"], strict=True))


def superclass_parents() -> dict[str, list[str]]:
    """superclass → parent pathways (NPClassifier ontology)."""
    terms = pd.read_csv(_data(TERMS), sep="\t", dtype=str)
    name = dict(zip(terms["np_term_id"], terms["name"], strict=True))
    e = pd.read_csv(_data(EDGES), sep="\t", dtype=str)
    e = e[(e["child_level"] == "superclass") & (e["parent_level"] == "pathway")]
    out: dict[str, set[str]] = {}
    for c, p in zip(e["child_np_term_id"], e["parent_np_term_id"], strict=True):
        out.setdefault(name[c], set()).add(name[p])
    return {k: sorted(v) for k, v in sorted(out.items())}


def ontology() -> dict:
    return {"version": ONTOLOGY_VERSION, "files": {f: sha256_file(_data(f)) for f in (TERMS, EDGES)}}


# ── structures ───────────────────────────────────────────────────────────────


def normalise_structure(smiles: str) -> tuple[dict | None, str | None]:
    """SMILES → ({inchikey, skeleton, smiles}, None) or (None, failure reason)."""
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import inchi
        from rdkit.Chem.MolStandardize import rdMolStandardize
    except ImportError as e:  # pragma: no cover
        raise LabelError("NPC labelling needs RDKit: pip install 'sanntis[train]'") from e

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "smiles_parse_failed"
    mol = rdMolStandardize.LargestFragmentChooser().choose(mol)
    mol = rdMolStandardize.Uncharger().uncharge(mol)
    try:
        mol = rdMolStandardize.TautomerEnumerator().Canonicalize(mol)
    except Exception:
        return None, "tautomer_canonicalisation_failed"
    for a in mol.GetAtoms():
        a.SetIsotope(0)
    try:
        ik = inchi.MolToInchiKey(mol)
    except Exception:
        ik = None
    if not ik:
        return None, "inchikey_failed"
    return {"inchikey": ik, "skeleton": ik.split("-")[0], "smiles": Chem.MolToSmiles(mol)}, None


# ── NPClassifier API ─────────────────────────────────────────────────────────


class NPClassifierClient:
    """Cached NPClassifier API client. 200 / 404 / 500 answers are cached (the service answers 500 deterministically
    for some inputs); connection errors are retried and then raised, never cached."""

    def __init__(self, cache: Path | None = None, *, min_interval: float = 1.0, attempts: int = 4):
        self.dir = Path(cache) if cache else cache_dir("npclassifier")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.min_interval, self.attempts = min_interval, attempts
        self._last = 0.0
        self.requests = 0

    def _path(self, key: str) -> Path:
        return self.dir / f"{hashlib.sha1(key.encode()).hexdigest()}.json"

    def _get(self, url: str, key: str) -> dict:
        p = self._path(key)
        if p.exists():
            return json.loads(p.read_text())
        status, body, err = None, "", None
        for i in range(self.attempts):
            time.sleep(max(0.0, self._last + self.min_interval - time.time()))
            self._last = time.time()
            self.requests += 1
            try:
                with urllib.request.urlopen(http_request(url), timeout=120) as r:
                    status, body = r.status, r.read().decode()
            except urllib.error.HTTPError as e:
                status, body = e.code, e.read().decode(errors="replace")
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                status, err = None, e
            if status in (200, 404) or (status == 500 and i >= 2):
                break
            time.sleep(min(60, 2 ** i))
        if status is None:
            raise LabelError(f"NPClassifier API unreachable ({url[:100]}): {err}")
        rec = {"key": key, "status": status, "body": body,
               "fetched_at": datetime.now(UTC).isoformat(timespec="seconds")}
        write_json(p, rec)
        return rec

    def classify(self, smiles: str) -> dict | None:
        """Parsed response for one SMILES, or None when the service gives no classification."""
        rec = self._get(f"{API}/classify?smiles={urllib.parse.quote(smiles, safe='')}", smiles)
        if rec["status"] != 200:
            return None
        try:
            return json.loads(rec["body"])
        except json.JSONDecodeError:
            return None

    def model_metadata(self) -> str:
        """``models PATHWAY v1, ...`` from the API model metadata (cached)."""
        rec = self._get(f"{API}/model/metadata", "model/metadata")
        try:
            d = json.loads(rec["body"])
            return "models " + ", ".join(f"{k}={v['model_spec']['name']} v{v['model_spec']['version']}"
                                         for k, v in sorted(d.items()))
        except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
            return f"unknown (metadata status {rec['status']})"


def seed_cache(spec: str | Path, cache: Path | None = None) -> dict:
    """Copy published NPClassifier answers into the cache. ``spec`` is a directory of cache files, or
    ``[<repo>@]<version>`` of the training-data dataset on the Hub (``npclassifier/<version>/``, tag
    ``npclassifier-<version>``). Answers already in the cache are kept. Every file is checked to be a cache record
    whose name is the sha1 of its key. Returns the provenance: source, version, files seen and files added."""
    dst = Path(cache) if cache else cache_dir("npclassifier")
    dst.mkdir(parents=True, exist_ok=True)
    p = Path(spec)
    if p.is_dir():
        src, version = p, None
    else:
        from sanntis.negpool.pool import hub_dataset_dir

        src, version = hub_dataset_dir(str(spec), "npclassifier")
    files = sorted(src.glob("*.json"))
    if not files:
        raise LabelError(f"{src}: no NPClassifier cache files (*.json)")
    added = 0
    for f in files:
        try:
            rec = json.loads(f.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise LabelError(f"{f}: not a cache record ({e})") from None
        if not isinstance(rec, dict) or not {"key", "status", "body"} <= set(rec) or not isinstance(rec["key"], str):
            raise LabelError(f"{f}: not a cache record (key, status and body expected)")
        if f.name != f"{hashlib.sha1(rec['key'].encode()).hexdigest()}.json":
            raise LabelError(f"{f}: file name is not the sha1 of its key")
        t = dst / f.name
        if t.exists():
            continue
        write_json(t, {k: rec[k] for k in ("key", "status", "body", "fetched_at") if k in rec})
        added += 1
    log.info("NPClassifier cache: %d answers from %s, %d added to %s", len(files), src, added, dst)
    return {"source": str(src), "version": version, "files": len(files), "added": added}


# ── labels ───────────────────────────────────────────────────────────────────


def _union(xs) -> list[str]:
    return sorted(set().union(*xs)) if xs else []


def label_entries(entries: dict[str, dict], client: NPClassifierClient | None = None
                  ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """MIBiG-format entries → (per-BGC labels indexed by accession: npc_status, pathway, superclass, npc_class;
    per-compound table; provenance)."""
    client = client or NPClassifierClient()
    vocab = vocabulary()
    counts: Counter = Counter()
    comps = []
    for acc in sorted(entries):
        for j, c in enumerate(entries[acc].get("compounds") or [], start=1):
            s = (c.get("structure") or "").strip()
            rec, why = normalise_structure(s) if s else (None, "no_structure")
            comps.append({"accession": acc, "compound_index": j, "compound_name": c.get("name") or "",
                          "structure": s, "status": why or "parsed", **(rec or {})})
            counts[f"compounds_{why or 'parsed'}"] += 1
    skel: dict[str, str] = {}  # skeleton → SMILES of its lowest InChIKey
    for r in sorted((r for r in comps if r["status"] == "parsed"), key=lambda r: r["inchikey"]):
        skel.setdefault(r["skeleton"], r["smiles"])
    terms: dict[str, dict[str, set[str]] | None] = {}
    unmapped: Counter = Counter()
    for i, (sk, smi) in enumerate(sorted(skel.items())):
        res = client.classify(smi)
        if res is None:
            terms[sk] = None
            counts["skeletons_classifier_failed"] += 1
            continue
        t: dict[str, set[str]] = {}
        for lv, key in LEVELS:
            names = set(res.get(key) or [])
            unmapped.update(f"{lv}:{n}" for n in names if (lv, n) not in vocab)
            t[lv] = {n for n in names if (lv, n) in vocab}
        terms[sk] = t
        if (i + 1) % 500 == 0:
            log.info("NPClassifier: %d / %d skeletons", i + 1, len(skel))
    for r in comps:
        t = terms.get(r.get("skeleton") or "")
        r["pathway"], r["superclass"], r["npc_class"] = ([sorted(t[lv]) for lv, _ in LEVELS] if t
                                                         else ([], [], []))
        if r["status"] == "parsed":
            r["status"] = ("classified" if t and any(t.values()) else
                           "classifier_failed" if t is None else "no_npc_term")
    cdf = pd.DataFrame(comps, columns=["accession", "compound_index", "compound_name", "structure", "status",
                                       "inchikey", "skeleton", "smiles", "pathway", "superclass", "npc_class"])
    rows = []
    for acc in sorted(entries):
        g = cdf[cdf["accession"] == acc]
        ok = g[g["status"] == "classified"]
        status = ("classified" if len(ok) else
                  "no_structure" if not g["status"].isin(["no_npc_term", "classifier_failed"]).any() else
                  "no_npc_term")
        rows.append({"accession": acc, "npc_status": status, "pathway": _union([set(x) for x in ok["pathway"]]),
                     "superclass": _union([set(x) for x in ok["superclass"]]),
                     "npc_class": _union([set(x) for x in ok["npc_class"]])})
    bgc = pd.DataFrame(rows, columns=["accession", "npc_status", "pathway", "superclass", "npc_class"])
    version = client.model_metadata() if skel else "not queried (no structures)"
    prov = {"classifier": f"NPClassifier API ({API})", "classifier_version": version,
            "ontology": ontology(), "skeletons": len(skel), "api_requests": client.requests,
            "counts": dict(sorted(counts.items())), "bgc_status": bgc["npc_status"].value_counts().to_dict(),
            "unmapped_terms": dict(unmapped.most_common())}
    return bgc.set_index("accession"), cdf, prov
