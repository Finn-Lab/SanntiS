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

"""InterProScan TSV (IPS 5.76+ or 6.x) → normalised hits → per-CDS token lists.

Both major versions write the same 15-column TSV and the same signature accessions; they differ in the analysis
names (col 4) and in col 10 (IPS 5: match status ``T``; IPS 6: the analysis name again). Neither records the IPS
version or the InterPro release.

Tokens come from the bundle's ``signature_map`` (signature → InterPro entry at the bundle's release), never from
the TSV's own IPR column: a signature absent from the map keeps its own accession.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from sanntis.config import load_recipe
from sanntis.io.fasta import open_text

log = logging.getLogger(__name__)

COLUMNS = ["protein_id", "md5", "length", "analysis", "signature", "description", "start", "end", "score",
           "col10", "date", "ipr", "ipr_description", "go", "pathways"]

# Normalised analysis name (case-folded; hyphens, underscores and spaces removed) → canonical app name.
ALIASES = {
    "pfam": "pfam", "ncbifam": "ncbifam", "tigrfam": "ncbifam", "sfld": "sfld", "superfamily": "superfamily",
    "gene3d": "gene3d", "cathgene3d": "gene3d", "hamap": "hamap", "coils": "coils", "cdd": "cdd",
    "prints": "prints", "pirsf": "pirsf", "prositeprofiles": "prositeprofiles", "prositepatterns": "prositepatterns",
    "mobidblite": "mobidblite", "smart": "smart",
    # analyses that are never allowed (dropped and counted)
    "panther": "panther", "cathfunfam": "funfam", "funfam": "funfam", "antifam": "antifam", "pirsr": "pirsr",
    "phobius": "phobius", "signalp": "signalp", "signalpeuk": "signalp", "signalpprok": "signalp",
    "signalpgrampositive": "signalp", "signalpgramnegative": "signalp", "tmhmm": "tmhmm", "deeptmhmm": "tmhmm",
    "deeptmhmm2": "tmhmm", "tmbed": "tmbed", "interpron": "interpron", "productdb": "productdb",
}
IPS5_NAMES = {"Pfam", "NCBIfam", "TIGRFAM", "SFLD", "SUPERFAMILY", "Gene3D", "Hamap", "Coils", "CDD", "PRINTS",
              "PIRSF", "ProSiteProfiles", "ProSitePatterns", "MobiDBLite", "SMART", "PANTHER", "FunFam", "AntiFam",
              "PIRSR", "Phobius", "SignalP_EUK", "SignalP_GRAM_POSITIVE", "SignalP_GRAM_NEGATIVE", "TMHMM"}
IPS6_NAMES = {"Pfam", "NCBIFAM", "SFLD", "SUPERFAMILY", "CATH-Gene3D", "HAMAP", "COILS", "CDD", "PRINTS", "PIRSF",
              "PROSITE profiles", "PROSITE patterns", "MobiDB-lite", "SMART", "PANTHER", "CATH-FunFam", "AntiFam",
              "PIRSR", "Phobius", "SignalP-Euk", "SignalP-Prok", "DeepTMHMM", "DeepTMHMM2", "TMbed", "InterPro-N"}


class IpsTsvError(ValueError):
    pass


def normalise_name(name: str) -> str:
    return re.sub(r"[-_\s]", "", name).casefold()


def canonical_app(name: str) -> str | None:
    return ALIASES.get(normalise_name(name))


@dataclass
class IpsHits:
    """Hits of the allowed apps, plus what was seen and dropped (run metadata, coverage QC)."""

    hits: pd.DataFrame  # protein_id, md5, app, signature, start, end, ipr
    version: str  # "5" | "6" | "unknown" (no rows)
    n_proteins: int
    dropped: Counter = field(default_factory=Counter)  # analysis → rows dropped
    per_app: Counter = field(default_factory=Counter)  # canonical app → rows kept


def _iter_rows(paths: Iterable[Path]):
    for p in paths:
        with open_text(p) as fh:
            for line in fh:
                if not line.strip():
                    continue
                s = line.rstrip("\n").split("\t")
                if len(s) < 11:
                    raise IpsTsvError(f"{p}: expected >= 11 tab-separated columns, got {len(s)}")
                yield s


def detect_version(analyses: set[str], col10: set[str]) -> str:
    """IPS 5: col 10 is the match status (T/F); IPS 6: col 10 repeats the analysis name."""
    if not analyses:
        return "unknown"
    status_like = col10 <= {"T", "F"}
    v6_names = analyses & (IPS6_NAMES - IPS5_NAMES)
    v5_names = analyses & (IPS5_NAMES - IPS6_NAMES)
    if status_like and not v6_names:
        return "5"
    if not status_like and col10 <= analyses and not v5_names:
        return "6"
    raise IpsTsvError(f"cannot determine the InterProScan major version (mixed file?): analyses={sorted(analyses)}, "
                      f"col10={sorted(col10)[:10]}")


def read_ips_tsv(paths: Path | Sequence[Path]) -> IpsHits:
    """Parse one or more TSVs (``.gz`` ok); keep the recipe's apps."""
    paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
    allowed = set(load_recipe().ips.app_names)
    rows = []
    analyses: set[str] = set()
    col10: set[str] = set()
    proteins: set[str] = set()
    dropped: Counter = Counter()
    unknown: Counter = Counter()
    for s in _iter_rows(paths):
        analyses.add(s[3])
        col10.add(s[9])
        proteins.add(s[0])
        app = canonical_app(s[3])
        if app is None:
            unknown[s[3]] += 1
        if app not in allowed:
            dropped[s[3]] += 1
            continue
        ipr = s[11] if len(s) > 11 and s[11] not in ("-", "") else None
        rows.append((s[0], s[1].lower(), app, s[4], int(s[6]), int(s[7]), ipr))  # IPS6 writes upper-case md5
    version = detect_version(analyses, col10)
    if unknown:
        log.warning("unknown InterProScan analyses dropped: %s", dict(unknown))
    if dropped:
        log.info("analyses outside the allowed apps dropped: %s", dict(dropped))
    hits = pd.DataFrame(rows, columns=["protein_id", "md5", "app", "signature", "start", "end", "ipr"])
    return IpsHits(hits=hits, version=version, n_proteins=len(proteins), dropped=dropped,
                   per_app=Counter(hits["app"]))


def check_apps(h: IpsHits, *, allow_incomplete: bool = False) -> list[str]:
    """Required apps with no hit in a file of >= N proteins stop the run; other missing apps warn."""
    r = load_recipe().ips
    missing = [a for a in r.app_names if h.per_app.get(a, 0) == 0]
    if not missing or h.n_proteins == 0:
        return missing
    fatal = [a for a in missing if a in r.required_apps and h.n_proteins >= r.required_apps_min_proteins]
    msg = f"InterProScan output has no hits for {missing} ({h.n_proteins} proteins)"
    if fatal and not allow_incomplete:
        raise IpsTsvError(msg + f"; {fatal} are required (was the TSV run with all 14 apps? "
                          "--allow-incomplete-ips overrides)")
    log.warning(msg)
    return missing


def token_of(hits: pd.DataFrame, signature_map: pd.DataFrame) -> pd.Series:
    """Signature → InterPro entry from the map; unmapped or unintegrated signatures keep their accession."""
    sm = signature_map.dropna(subset=["entry"])
    m = dict(zip(sm["signature"], sm["entry"], strict=True))
    return hits["signature"].map(m).fillna(hits["signature"])


def tokens_by_key(hits: pd.DataFrame, signature_map, key: str = "protein_id") -> dict[str, list[str]]:
    """key (protein id or md5) → tokens ordered by hit start (ties by token)."""
    if hits.empty:
        return {}
    df = pd.DataFrame({"k": hits[key].to_numpy(), "start": hits["start"].to_numpy(),
                       "tok": token_of(hits, signature_map).to_numpy()})
    out: dict[str, list[str]] = {}
    for k, _start, t in sorted(df.itertuples(index=False, name=None)):
        out.setdefault(k, []).append(t)
    return out


def signature_map_from_entries(entries_json: Path) -> pd.DataFrame:
    """Build ``signature_map`` from an InterPro release's ``entries.json`` (as shipped with IPS6 data)."""
    allowed = set(load_recipe().ips.app_names)
    d = json.loads(Path(entries_json).read_text())
    rows = []
    for acc, v in d.items():
        app = canonical_app(v.get("database", ""))
        if app in allowed:
            rows.append((acc, app, v.get("integrated")))
    return (pd.DataFrame(rows, columns=["signature", "app", "entry"])
            .sort_values("signature").reset_index(drop=True))


def entry_types_from_entries(entries_json: Path) -> dict[str, str]:
    """InterPro entry → type (Family, Domain, Homologous_superfamily, ...) from an InterPro ``entries.json``."""
    d = json.loads(Path(entries_json).read_text())
    return {acc: v["type"] for acc, v in d.items() if v.get("database") == "InterPro" and v.get("type")}


@dataclass
class JoinStats:
    by_id: int = 0  # CDS joined by protein id (TSV col 1 = locus tag)
    by_md5: int = 0  # CDS joined by sequence md5 only
    id_md5_conflicts: int = 0  # id matched but the TSV md5 differs from the CDS translation
    tsv_proteins: int = 0  # proteins with >= 1 kept hit in the TSV
    tsv_unmatched: int = 0  # of those, matched to no CDS of this input


def join_tokens(cds: pd.DataFrame, hits: pd.DataFrame,
                signature_map) -> tuple[list[list[str]], list[tuple[str, str] | None], JoinStats]:
    """Per-CDS tokens for one input: by protein id (``locus_tag``) when the TSV knows it, else by md5. A CDS whose
    md5 is in the TSV takes the md5 tokens (same sequence, whatever the id). Also returns the key each CDS joined on,
    ``("md5" | "protein_id", value)`` or None."""
    st = JoinStats()
    if hits.empty:
        return [[] for _ in range(len(cds))], [None] * len(cds), st
    by_id = tokens_by_key(hits, signature_map, "protein_id")
    by_md5 = tokens_by_key(hits, signature_map, "md5")
    id_md5 = dict(zip(hits["protein_id"], hits["md5"], strict=True))
    used_ids, used_md5 = set(), set()
    out, keys = [], []
    for tag, m in zip(cds["locus_tag"], cds["md5"], strict=True):
        if m in by_md5 and (tag not in by_id or id_md5[tag] == m):
            st.by_md5 += tag not in by_id
            st.by_id += tag in by_id
            used_md5.add(m)
            used_ids.add(tag)
            out.append(by_md5[m])
            keys.append(("md5", m))
        elif tag in by_id:
            st.by_id += 1
            st.id_md5_conflicts += 1
            used_ids.add(tag)
            out.append(by_id[tag])
            keys.append(("protein_id", tag))
        else:
            out.append([])
            keys.append(None)
    tsv = hits.drop_duplicates("protein_id")
    st.tsv_proteins = len(tsv)
    st.tsv_unmatched = int((~tsv["protein_id"].isin(used_ids) & ~tsv["md5"].isin(used_md5)).sum())
    return out, keys, st
