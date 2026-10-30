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

"""Training steps, run in the order of :data:`STEPS` by :mod:`sanntis.train.orchestrate`.

::

    collect → annotate → entities → cv (units: split × seed) → tune → finalize (units: seeds) → report → package

The report reads the pooled out-of-fold (``mc`` split) predictions of ``cv`` / ``tune`` only.

Run-dir layout: ``annotate/positives.parquet``, ``entities/entities.parquet``, ``cv/<unit>/``, ``tune/``,
``finalize/``, ``report/``, ``bundle/`` (the production bundle).
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

import pandas as pd

from sanntis.annotate.ips6 import annotate, default_datadir
from sanntis.config import load_recipe
from sanntis.features.vocab import Vocab
from sanntis.io.fasta import iter_fasta
from sanntis.io.ips_tsv import entry_types_from_entries, read_ips_tsv, signature_map_from_entries, tokens_by_key
from sanntis.model.bundle import SanntisModel
from sanntis.predict import Predictor
from sanntis.runner.tasks import Unit
from sanntis.train.collect import CollectOptions, collect
from sanntis.train.cv import run_cv_unit
from sanntis.train.entities import entity_table, negative_entities, pool_frames, positive_entities
from sanntis.train.finalize import build_bundle, run_seed_unit
from sanntis.train.labels import superclass_parents
from sanntis.train.ncbi import LineageCache, lineage_group
from sanntis.train.orchestrate import RunContext, Step
from sanntis.train.report import write_report
from sanntis.train.tune import run_tune

log = logging.getLogger(__name__)

# ── helpers ──────────────────────────────────────────────────────────────────


def _lists(df: pd.DataFrame, col: str = "cds_tokens") -> pd.DataFrame:
    df[col] = df[col].map(lambda c: [list(g) for g in c])
    return df


def load_entities(run: Path) -> pd.DataFrame:
    t = _lists(pd.read_parquet(Path(run) / "entities" / "entities.parquet"))
    t["classes"] = t["classes"].map(tuple)
    return t


def entries_json(ctx: RunContext) -> Path:
    return Path(ctx.cfg.entries_json or (Path(ctx.cfg.ips_datadir or default_datadir())
                                         / "interpro" / load_recipe().ips.interpro / "entries.json"))


def labels_and_parents(run: Path):
    """NPC labels of the collected loci (from their compounds, ``collect``) and the superclass → pathway map."""
    loci = pd.read_parquet(Path(run) / "collect" / "loci.parquet")
    lab = pd.DataFrame({"accession": loci["entity_id"], "npc_status": loci["npc_status"].fillna("unlabelled"),
                        "pathway": loci["npc_pathway"].map(lambda x: list(x) if x is not None else []),
                        "superclass": loci["npc_superclass"].map(lambda x: list(x) if x is not None else [])})
    return lab.set_index("accession"), superclass_parents()


def lineage_of(run: Path) -> dict[str, str | None]:
    """entity → prokaryotes / fungi / plants / other eukaryotes, from the MIBiG taxid (NCBI lineage, cached)."""
    loci = pd.read_parquet(Path(run) / "collect" / "loci.parquet")
    km = Path(run) / "collect" / "kingdom_map.tsv"
    kmap = pd.read_csv(km, sep="\t").set_index("accession") if km.exists() else pd.DataFrame()
    euk = [e for e, k in zip(loci["entity_id"], loci["kingdom"], strict=True) if k == "euk" and e in kmap.index]
    tid = {e: int(kmap.at[e, "ncbi_tax_id"]) for e in euk if int(kmap.at[e, "ncbi_tax_id"]) > 0}
    lin = LineageCache().resolve(tid.values()) if tid else {}
    return {e: "prokaryotes" if k == "proka" else (lineage_group(lin.get(tid[e], {})) if e in tid else None)
            for e, k in zip(loci["entity_id"], loci["kingdom"], strict=True)}


def pool_dir(run: Path) -> Path:
    prov = json.loads((Path(run) / "collect" / "provenance.json").read_text())
    if "negatives" not in prov:
        raise RuntimeError("no negative pool: pass --negatives <pool dir | hub revision>")
    return Path(prov["negatives"]["path"])


# ── collect ──────────────────────────────────────────────────────────────────


def _collect_inputs(ctx: RunContext) -> dict:
    c = ctx.cfg
    return {"mibig": c.mibig, "external": [[Path(g).resolve(), Path(j).resolve() if j else None]
                                           for g, j in c.external], "negatives": c.negatives,
            **({"npc_cache": c.npc_cache} if c.npc_cache else {})}  # absent when unset: keys of older runs hold


def _collect_run(ctx: RunContext) -> dict:
    opts = CollectOptions(mibig=ctx.cfg.mibig, external=ctx.cfg.external, negatives=ctx.cfg.negatives,
                          npc_cache=ctx.cfg.npc_cache)
    return collect(ctx.step_dir("collect"), opts)


# ── annotate ─────────────────────────────────────────────────────────────────


def _annotate_inputs(ctx: RunContext) -> dict:
    return {"ips_tsv": list(ctx.cfg.train_ips_tsv),
            "entries": entries_json(ctx)}


def _annotate_run(ctx: RunContext) -> dict:
    """Positives' proteins → IPS6 (managed, md5 cache) or precomputed TSVs (IPS 5.76+ / 6.x) → tokens per CDS."""
    run = ctx.run_dir
    out = ctx.step_dir("annotate")
    cds = pd.read_parquet(run / "collect" / "cds.parquet").sort_values(["entity_id", "cds_idx"])
    loci = pd.read_parquet(run / "collect" / "loci.parquet")
    if ctx.cfg.train_ips_tsv:
        h = read_ips_tsv(ctx.cfg.train_ips_tsv)
        hits, source = h.hits, f"precomputed TSV (IPS {h.version})"
    else:
        proteins = dict(iter_fasta(run / "collect" / "proteins.faa"))
        ann = annotate(proteins, ctx.cfg.nextflow, out / "ips6_run", datadir=ctx.cfg.ips_datadir)
        hits, source = ann.hits, "InterProScan 6 (managed)"
    sm = signature_map_from_entries(entries_json(ctx))
    by_md5 = tokens_by_key(hits, sm, key="md5")
    pfam = hits[hits["app"] == "pfam"].groupby("md5")["signature"].agg(set).to_dict()
    toks = cds.groupby("entity_id", sort=False)["md5"].agg(lambda s: [by_md5.get(m, []) for m in s])
    pf = cds.groupby("entity_id", sort=False)["md5"].agg(lambda s: sorted(set().union(*(pfam.get(m, set())
                                                                                          for m in s))))
    lin = lineage_of(run)
    pos = pd.DataFrame({"entity_id": loci["entity_id"], "source": loci["source"], "kingdom": loci["kingdom"],
                        "class_raw": loci["class_raw"].replace("", None), "status": loci["status"],
                        "length": loci["length"], "lineage": loci["entity_id"].map(lin),
                        "cds_tokens": loci["entity_id"].map(toks), "pfam": loci["entity_id"].map(pf)})
    pos.to_parquet(out / "positives.parquet", index=False)
    app_counts = Counter(hits.loc[hits["md5"].isin(set(cds["md5"])), "app"])
    n_cds = len(cds)
    with_tok = sum(1 for m in cds["md5"] if by_md5.get(m))
    ref = {"frac_cds_with_tokens": round(with_tok / n_cds, 4),
           "hits_per_cds": {a: round(c / n_cds, 4) for a, c in sorted(app_counts.items())}}
    (out / "coverage_positives.json").write_text(json.dumps(ref, indent=1))
    return {"loci": len(pos), "cds": n_cds, "cds_with_tokens": with_tok, "annotation": source}


# ── entities ─────────────────────────────────────────────────────────────────


def _build(pos: pd.DataFrame, pool, recipe):
    r = recipe
    pe, ps = positive_entities(_lists(pos.copy()), dice_threshold=r.dedup.dice_threshold,
                               max_locus_bp=r.dedup.max_locus_bp)
    meta = pos.set_index("entity_id")
    pe["lineage"] = pe["entity_id"].map(meta["lineage"])
    pe["pfam"] = pe["entity_id"].map(meta["pfam"]).map(list)
    c = r.compose
    ne, ns = negative_entities(pool, pe, chunks_per_positive=c.chunks_per_positive, clip=c.chunk_len_clip,
                               background_min_cds=c.background_min_cds, seed=c.seed)
    t = entity_table(pe, ne.assign(lineage=None, pfam=None))
    t["classes"] = t["classes"].map(list)
    return t, {**ps, **ns}


def _entities_inputs(ctx: RunContext) -> dict:
    return {"recipe": load_recipe().model_dump()}


def _entities_run(ctx: RunContext) -> dict:
    recipe = load_recipe()
    out = ctx.step_dir("entities")
    pos = pd.read_parquet(ctx.run_dir / "annotate" / "positives.parquet")
    pool = pool_frames(pool_dir(ctx.run_dir))
    t, st = _build(pos, pool, recipe)
    t.to_parquet(out / "entities.parquet", index=False)
    (out / "stats.json").write_text(json.dumps({"production": st}, indent=1, default=str))
    return {"production_positives": st["representatives"]}


# ── cv / tune ────────────────────────────────────────────────────────────────


def cv_unit(run: Path, unit_id: str) -> None:
    k, s = unit_id.removeprefix("split").split("_seed")
    run_cv_unit(load_entities(run), int(k), int(s), Path(run) / "cv" / unit_id)


def _cv_run(ctx: RunContext) -> dict:
    recipe = load_recipe()
    label = "gpu" if ctx.cfg.gpu else "train"
    units = [Unit(f"split{k}_seed0", label) for k in range(recipe.splits.n_splits)]
    ctx.dispatch("cv", units)
    if ctx.dry_run:
        return {}
    m = [json.loads((ctx.run_dir / "cv" / u.unit_id / "metrics.json").read_text()) for u in units]
    return {"best_epochs": [x["best_epoch"] for x in m], "val_orf_aucpr": [round(x["val_orf_aucpr"], 4) for x in m]}


def _tune_run(ctx: RunContext) -> dict:
    lab, par = labels_and_parents(ctx.run_dir)
    res = run_tune(ctx.run_dir, load_entities(ctx.run_dir), labels=lab, parents=par,
                   entry_types=entry_types_from_entries(entries_json(ctx)))
    return {k: res[k] for k in ("epochs", "t_star", "floor_met")}


# ── fits ─────────────────────────────────────────────────────────────────────


def finalize_unit(run: Path, unit_id: str) -> None:
    tune = json.loads((Path(run) / "tune" / "tune.json").read_text())
    run_seed_unit(load_entities(run), int(unit_id.removeprefix("seed")), tune["epochs"], Path(run) / "finalize")


def _finalize_run(ctx: RunContext) -> dict:
    label = "gpu" if ctx.cfg.gpu else "train"
    ctx.dispatch("finalize", [Unit(f"seed{s}", label) for s in load_recipe().train.seeds])
    return {}


def coverage_reference(run: Path) -> dict | None:
    """QC reference of the bundle (``sanntis.qc``), per kingdom: genome-wide coverage of the negative pool (its
    manifest) plus the ``<unk>`` share of the pool tokens under the bundle vocabulary. Pools without coverage fall
    back to the positives (BGC proteins carry more hits than a genome, so the per-app rates are only a rough
    reference)."""
    pd_ = pool_dir(run)
    pool_cov = json.loads((pd_ / "manifest.json").read_text()).get("coverage")
    if not pool_cov:
        cov = run / "annotate" / "coverage_positives.json"
        return json.loads(cov.read_text()) if cov.exists() else None
    vocab = Vocab.load(next(iter(sorted((run / "finalize").glob("vocab-seed*.json")))))
    kingdom = pd.read_parquet(pd_ / "genomes.parquet").set_index("genome")["kingdom"]
    cds = pd.read_parquet(pd_ / "cds_tokens.parquet", columns=["genome", "tokens"]).dropna(subset=["tokens"])
    out: dict = {"source": "negative pool"}
    for k, ref in pool_cov.items():
        n = unk = 0
        for ts in cds.loc[cds["genome"].map(kingdom).eq(k), "tokens"]:
            n += len(ts)
            unk += sum(1 for t in ts if t not in vocab.stoi)
        out[k] = {"frac_cds_with_tokens": ref["frac_cds_with_tokens"],
                  "frac_unk_tokens": round(unk / n, 4) if n else 0.0, "hits_per_cds": ref["hits_per_cds"]}
    return out


def _bundle(ctx: RunContext, report_dir: Path, extra: dict):
    run = ctx.run_dir
    tune = json.loads((run / "tune" / "tune.json").read_text())
    lab, par = labels_and_parents(run)
    oof = _lists(pd.read_parquet(run / "tune" / "oof_regions.parquet"))
    prov = json.loads((run / "collect" / "provenance.json").read_text())
    return build_bundle(load_entities(run), tune, run / "finalize", run / "bundle",
                        signature_map=signature_map_from_entries(entries_json(ctx)), labels=lab, parents=par,
                        entry_types=entry_types_from_entries(entries_json(ctx)),
                        oof_regions=oof, report_dir=report_dir, coverage_reference=coverage_reference(run),
                        provenance={"fit": "production", "mibig": prov.get("mibig", {}),
                                    "external": prov.get("external"), "negatives": prov.get("negatives", {}),
                                    "npc_labels": prov.get("npc_labels"), **extra})


# ── report / package ─────────────────────────────────────────────────────────


def _mibig_cell(m: dict) -> str:
    if not m:
        return "none (external data only)"
    files = " · ".join(f"`{m[k]['name']}` (sha256 {m[k]['sha256'][:12]}…)" for k in ("json", "gbk") if k in m)
    sc = m.get("status_counts", {})
    return (f"{m.get('release', '?')}: {m.get('entries', '?')} entries "
            f"({', '.join(f'{k} {v}' for k, v in sc.items())}); active entries used; {files}")


def _external_cell(ext: list | None) -> str:
    if not ext:
        return "none"
    return " · ".join(f"`{e['gbk']['name']}`" + (f" + `{e['json']['name']}`" if e.get("json") else " (GBK only)")
                      + f": {e['n']} BGCs" for e in ext)


def _npc_cell(n: dict | None) -> str:
    if not n:
        return "none (no JSON entries)"
    st = n.get("bgc_status", {})
    return (f"{n['classifier']}, {n['classifier_version']}; ontology {n['ontology']['version']}; "
            f"{n['skeletons']} skeletons; BGCs {', '.join(f'{k} {v}' for k, v in st.items())}")


def _pool_cell(n: dict) -> str:
    if not n:
        return "none"
    sha = n.get("manifest_sha256", "")
    return f"`{n.get('version', '?')}` (manifest sha256 {sha[:12]}…), {n.get('path', '')}"


def _report_run(ctx: RunContext) -> dict:
    run = ctx.run_dir
    st = json.loads((run / "entities" / "stats.json").read_text())["production"]
    prov = json.loads((run / "collect" / "provenance.json").read_text())
    meta = {"Run": str(run), "MIBiG": _mibig_cell(prov.get("mibig") or {}),
            "External data": _external_cell(prov.get("external")),
            "NPC labels": _npc_cell(prov.get("npc_labels")),
            "Negative pool": _pool_cell(prov.get("negatives") or {}),
            "Training pool (CV)": f"{st['representatives']} positives (proka {st['representatives_proka']}"
                                  f" / euk {st['representatives_euk']})",
            "Features": f"InterProScan {load_recipe().ips.version}, InterPro {load_recipe().ips.interpro}, "
                        f"{len(load_recipe().ips.apps)} apps"}
    rep = write_report(run, load_entities(run), metadata=meta)
    return {"report": str(rep / "report.md")}


def _package_run(ctx: RunContext) -> dict:
    """Production bundle with the report, then a smoke test: load it and predict one training locus."""
    run = ctx.run_dir
    tune = json.loads((run / "tune" / "tune.json").read_text())
    perf = {"Validation (CV, pooled OOF)": tune["validation"]["all"]}
    keep = ("precision_t", "recall_t", "recall_macro_t", "f1_t", "f1_macro_t", "bgc_precision", "bgc_recall",
            "bgc_macro_recall", "f1_macro", "f2_macro", "n_gt")
    perf = {k: {m: round(float(v[m]), 4) for m in keep if m in v} for k, v in perf.items()}
    _bundle(ctx, report_dir=run / "report", extra={"performance": perf})
    m = SanntisModel.from_pretrained(run / "bundle", allow_ips_mismatch=True)
    t = load_entities(run)
    p = t[t["kind"] == "positive"].iloc[0]
    n = len(p["cds_tokens"])
    cds = pd.DataFrame({"record_id": p["entity_id"], "cds_idx": range(n), "locus_tag": [f"c{i}" for i in range(n)],
                        "start": [i * 1000 for i in range(n)], "end": [i * 1000 + 900 for i in range(n)],
                        "tokens": p["cds_tokens"]})
    pred = Predictor(m, kingdom=p["kingdom"]).predict(cds)
    return {"bundle": str(run / "bundle"), "smoke_regions": len(pred.regions), "t_star": m.config.decode.threshold}


# ── step list ────────────────────────────────────────────────────────────────


def _no_inputs(ctx: RunContext) -> dict:
    return {}  # depends only on the previous steps' keys


STEPS = [
    Step("collect", _collect_run, _collect_inputs),
    Step("annotate", _annotate_run, _annotate_inputs, deps=("collect",)),
    Step("entities", _entities_run, _entities_inputs, deps=("annotate",)),
    Step("cv", _cv_run, _no_inputs, deps=("entities",), fanout=True),
    Step("tune", _tune_run, _no_inputs, deps=("cv",)),
    Step("finalize", _finalize_run, _no_inputs, deps=("tune",), fanout=True),
    Step("report", _report_run, _no_inputs, deps=("tune",)),
    Step("package", _package_run, _no_inputs, deps=("finalize", "report")),
]
