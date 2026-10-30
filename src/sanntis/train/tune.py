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

"""``tune``: values re-decided per run from the pooled out-of-fold predictions.

- epochs = median best epoch of the CV runs (integer part);
- t* = the sweep threshold with the highest F2_macro among those with ``operating_point.floor_metric`` (BGC
  precision) >= floor and enough regions (``min_pred_ratio``); ties → higher;
- NPC thresholds: CV of the NPC members on loci + OOF regions at t*, thresholds fitted on the pooled predictions;
- NPC pathway key-domain rules (:mod:`sanntis.train.key_domains`): per split fitted on the training loci and
  evaluated on the OOF regions with cross-split thresholds → gated pathways, OOF coverage, false-alarm / catch.

Validation scoring: per split and kingdom, the validation composition (validation positives, chunks and
validation-genome background) with the split's decoded + RF-filtered OOF regions. Bootstrap unit = validation
entity (a prediction belongs to the entity holding most of its ORFs).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sanntis.config import Recipe, load_recipe
from sanntis.train import key_domains as kd
from sanntis.train import metrics as M
from sanntis.train.compose import compose
from sanntis.train.cv import split_data
from sanntis.train.npc_train import (
    PATHWAYS,
    LabelSpace,
    cross_split_thresholds,
    cv_split,
    fit_thresholds,
    labelled,
    locus_examples,
    main_entity,
    region_examples,
)
from sanntis.train.splits import mc_split

log = logging.getLogger(__name__)


@dataclass
class OofSplit:
    split: int
    kingdom: str
    scores: np.ndarray
    label: np.ndarray
    entity: np.ndarray  # entity-table row per ORF
    regions: pd.DataFrame  # this split/kingdom's kept regions (all thresholds)


def load_oof(cv_dir: Path, table: pd.DataFrame, n_splits: int, seed: int = 0) -> list[OofSplit]:
    out = []
    for k in range(n_splits):
        d = Path(cv_dir) / f"split{k}_seed{seed}"
        z = np.load(d / "oof.npz")
        reg = pd.read_parquet(d / "regions.parquet")
        reg = reg[reg["kept"]]
        for kingdom in sorted({x.split("_", 1)[0] for x in z.files}):
            out.append(OofSplit(k, kingdom, z[f"{kingdom}_scores"], z[f"{kingdom}_label"], z[f"{kingdom}_entity"],
                                reg[reg["kingdom"] == kingdom]))
    return out


def score_oof(o: OofSplit, regions: list[tuple[int, int]], table: pd.DataFrame) -> M.Scored:
    orf_bgc = np.where(o.label == 1, o.entity, -1)
    classes = {int(e): tuple(table.at[int(e), "classes"]) for e in np.unique(o.entity[o.label == 1])}

    def unit_of(a: int, b: int) -> str:
        ents, cnt = np.unique(o.entity[a:b], return_counts=True)
        return f"{o.split}:{int(ents[int(np.argmax(cnt))])}"

    s = M.score_contig(regions, orf_bgc, bgc_classes=classes, pred_unit=unit_of)
    s.gt["unit"] = [f"{o.split}:{int(b)}" for b in s.gt["bgc"]]
    s.gt["kingdom"] = o.kingdom
    s.pred["kingdom"] = o.kingdom
    return s


def scored_at(oof: list[OofSplit], t: float, table: pd.DataFrame) -> M.Scored:
    parts = []
    for o in oof:
        r = o.regions[np.isclose(o.regions["threshold"], t)]
        parts.append(score_oof(o, list(zip(r["start"], r["end"], strict=True)), table))
    return M.concat(parts)


def subset(s: M.Scored, kingdom: str) -> M.Scored:
    return M.Scored(s.gt[s.gt["kingdom"] == kingdom].reset_index(drop=True),
                    s.pred[s.pred["kingdom"] == kingdom].reset_index(drop=True))


def sweep(oof: list[OofSplit], table: pd.DataFrame, recipe: Recipe) -> pd.DataFrame:
    rows = []
    for t in M.threshold_grid(recipe.operating_point.grid):
        s = scored_at(oof, t, table)
        rows.append({"threshold": t, "stratum": "all", **M.aggregate(s)})
        for k in sorted(s.gt["kingdom"].unique()):
            rows.append({"threshold": t, "stratum": k, **M.aggregate(subset(s, k))})
    return pd.DataFrame(rows)


def oof_region_tokens(table: pd.DataFrame, oof: list[OofSplit], t: float, recipe: Recipe) -> pd.DataFrame:
    """OOF regions kept at t, with their CDS tokens (recomposed deterministically) and main entity."""
    rows = []
    comp = {}
    for o in oof:
        if o.split not in comp:
            sd = split_data(table, o.split, recipe)
            comp[o.split] = {s.kingdom: s for s in compose(sd.val, duplicate_positives=1,
                                                             seed=recipe.compose.seed + 1)}
        seq = comp[o.split][o.kingdom]
        assert np.array_equal(seq.orf_entity, o.entity), "recomposition differs from the CV run"
        r = o.regions[np.isclose(o.regions["threshold"], t)]
        spans = [np.arange(a, b) for a, b in zip(r["start"], r["end"], strict=True)]
        mains = main_entity(spans, o.entity, o.label, list(table["entity_id"]))
        for (a, b), mn in zip(zip(r["start"], r["end"], strict=True), mains, strict=True):
            rows.append({"split": o.split, "kingdom": o.kingdom, "start": a, "end": b, "main": mn,
                         "cds_tokens": seq.cds_tokens[a:b]})
    return pd.DataFrame(rows, columns=["split", "kingdom", "start", "end", "main", "cds_tokens"])


def run_tune(run: Path, table: pd.DataFrame, *, recipe: Recipe | None = None, labels: pd.DataFrame | None = None,
             parents: dict | None = None, n_boot: int = 1000, entry_types: dict[str, str] | None = None) -> dict:
    """``entry_types``: InterPro entry → type, for ``npc.key_domains.entry_types``."""
    recipe = recipe or load_recipe()
    run = Path(run)
    out = run / "tune"
    out.mkdir(parents=True, exist_ok=True)
    n = recipe.splits.n_splits
    cvm = pd.DataFrame([json.loads((run / "cv" / f"split{k}_seed0" / "metrics.json").read_text())
                        for k in range(n)])
    epochs = int(np.median(cvm["best_epoch"]))
    oof = load_oof(run / "cv", table, n)
    sw = sweep(oof, table, recipe)
    sw.to_csv(out / "sweep.tsv", sep="\t", index=False)
    op = recipe.operating_point
    floor, metric, ratio = op.floor, op.floor_metric, op.min_pred_ratio
    t_star, floor_met = M.operating_point(sw[sw["stratum"] == "all"], floor, metric, ratio)
    if not floor_met:
        log.warning("no threshold reaches %s >= %.2f; t* = %.2f (highest %s)", metric, floor, t_star, metric)
    s = scored_at(oof, t_star, table)
    val = {"all": M.aggregate(s)}
    ci = {"all": M.bootstrap(s, n_boot=n_boot)}
    for k in sorted(s.gt["kingdom"].unique()):
        val[k] = M.aggregate(subset(s, k))
        ci[k] = M.bootstrap(subset(s, k), n_boot=n_boot)
    res = {"epochs": epochs, "best_epochs": cvm["best_epoch"].tolist(),
           "val_orf_aucpr": cvm["val_orf_aucpr"].tolist(), "t_star": t_star,
           "threshold_low": recipe.decode.model_copy(update={"threshold": t_star}).threshold_low,
           "floor_metric": metric, "floor": floor, "min_pred_ratio": ratio, "floor_met": floor_met,
           "validation": val, "validation_ci": ci}

    regions = oof_region_tokens(table, oof, t_star, recipe)
    regions.to_parquet(out / "oof_regions.parquet", index=False)
    if labels is not None and parents is not None:
        n_lab = len(labelled(labels, table.loc[table["kind"] == "positive", "entity_id"].tolist()))
        if n_lab >= recipe.npc.min_labelled:
            res["npc"] = tune_npc(table, regions, labels, parents, recipe, out, entry_types)
        else:
            log.warning("NPC head off: %d NPC-labelled BGCs (< %d)", n_lab, recipe.npc.min_labelled)
    (out / "tune.json").write_text(json.dumps(res, indent=2, default=float))
    return res


def tune_npc(table: pd.DataFrame, regions: pd.DataFrame, labels: pd.DataFrame, parents: dict, recipe: Recipe,
             out: Path, entry_types: dict[str, str] | None = None) -> dict:
    kd_cfg = recipe.npc.key_domains
    allowed = kd.allowed_tokens(kd_cfg, entry_types)
    loci = table[table["kind"] == "positive"].set_index("entity_id")
    preds, kd_parts = [], []
    for k in range(recipe.splits.n_splits):
        role = mc_split(table, k, n_splits=recipe.splits.n_splits, val_fraction=recipe.splits.val_fraction)
        pos = table["kind"] == "positive"
        tr = table.loc[pos & (role == "train"), "entity_id"].tolist()
        va = table.loc[pos & (role == "val"), "entity_id"].tolist()
        p = cv_split(loci, regions, labels, parents, tr, va, k, recipe.npc)
        if p is None:
            continue
        preds.append(p)
        # key-domain rules: same validation regions and order as cv_split
        ls = LabelSpace(superclasses=[], parents=parents)
        fit = locus_examples(loci, labels, labelled(labels, tr))
        bg = table.loc[(table["kind"] == "background") & (role == "train"), "cds_tokens"].tolist()
        rules = kd.fit_rules(fit.cds, ls.encode(labels, fit.ids)[0], PATHWAYS, kd_cfg, allowed=allowed, background=bg)
        val = region_examples(regions[regions["split"] == k], labels, va)
        has, with_rules = kd.has_rule(rules, val.cds, PATHWAYS)
        kd_parts.append({"preds": p, "has": has, "with_rules": with_rules})
    targets = recipe.npc.precision_targets
    th = fit_thresholds(preds, targets)
    res = {"pathway_thresholds": th.pathway.tolist(),
           "abstention": {lv: {str(x): v for x, v in d.items()} for lv, d in th.abstention.items()},
           "n_cv_regions": int(sum(len(p.pp) for p in preds))}
    if kd_parts:
        calls = kd.oof_calls(kd_parts, lambda k: cross_split_thresholds(preds, k, targets),
                             recipe.npc.default_precision, PATHWAYS)
        ev = kd.evaluate(calls, kd.coverage(kd_parts, PATHWAYS), kd_cfg.min_coverage)
        res["key_domains"] = {"setting": kd.setting(kd_cfg), **ev}
        pd.DataFrame(ev["per_category"]).to_csv(out / "npc_key_domains_oof.tsv", sep="\t", index=False,
                                                float_format="%.4g")
    np.savez_compressed(out / "npc_cv_preds.npz", **{f"s{i}_{f}": getattr(p, f) for i, p in enumerate(preds)
                                                      for f in ("pp", "ps", "yp", "ys", "m")})
    return res
