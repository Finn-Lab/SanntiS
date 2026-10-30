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

"""Detection metrics in ORF space.

Scoring is two-stage so that strata and bootstrap resamples never re-decode:

1. :func:`score_contig` turns one contig's predictions + ground truth into per-GT rows (BGC, classes, ORF count,
   covered ORFs) and per-prediction rows (true ORFs, denominator) — each tagged with a resampling ``unit``.
2. :func:`aggregate` computes the metric set from those rows, optionally with per-unit weights (bootstrap, same
   core :func:`_metrics`) and a GT filter (strata). Precision always reads every scored prediction of the dataset.

Definitions (no cardinality corrections):

- Recall_T = mean over true BGCs of the fraction of their ORFs covered by any prediction; Recall_macro_T = mean
  over classes of the per-class Recall_T (a hybrid counts in every class it carries).
- Precision_T = mean over scored predictions of (ORFs in a true BGC) / |p|.
- BGC recall = share of true BGCs with >= 1 covered ORF (macro: per class, then mean); BGC precision = share of
  scored predictions sharing >= 1 ORF with a true BGC.
- F1_T, F1_macro_T, F1_macro (BGC precision × BGC macro recall), F2_macro (recall-weighted, selects t*).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sanntis.config import load_recipe

METRICS = ["precision_t", "recall_t", "recall_macro_t", "f1_t", "f1_macro_t", "bgc_precision", "bgc_recall",
           "bgc_macro_recall", "f1_macro", "f2_macro"]
METRIC_LABELS = {"precision_t": "Precision_T", "recall_t": "Recall_T", "recall_macro_t": "Recall_macro_T",
                 "f1_t": "F1_T", "f1_macro_t": "F1_macro_T", "bgc_precision": "BGC precision",
                 "bgc_recall": "BGC recall", "bgc_macro_recall": "BGC macro recall", "f1_macro": "F1_macro",
                 "f2_macro": "F2_macro"}
CLASSES = list(load_recipe().postfilter.classes)

Region = tuple[int, int]


def fbeta(p: float, r: float, beta: float = 1.0) -> float:
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r) if (b2 * p + r) > 0 else 0.0


def gt_regions(orf_bgc: np.ndarray) -> list[tuple[int, int, object]]:
    """Maximal runs of ORFs of one BGC (``orf_bgc`` = BGC id per ORF, None / -1 / "" for background)."""
    out = []
    n = len(orf_bgc)
    i = 0
    while i < n:
        b = orf_bgc[i]
        if b is None or b == -1 or b == "" or (isinstance(b, float) and np.isnan(b)):
            i += 1
            continue
        j = i
        while j < n and orf_bgc[j] == b:
            j += 1
        out.append((i, j, b))
        i = j
    return out


@dataclass
class Scored:
    gt: pd.DataFrame  # unit, bgc, n_orfs, covered, classes (tuple), + any meta columns
    pred: pd.DataFrame  # unit, n_true, denom, hit


def score_contig(pred: Sequence[Region], orf_bgc: np.ndarray, *, unit: object = 0, bgc_classes: dict | None = None,
                 pred_unit: Callable[[int, int], object] | None = None) -> Scored:
    """``pred_unit(a, b)`` assigns a prediction to a resampling unit (default: ``unit``)."""
    n = len(orf_bgc)
    is_bgc = np.array([not (b is None or b == -1 or b == "" or (isinstance(b, float) and np.isnan(b)))
                       for b in orf_bgc], dtype=bool)
    cov = np.zeros(n + 1, dtype=np.int64)
    for a, b in pred:
        cov[a] += 1
        cov[b] -= 1
    covered = np.cumsum(cov)[:-1] > 0
    pos_cum = np.concatenate([[0], np.cumsum(is_bgc)])
    gts = [{"unit": unit, "bgc": g, "n_orfs": b - a, "covered": int(covered[a:b].sum()),
            "classes": tuple((bgc_classes or {}).get(g, ()))} for a, b, g in gt_regions(orf_bgc)]
    preds = []
    for a, b in pred:
        n_true = int(pos_cum[b] - pos_cum[a])
        preds.append({"unit": pred_unit(a, b) if pred_unit else unit, "n_true": n_true,
                      "denom": max(b - a, 1), "hit": n_true > 0})
    return Scored(pd.DataFrame(gts, columns=["unit", "bgc", "n_orfs", "covered", "classes"]),
                  pd.DataFrame(preds, columns=["unit", "n_true", "denom", "hit"]))


def concat(parts: Sequence[Scored]) -> Scored:
    return Scored(pd.concat([p.gt for p in parts], ignore_index=True),
                  pd.concat([p.pred for p in parts], ignore_index=True))


def _wmean(x: np.ndarray, w: np.ndarray) -> float:
    s = w.sum()
    return float((x * w).sum() / s) if s > 0 else float("nan")


def _metrics(r, hit, pt, ph, masks: dict[str, np.ndarray], wg: np.ndarray, wp: np.ndarray) -> dict[str, float]:
    """The metric set from per-GT (``r``: covered share, ``hit``) and per-prediction (``pt``: true share, ``ph``:
    hit) arrays, weighted per row (``wg`` / ``wp``); ``masks``: class → GT rows carrying it."""
    p_t = _wmean(pt, wp) if len(pt) else 0.0
    bgc_p = _wmean(ph, wp) if len(ph) else 0.0
    out = {"precision_t": p_t, "recall_t": _wmean(r, wg), "bgc_precision": bgc_p, "bgc_recall": _wmean(hit, wg),
           "n_gt": float(wg.sum()), "n_pred": float(wp.sum())}
    per_r, per_h = [], []
    for c, m in masks.items():
        if wg[m].sum() == 0:
            continue
        out[f"recall_t[{c}]"] = _wmean(r[m], wg[m])
        out[f"bgc_recall[{c}]"] = _wmean(hit[m], wg[m])
        out[f"n_gt[{c}]"] = float(wg[m].sum())
        per_r.append(out[f"recall_t[{c}]"])
        per_h.append(out[f"bgc_recall[{c}]"])
    out["recall_macro_t"] = float(np.mean(per_r)) if per_r else float("nan")
    out["bgc_macro_recall"] = float(np.mean(per_h)) if per_h else float("nan")
    out["f1_t"] = fbeta(p_t, out["recall_t"])
    out["f1_macro_t"] = fbeta(p_t, out["recall_macro_t"])
    out["f1_macro"] = fbeta(bgc_p, out["bgc_macro_recall"])
    out["f2_macro"] = fbeta(bgc_p, out["bgc_macro_recall"], beta=2.0)
    return out


def _arrays(gt: pd.DataFrame, pr: pd.DataFrame) -> tuple:
    r = (gt["covered"] / gt["n_orfs"]).to_numpy(float)
    hit = (gt["covered"] > 0).to_numpy(float)
    pt = (pr["n_true"] / pr["denom"]).to_numpy(float)
    ph = pr["hit"].to_numpy(float)
    cls_sets = gt["classes"].to_numpy()
    masks = {c: np.array([c in cs for cs in cls_sets], dtype=bool) for c in CLASSES}
    return r, hit, pt, ph, {c: m for c, m in masks.items() if m.any()}


def aggregate(s: Scored, *, gt_mask: np.ndarray | None = None, unit_weight: dict | None = None) -> dict[str, float]:
    gt = s.gt if gt_mask is None else s.gt[gt_mask]
    pr = s.pred
    wg = gt["unit"].map(unit_weight).fillna(0).to_numpy(float) if unit_weight else np.ones(len(gt))
    wp = pr["unit"].map(unit_weight).fillna(0).to_numpy(float) if unit_weight else np.ones(len(pr))
    return _metrics(*_arrays(gt, pr), wg, wp)


def bootstrap(s: Scored, *, n_boot: int = 1000, seed: int = 0) -> dict[str, tuple[float, float]]:
    """95% percentile intervals over resampled units (source contigs on test sets, entities on validation).

    Per resample, unit counts become per-row weights (identical to ``aggregate(unit_weight=...)``)."""
    units = pd.unique(pd.concat([s.gt["unit"], s.pred["unit"]], ignore_index=True))
    code = {u: i for i, u in enumerate(units)}
    gc = s.gt["unit"].map(code).to_numpy(int)
    pc = s.pred["unit"].map(code).to_numpy(int)
    arrays = _arrays(s.gt, s.pred)
    rng = np.random.default_rng(seed)
    vals: dict[str, list[float]] = {m: [] for m in METRICS}
    for _ in range(n_boot):
        counts = np.bincount(rng.integers(0, len(units), len(units)), minlength=len(units)).astype(float)
        a = _metrics(*arrays, counts[gc], counts[pc])
        for m in METRICS:
            vals[m].append(a[m])
    return {m: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) for m, v in vals.items()}


def threshold_grid(spec: Sequence[float] = (0.10, 0.95, 0.05)) -> list[float]:
    lo, hi, step = spec
    return [round(x, 2) for x in np.arange(lo, hi + step / 2, step)]


def degenerate(sweep: pd.DataFrame, min_pred_ratio: float = 0.5) -> pd.Series:
    """Thresholds whose regions number fewer than ``min_pred_ratio`` x true BGCs (contigs merged into a few regions)."""
    if not {"n_pred", "n_gt"} <= set(sweep.columns):
        return pd.Series(False, index=sweep.index)
    return sweep["n_pred"] < min_pred_ratio * sweep["n_gt"]


def operating_point(sweep: pd.DataFrame, floor: float = 0.80, metric: str = "bgc_precision",
                    min_pred_ratio: float = 0.5) -> tuple[float, bool]:
    """t* = highest F2_macro among non-degenerate thresholds with ``metric`` >= floor; ties → the higher threshold.
    Returns (t*, floor_met). If no threshold meets the floor, the one with the highest ``metric`` (ties: higher t)."""
    cand = sweep[~degenerate(sweep, min_pred_ratio)]
    if cand.empty:  # every threshold degenerate (tiny inputs): fall back to the full grid
        cand = sweep
    ok = cand[cand[metric] >= floor]
    if len(ok):
        best = ok.sort_values(["f2_macro", "threshold"], ascending=[False, False]).iloc[0]
        return float(best["threshold"]), True
    best = cand.sort_values([metric, "threshold"], ascending=[False, False]).iloc[0]
    return float(best["threshold"]), False

