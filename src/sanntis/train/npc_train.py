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

"""NPClassifier stage training: label space, CV predictions, threshold / abstention fit, final fit.

- Label space: superclasses with >= ``min_support`` training BGCs; rarer ones fold into ``other <pathway>`` (one
  per parent pathway the BGC carries). A missing superclass is masked, never a negative.
- Examples: labelled loci, plus OOF detector regions (kept by the RF post-filter at t*) that take the labels of
  the positive with which they share the most ORFs ("main entity").
- CV (per ``mc`` split k): label space + members fitted on the training loci and the OOF regions of the other
  splits whose main entity is a training BGC; predictions on split k's OOF regions of validation BGCs.
- Thresholds: per-label pathway threshold (max F1, grid 0.05..0.95 step 0.025) and top-1 abstention thresholds
  (lowest confidence whose cumulative precision >= target) fitted on the pooled CV region predictions.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from sanntis.config import NpcConfig
from sanntis.model.npc import NpcModel, NpcThresholds, remap

log = logging.getLogger(__name__)

PATHWAYS = ["Alkaloids", "Amino acids and Peptides", "Carbohydrates", "Fatty acids", "Polyketides",
            "Shikimates and Phenylpropanoids", "Terpenoids"]


@dataclass
class LabelSpace:
    superclasses: list[str]
    parents: dict[str, list[str]]
    folded: dict[str, list[str]] = field(default_factory=dict)
    pathways: list[str] = field(default_factory=lambda: list(PATHWAYS))

    @classmethod
    def fit(cls, labels: pd.DataFrame, train_ids: Sequence[str], parents: dict[str, list[str]],
            min_support: int = 10) -> LabelSpace:
        cnt = pd.Series([s for x in train_ids for s in set(labels.at[x, "superclass"])]).value_counts()
        keep = sorted(cnt[cnt >= min_support].index)
        rare = [s for s in cnt.index if s not in set(keep)]
        others = sorted({f"other {p}" for s in rare for p in parents.get(s, [])})
        return cls(superclasses=keep + others, parents=parents,
                   folded={s: [f"other {p}" for p in parents.get(s, [])] for s in rare})

    def fold(self, sc: Sequence[str], pw: Sequence[str]) -> list[str]:
        out = []
        for s in sc:
            if s in self.superclasses:
                out.append(s)
                continue
            tgt = self.folded.get(s) or [f"other {p}" for p in self.parents.get(s, [])]
            inpw = [t for t in tgt if t[6:] in pw] or tgt
            out.extend(t for t in inpw if t in self.superclasses)
        return sorted(set(out))

    def encode(self, labels: pd.DataFrame, ids: Sequence[str]):
        pi = {p: i for i, p in enumerate(self.pathways)}
        si = {s: i for i, s in enumerate(self.superclasses)}
        yp = np.zeros((len(ids), len(pi)), np.float32)
        ys = np.zeros((len(ids), len(si)), np.float32)
        m = np.zeros(len(ids), np.float32)
        for r, x in enumerate(ids):
            pw = list(labels.at[x, "pathway"])
            for p in pw:
                yp[r, pi[p]] = 1
            sc = self.fold(labels.at[x, "superclass"], pw)
            for s in sc:
                ys[r, si[s]] = 1
            m[r] = float(len(sc) > 0)
        return yp, ys, m


@dataclass
class Examples:
    ids: list[str]  # label source (entity) per example
    cds: list[list[list[str]]]
    kingdom: np.ndarray
    kind: list[str]  # locus | region

    def __len__(self) -> int:
        return len(self.ids)

    @staticmethod
    def concat(parts: Sequence[Examples]) -> Examples:
        parts = [p for p in parts if len(p)]
        if not parts:
            return Examples([], [], np.zeros(0, dtype=object), [])
        return Examples(sum((p.ids for p in parts), []), sum((p.cds for p in parts), []),
                        np.concatenate([p.kingdom for p in parts]), sum((p.kind for p in parts), []))


def labelled(labels: pd.DataFrame, ids: Sequence[str]) -> list[str]:
    ok = set(labels.index[labels["npc_status"] == "classified"])
    return [x for x in ids if x in ok]


def locus_examples(loci: pd.DataFrame, labels: pd.DataFrame, ids: Sequence[str]) -> Examples:
    """``loci`` indexed by entity_id with cds_tokens, kingdom."""
    ids = labelled(labels, [x for x in ids if x in loci.index])
    return Examples(list(ids), [list(loci.at[x, "cds_tokens"]) for x in ids],
                    loci.loc[list(ids), "kingdom"].to_numpy(), ["locus"] * len(ids))


def region_examples(regions: pd.DataFrame, labels: pd.DataFrame, entity_ids: Sequence[str] | None = None
                    ) -> Examples:
    """``regions``: main (entity id or None), cds_tokens, kingdom. Keeps regions whose main entity is labelled."""
    r = regions[regions["main"].notna()]
    ok = set(labelled(labels, r["main"].unique()))
    if entity_ids is not None:
        ok &= set(entity_ids)
    r = r[r["main"].isin(ok)]
    return Examples(r["main"].tolist(), [list(c) for c in r["cds_tokens"]], r["kingdom"].to_numpy(),
                    ["region"] * len(r))


# ── thresholds ───────────────────────────────────────────────────────────────


def best_thresholds(Y: np.ndarray, P: np.ndarray) -> np.ndarray:
    grid = np.arange(0.05, 0.951, 0.025)
    t = np.full(Y.shape[1], 0.5)
    for j in range(Y.shape[1]):
        y = Y[:, j] > 0.5
        if not y.any():
            continue
        f = [2 * (y & (P[:, j] >= g)).sum() / max(y.sum() + (P[:, j] >= g).sum(), 1) for g in grid]
        t[j] = grid[int(np.argmax(f))]
    return t


def conf_threshold(conf: np.ndarray, correct: np.ndarray, target: float) -> float:
    o = np.argsort(-conf, kind="stable")
    prec = np.cumsum(correct[o]) / np.arange(1, len(o) + 1)
    ok = np.nonzero(prec >= target)[0]
    return float(conf[o][ok.max()]) if len(ok) else float("inf")


def top1(Y: np.ndarray, P: np.ndarray):
    return (Y[np.arange(len(Y)), P.argmax(1)] > 0.5).astype(float), P.max(1)


@dataclass
class CvPreds:
    """Raw (un-remapped) ensemble predictions on one split's validation regions."""

    pp: np.ndarray
    ps: np.ndarray
    yp: np.ndarray
    ys: np.ndarray
    m: np.ndarray
    kingdom: np.ndarray
    superclasses: list[str]


def cross_split_thresholds(parts: Sequence[CvPreds], k: int, targets: Sequence[float]) -> NpcThresholds:
    """Thresholds for split ``k`` fitted on the other splits (all of them when there is only one)."""
    return fit_thresholds([p for j, p in enumerate(parts) if j != k] or parts, targets)


def fit_thresholds(parts: Sequence[CvPreds], targets: Sequence[float] = (0.8, 0.9)) -> NpcThresholds:
    Y = np.concatenate([d.yp for d in parts])
    P = np.concatenate([d.pp for d in parts])
    t = best_thresholds(Y, P)
    c, q = top1(Y, remap(P, t))
    sc = [top1(d.ys[d.m > 0.5], d.ps[d.m > 0.5]) for d in parts]  # superclass spaces differ per split
    sc_c = np.concatenate([x for x, _ in sc]) if sc else np.zeros(0)
    sc_q = np.concatenate([x for _, x in sc]) if sc else np.zeros(0)
    return NpcThresholds(t, {"pw": {x: conf_threshold(q, c, x) for x in targets},
                             "sc": {x: conf_threshold(sc_q, sc_c, x) for x in targets}})


def fit_members(train: Examples, labels: pd.DataFrame, ls: LabelSpace, thresholds: NpcThresholds, npc_cfg: NpcConfig,
                seed: int = 0, n_jobs: int = -1) -> NpcModel:
    yp, ys, m = ls.encode(labels, train.ids)
    return NpcModel.fit(train.cds, train.kingdom, yp, ys, m, ls.pathways, ls.superclasses, thresholds,
                        rf_trees=npc_cfg.rf.n_estimators, lr_C=npc_cfg.lr.C, knn_k=npc_cfg.knn.k,
                        min_df=npc_cfg.min_df, knn_min_df=npc_cfg.knn.min_df, seed=seed, n_jobs=n_jobs)


def cv_split(loci: pd.DataFrame, regions: pd.DataFrame, labels: pd.DataFrame, parents: dict, train_ids, val_ids,
             split: int, npc_cfg: NpcConfig, seed: int = 0) -> CvPreds | None:
    """``regions`` carries a ``split`` column (the CV split in which the region was out-of-fold)."""
    tr = labelled(labels, train_ids)
    ls = LabelSpace.fit(labels, tr, parents, npc_cfg.min_support)
    others = regions[regions["split"] != split]
    train = Examples.concat([locus_examples(loci, labels, tr), region_examples(others, labels, tr)])
    val = region_examples(regions[regions["split"] == split], labels, val_ids)
    if not len(val) or not len(train):
        return None
    dummy = NpcThresholds(np.full(len(ls.pathways), 0.5), {"pw": {}, "sc": {}})
    model = fit_members(train, labels, ls, dummy, npc_cfg, seed)
    pp, ps = model.raw_proba(val.cds, val.kingdom)
    yp, ys, m = ls.encode(labels, val.ids)
    return CvPreds(pp, ps, yp, ys, m, val.kingdom, ls.superclasses)


def main_entity(region_orfs: Sequence[np.ndarray], orf_entity: np.ndarray, orf_label: np.ndarray,
                entity_ids: Sequence[str]) -> list[str | None]:
    """Per region (array of ORF indices): the positive entity with the most ORFs in it (None if no positive ORF)."""
    out = []
    for idx in region_orfs:
        pos = idx[orf_label[idx] == 1]
        if not len(pos):
            out.append(None)
            continue
        ents, cnt = np.unique(orf_entity[pos], return_counts=True)
        out.append(entity_ids[int(ents[int(np.argmax(cnt))])])
    return out
