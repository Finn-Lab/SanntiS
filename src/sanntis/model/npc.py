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

"""NPClassifier stage: RF + LR (TF-IDF) + kNN-Jaccard ensemble on region token bags.

Members read the order-free bag of the region's raw domain tokens (no vocabulary filter beyond their own
``min_df``). The ensemble is the unweighted mean of the members' probabilities per level. Pathway
probabilities are remapped per label so that each tuned threshold maps to 0.5; abstention compares the top-1
(remapped) confidence with the tier threshold.

Bundle files: ``npc.skops`` (the sklearn estimators + bag vocabularies + label names), ``npc_knn.npz``
(binary reference matrix + label vectors), ``npc_thresholds.json``, optionally ``npc_key_domains.json`` (pathway
key-domain support rules, :mod:`sanntis.model.key_domains`).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from sanntis.features.tokens import token_counts
from sanntis.features.vocab import frequent_tokens
from sanntis.model import _skops
from sanntis.model.key_domains import KeyDomains

Cds = Sequence[Sequence[str]]


def _with_kingdom(X: sp.csr_matrix, kingdom: np.ndarray) -> sp.csr_matrix:
    return sp.hstack([X, sp.csr_matrix((kingdom == "euk").astype(np.float32)[:, None])]).tocsr()


def _proba_cols(est, X, n: int) -> np.ndarray:
    pr = est.predict_proba(X)
    if isinstance(pr, list):  # native multi-output (RandomForest)
        cols = []
        for j, q in enumerate(pr):
            cl = list(est.classes_[j])
            cols.append(q[:, cl.index(1)] if 1 in cl else np.zeros(X.shape[0]))
        return np.stack(cols, 1)
    return pr.reshape(X.shape[0], n)


def remap(P: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Piecewise-linear per-label map sending threshold ``t`` to 0.5."""
    lo = 0.5 * P / t
    hi = 0.5 + 0.5 * (P - t) / (1 - t)
    return np.where(t > P, lo, hi)


def pathway_calls(P: np.ndarray) -> np.ndarray:
    """Called pathways per region (remapped probabilities): every one >= 0.5, else the top one."""
    called = P >= 0.5
    called[np.arange(len(P)), P.argmax(1)] |= ~called.any(1)
    return called


@dataclass
class NpcThresholds:
    pathway: np.ndarray  # per pathway, raw probability scale
    abstention: dict[str, dict[float, float]]  # {"pw"|"sc": {precision target: top-1 confidence}}

    def to_json(self, pathways: list[str]) -> dict:
        return {"pathways": pathways, "pathway_thresholds": [float(x) for x in self.pathway],
                "abstention": {lv: {str(k): float(v) for k, v in d.items()} for lv, d in self.abstention.items()}}

    @classmethod
    def from_json(cls, d: dict) -> NpcThresholds:
        return cls(np.asarray(d["pathway_thresholds"], dtype=float),
                   {lv: {float(k): float(v) for k, v in x.items()} for lv, x in d["abstention"].items()})


@dataclass
class NpcModel:
    pathways: list[str]
    superclasses: list[str]
    rf_vocab: list[str]
    rf_pathway: object
    rf_superclass: object
    lr_vocab: list[str]
    lr_tfidf: object  # sklearn TfidfTransformer
    lr_pathway: object
    lr_superclass: object
    knn_vocab: list[str]
    knn_X: sp.csr_matrix  # binary bags of the reference examples
    knn_yp: np.ndarray
    knn_ys: np.ndarray
    knn_mask: np.ndarray
    thresholds: NpcThresholds
    knn_k: int = 5
    key_domains: KeyDomains | None = None
    _idx: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self._idx = {name: {t: i for i, t in enumerate(v)} for name, v in
                     (("rf", self.rf_vocab), ("lr", self.lr_vocab), ("knn", self.knn_vocab))}
        self._knn_n = np.asarray(self.knn_X.sum(1)).ravel()
        for est in (self.rf_pathway, self.rf_superclass, self.lr_pathway, self.lr_superclass):
            if hasattr(est, "n_jobs"):
                est.n_jobs = 1

    # ── members ──────────────────────────────────────────────────────────────
    def _rf(self, regions, kingdom):
        X = token_counts(regions, self._idx["rf"]).astype(np.float32).log1p()
        X = _with_kingdom(X, kingdom)
        return (_proba_cols(self.rf_pathway, X, len(self.pathways)),
                _proba_cols(self.rf_superclass, X, len(self.superclasses)))

    def _lr(self, regions, kingdom):
        X = self.lr_tfidf.transform(token_counts(regions, self._idx["lr"]).astype(np.float32))
        X = _with_kingdom(X, kingdom)
        return (_proba_cols(self.lr_pathway, X, len(self.pathways)),
                _proba_cols(self.lr_superclass, X, len(self.superclasses)))

    def _knn(self, regions):
        Q = token_counts(regions, self._idx["knn"], binary=True).astype(np.float32)
        inter = (Q @ self.knn_X.T).toarray()
        nq = np.asarray(Q.sum(1)).ravel()
        sim = inter / np.maximum(nq[:, None] + self._knn_n[None, :] - inter, 1)

        def vote(Y, mask):
            s = np.where(mask[None, :] > 0.5, sim, -1)
            idx = np.argsort(-s, 1)[:, : self.knn_k]
            w = np.take_along_axis(s, idx, 1).clip(min=0) + 1e-9
            return (w[:, :, None] * Y[idx]).sum(1) / w.sum(1, keepdims=True)

        return vote(self.knn_yp, np.ones(len(self.knn_yp))), vote(self.knn_ys, self.knn_mask)

    # ── ensemble ─────────────────────────────────────────────────────────────
    def raw_proba(self, regions: Sequence[Cds], kingdom: np.ndarray | str) -> tuple[np.ndarray, np.ndarray]:
        """→ (pathway, superclass) ensemble means, before the pathway remap."""
        kingdom = np.full(len(regions), kingdom) if isinstance(kingdom, str) else np.asarray(kingdom)
        out = [self._rf(regions, kingdom), self._knn(regions), self._lr(regions, kingdom)]
        return sum(o[0] for o in out) / 3.0, sum(o[1] for o in out) / 3.0

    def proba(self, regions: Sequence[Cds], kingdom: np.ndarray | str) -> tuple[np.ndarray, np.ndarray]:
        """→ (remapped pathway probabilities, superclass probabilities)."""
        pp, ps = self.raw_proba(regions, kingdom)
        return remap(pp, self.thresholds.pathway), ps

    def calls(self, pp: np.ndarray, ps: np.ndarray, precision: float = 0.9) -> pd.DataFrame:
        tp = self.thresholds.abstention["pw"][precision]
        ts = self.thresholds.abstention["sc"][precision]
        rows = []
        for a, b, c in zip(pp, ps, pathway_calls(pp), strict=True):
            sel = np.nonzero(c)[0].tolist()
            j = int(b.argmax())
            rows.append({"npc_pathway": "|".join(self.pathways[i] for i in sel),
                         "npc_pathway_score": float(a.max()),
                         "npc_pathway_confident": bool(a.max() >= tp),
                         "npc_superclass": self.superclasses[j] if b[j] >= ts else "unassigned",
                         "npc_superclass_top": self.superclasses[j], "npc_superclass_score": float(b[j])})
        return pd.DataFrame(rows, columns=["npc_pathway", "npc_pathway_score", "npc_pathway_confident",
                                           "npc_superclass", "npc_superclass_top", "npc_superclass_score"])

    def predict(self, regions: Sequence[Cds], kingdom: str = "proka", precision: float = 0.9) -> pd.DataFrame:
        if not len(regions):
            out = self.calls(np.zeros((0, len(self.pathways))), np.zeros((0, len(self.superclasses))), precision)
        else:
            out = self.calls(*self.proba(regions, kingdom), precision)
        return self.with_key_domains(out, regions)

    def with_key_domains(self, calls: pd.DataFrame, regions: Sequence[Cds]) -> pd.DataFrame:
        """Adds ``npc_pathway_key_domains_support`` / ``_unsupported`` (None without key-domain rules)."""
        sup = [self.key_domains.support(c, p.split("|")) for c, p in zip(regions, calls["npc_pathway"], strict=True)] \
            if self.key_domains is not None else [(None, None)] * len(calls)
        return calls.assign(npc_pathway_key_domains_support=pd.Series([s for s, _ in sup], index=calls.index,
                                                                      dtype=object),
                            npc_pathway_key_domains_unsupported=pd.Series([u for _, u in sup], index=calls.index,
                                                                          dtype=object))

    # ── fitting (training) ───────────────────────────────────────────────────
    @classmethod
    def fit(cls, regions: Sequence[Cds], kingdom: np.ndarray, yp: np.ndarray, ys: np.ndarray, sc_mask: np.ndarray,
            pathways: list[str], superclasses: list[str], thresholds: NpcThresholds, *, rf_trees: int = 500,
            lr_C: float = 10.0, knn_k: int = 5, min_df: int = 2, knn_min_df: int = 1, seed: int = 0,
            n_jobs: int = -1) -> NpcModel:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_extraction.text import TfidfTransformer
        from sklearn.linear_model import LogisticRegression
        from sklearn.multiclass import OneVsRestClassifier

        kingdom = np.asarray(kingdom)
        m = sc_mask > 0.5
        vocab = frequent_tokens(regions, min_df)
        idx = {t: i for i, t in enumerate(vocab)}
        C = token_counts(regions, idx)

        def rf():
            return RandomForestClassifier(n_estimators=rf_trees, max_features="sqrt", random_state=seed, n_jobs=n_jobs)

        def lr():
            return OneVsRestClassifier(LogisticRegression(C=lr_C, max_iter=5000, solver="liblinear"),
                                       n_jobs=n_jobs)

        Xrf = _with_kingdom(C.astype(np.float32).log1p(), kingdom)
        rf_p, rf_s = rf().fit(Xrf, yp.astype(int)), rf().fit(Xrf[m], ys[m].astype(int))
        tf = TfidfTransformer(sublinear_tf=True).fit(C)
        Xlr = _with_kingdom(tf.transform(C.astype(np.float32)), kingdom)
        lr_p, lr_s = lr().fit(Xlr, yp.astype(int)), lr().fit(Xlr[m], ys[m].astype(int))
        kv = frequent_tokens(regions, knn_min_df)
        Xk = token_counts(regions, {t: i for i, t in enumerate(kv)}, binary=True).astype(np.float32)
        return cls(pathways=pathways, superclasses=superclasses, rf_vocab=vocab, rf_pathway=rf_p, rf_superclass=rf_s,
                   lr_vocab=vocab, lr_tfidf=tf, lr_pathway=lr_p, lr_superclass=lr_s, knn_vocab=kv, knn_X=Xk,
                   knn_yp=yp.astype(float), knn_ys=ys.astype(float), knn_mask=sc_mask.astype(float),
                   thresholds=thresholds, knn_k=knn_k)

    # ── serialisation ────────────────────────────────────────────────────────
    def save(self, bundle: Path) -> list[str]:
        bundle = Path(bundle)
        _skops.dump({"pathways": self.pathways, "superclasses": self.superclasses,
                     "rf": {"vocab": self.rf_vocab, "pathway": self.rf_pathway, "superclass": self.rf_superclass},
                     "lr": {"vocab": self.lr_vocab, "tfidf": self.lr_tfidf, "pathway": self.lr_pathway,
                            "superclass": self.lr_superclass}}, bundle / "npc.skops")
        X = self.knn_X.tocsr()
        np.savez_compressed(bundle / "npc_knn.npz", data=X.data, indices=X.indices, indptr=X.indptr,
                            shape=np.asarray(X.shape), vocab=np.asarray(self.knn_vocab, dtype=str), yp=self.knn_yp,
                            ys=self.knn_ys, mask=self.knn_mask, k=np.asarray(self.knn_k))
        th = self.thresholds.to_json(self.pathways)
        th["superclasses"] = self.superclasses
        (bundle / "npc_thresholds.json").write_text(json.dumps(th, indent=2))
        files = ["npc.skops", "npc_knn.npz", "npc_thresholds.json"]
        return files + ([self.key_domains.save(bundle)] if self.key_domains is not None else [])

    @classmethod
    def load(cls, bundle: Path) -> NpcModel:
        bundle = Path(bundle)
        d = _skops.load(bundle / "npc.skops")
        z = np.load(bundle / "npc_knn.npz", allow_pickle=False)
        X = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
        th = NpcThresholds.from_json(json.loads((bundle / "npc_thresholds.json").read_text()))
        return cls(pathways=list(d["pathways"]), superclasses=list(d["superclasses"]), rf_vocab=list(d["rf"]["vocab"]),
                   rf_pathway=d["rf"]["pathway"], rf_superclass=d["rf"]["superclass"],
                   lr_vocab=list(d["lr"]["vocab"]), lr_tfidf=d["lr"]["tfidf"], lr_pathway=d["lr"]["pathway"],
                   lr_superclass=d["lr"]["superclass"], knn_vocab=[str(x) for x in z["vocab"]], knn_X=X,
                   knn_yp=z["yp"], knn_ys=z["ys"], knn_mask=z["mask"], thresholds=th, knn_k=int(z["k"]),
                   key_domains=KeyDomains.load(bundle))
