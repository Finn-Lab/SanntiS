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

"""RF post-filter + MIBiG class: one-vs-rest RandomForest on the raw region token-count bag."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from scipy import sparse

from sanntis.features.tokens import token_counts
from sanntis.features.vocab import Vocab
from sanntis.model import _skops

NON_BGC = "non-BGC"


class PostFilter:
    def __init__(self, model, vocab: Vocab):
        self.model = model
        self.vocab = vocab
        self.classes_: list[str] = [str(c) for c in model.classes_]
        for est in getattr(model, "estimators_", []):  # per-call thread pools cost more than a few regions
            if hasattr(est, "n_jobs"):
                est.n_jobs = 1
        if hasattr(model, "n_jobs"):
            model.n_jobs = 1

    @classmethod
    def fit(cls, region_tokens: list[Sequence[Sequence[str]]], classes: Sequence[str], vocab: Vocab, *,
            n_estimators: int = 100, random_state: int = 0, n_jobs: int = -1) -> PostFilter:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.multiclass import OneVsRestClassifier

        X = token_counts(region_tokens, vocab.stoi).astype(np.float32)
        model = OneVsRestClassifier(RandomForestClassifier(n_estimators=n_estimators, random_state=random_state,
                                                           n_jobs=n_jobs))
        model.fit(X, np.asarray(classes))
        return cls(model, vocab)

    def predict_proba(self, X: sparse.csr_matrix) -> np.ndarray:
        p = self.model.predict_proba(X)
        return p / np.clip(p.sum(1, keepdims=True), 1e-9, None)

    def classify(self, region_tokens: list[Sequence[Sequence[str]]], drop_threshold: float):
        """→ (keep mask, BGC class = argmax over the BGC classes, P(non-BGC)) per region."""
        if not region_tokens:
            return np.zeros(0, bool), [], np.zeros(0)
        p = self.predict_proba(token_counts(region_tokens, self.vocab.stoi).astype(np.float32))
        nb = self.classes_.index(NON_BGC)
        p_non = p[:, nb]
        p_bgc = p.copy()
        p_bgc[:, nb] = -1
        cls = [self.classes_[i] for i in p_bgc.argmax(1)]
        return p_non < drop_threshold, cls, p_non

    def save(self, path: Path) -> None:
        _skops.dump({"model": self.model, "classes": self.classes_}, path)

    @classmethod
    def load(cls, path: Path, vocab: Vocab) -> PostFilter:
        d = _skops.load(path)
        return cls(d["model"], vocab)
