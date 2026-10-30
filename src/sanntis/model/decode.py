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

"""Token probabilities → ORF scores → candidate regions."""

from __future__ import annotations

import numpy as np

from sanntis.config import DecodeConfig

Region = tuple[int, int]  # [start_orf, end_orf)


def orf_scores(token_probs: np.ndarray, orf_index: np.ndarray, n_orfs: int, *, agg: str = "mean",
               fill: str = "zero") -> np.ndarray:
    """Aggregate token probabilities per ORF. ORFs without tokens score 0 (``fill="zero"``)."""
    out = np.zeros(n_orfs, dtype=np.float64)
    has = np.zeros(n_orfs, dtype=bool)
    if token_probs.size:
        has[orf_index] = True
        if agg == "max":
            np.maximum.at(out, orf_index, token_probs)
        else:
            sums = np.bincount(orf_index, weights=token_probs, minlength=n_orfs)
            cnt = np.bincount(orf_index, minlength=n_orfs)
            out = np.divide(sums, cnt, out=np.zeros(n_orfs), where=cnt > 0)
    if fill == "neighbors" and has.any() and not has.all():
        idx = np.arange(n_orfs)
        ann = idx[has]
        left = np.searchsorted(ann, idx, side="right") - 1
        right = np.searchsorted(ann, idx, side="left")
        lv = np.where(left >= 0, out[ann[np.clip(left, 0, len(ann) - 1)]], 0.0)
        rv = np.where(right < len(ann), out[ann[np.clip(right, 0, len(ann) - 1)]], 0.0)
        out = np.where(has, out, np.minimum(lv, rv))
    return out


def _runs(mask: np.ndarray) -> list[Region]:
    if not mask.any():
        return []
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist(), strict=True))


def decode(orf_s: np.ndarray, cfg: DecodeConfig, threshold: float | None = None) -> list[Region]:
    """Hysteresis (t, t - gap) then merge regions separated by <= ``merge_gap`` ORFs. ``threshold`` overrides
    ``cfg.threshold`` (threshold sweeps)."""
    if threshold is not None:
        cfg = cfg.model_copy(update={"threshold": threshold})
    t, t_low = cfg.threshold, cfg.threshold_low
    s = orf_s
    regions = [r for r in _runs(s >= t_low) if s[r[0] : r[1]].max() >= t] if t_low < t else _runs(s >= t)
    if cfg.merge_gap > 0 and regions:
        merged = [regions[0]]
        for a, b in regions[1:]:
            if a - merged[-1][1] <= cfg.merge_gap:
                merged[-1] = (merged[-1][0], b)
            else:
                merged.append((a, b))
        regions = merged
    return [r for r in regions if r[1] - r[0] >= cfg.min_orfs]
