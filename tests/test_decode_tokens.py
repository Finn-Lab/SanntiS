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

import numpy as np

from sanntis.config import DecodeConfig
from sanntis.features.tokens import linearize
from sanntis.features.vocab import SEP, UNK, Vocab
from sanntis.model.decode import decode, orf_scores
from sanntis.model.detector import window_starts


def test_linearize_spec_rules():
    v = Vocab(["A", "B"])
    ts = linearize([["A", "A", "B"], [], ["C", "C", "A"]], v)
    a, b = v.stoi["A"], v.stoi["B"]
    # <sep> before every CDS with tokens; contiguous duplicates removed on raw tokens; OOV → <unk>
    assert ts.tokens.tolist() == [SEP, a, b, SEP, UNK, a]
    assert ts.orf_index.tolist() == [0, 0, 0, 2, 2, 2]
    assert ts.n_orfs == 3


def test_orf_scores_mean_zero_fill():
    s = orf_scores(np.array([0.2, 0.4, 0.9]), np.array([0, 0, 2]), 4, agg="mean", fill="zero")
    np.testing.assert_allclose(s, [0.3, 0.0, 0.9, 0.0])


def test_decode_hysteresis_and_merge():
    cfg = DecodeConfig(threshold=0.5, hysteresis_gap=0.15, merge_gap=3)
    s = np.array([0.1, 0.4, 0.6, 0.4, 0.0, 0.0, 0.0, 0.36, 0.55, 0.1, 0.0, 0.0, 0.0, 0.0, 0.4, 0.45])
    # runs >= 0.35: [1,4) has 0.6 → kept; [7,9) has 0.55 → kept; [14,16) max 0.45 → dropped
    # gap between [1,4) and [7,9) is 3 ORFs → merged
    assert decode(s, cfg) == [(1, 9)]
    assert decode(s, DecodeConfig(threshold=0.5, hysteresis_gap=0.15, merge_gap=2)) == [(1, 4), (7, 9)]


def test_window_starts_cover():
    for n in (1, 50, 200, 201, 333, 1000):
        st = window_starts(n, 200)
        cov = np.zeros(n)
        for s in st:
            cov[s : s + 200] += 1
        assert (cov > 0).all() and st[0] == 0
