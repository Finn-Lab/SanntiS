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
import pandas as pd
import pytest

from sanntis.train.metrics import aggregate, bootstrap, concat, operating_point, score_contig


def _contig():
    # ORFs: 0-9; BGC A = ORFs 2-5 (NRPS;PKS hybrid), BGC B = ORFs 8-9 (terpene)
    orf_bgc = np.array([None, None, "A", "A", "A", "A", None, None, "B", "B"], dtype=object)
    cls = {"A": ("NRPS", "PKS"), "B": ("terpene",)}
    return orf_bgc, cls


def test_definitions():
    orf_bgc, cls = _contig()
    # p1 covers A ORFs 3-5 + background ORF 6; p2 is pure background
    s = score_contig([(3, 7), (7, 8)], orf_bgc, bgc_classes=cls)
    assert len(s.pred) == 2
    m = aggregate(s)
    assert m["recall_t"] == pytest.approx((3 / 4 + 0) / 2)
    assert m["precision_t"] == pytest.approx((3 / 4 + 0) / 2)
    assert m["bgc_recall"] == 0.5 and m["bgc_precision"] == 0.5
    # hybrid A counts in NRPS and PKS: macro over NRPS, PKS, terpene
    assert m["recall_macro_t"] == pytest.approx((0.75 + 0.75 + 0) / 3)
    assert m["bgc_macro_recall"] == pytest.approx(2 / 3)
    p, r = 0.5, 2 / 3
    assert m["f2_macro"] == pytest.approx(5 * p * r / (4 * p + r))


def test_bootstrap_and_weights():
    orf_bgc, cls = _contig()
    parts = [score_contig([(2, 6)], orf_bgc, unit=u, bgc_classes=cls) for u in range(5)]
    s = concat(parts)
    ci = bootstrap(s, n_boot=50)
    lo, hi = ci["recall_t"]
    assert lo <= aggregate(s)["recall_t"] <= hi


def test_operating_point_rule():
    sw = pd.DataFrame({"threshold": [0.3, 0.4, 0.5, 0.6], "precision_t": [0.7, 0.81, 0.85, 0.9],
                       "f2_macro": [0.9, 0.8, 0.8, 0.7]})
    assert operating_point(sw, 0.80, "precision_t") == (0.5, True)  # tie 0.4/0.5 → higher; 0.3 below the floor
    sw.loc[1, "f2_macro"] = 0.85
    assert operating_point(sw, 0.80, "precision_t") == (0.4, True)
    assert operating_point(sw, 0.85, "precision_t") == (0.5, True)
    assert operating_point(sw, 0.95, "precision_t") == (0.6, False)


def test_operating_point_bgc_precision_floor_skips_degenerate():
    # 0.1: whole contigs merged into 10 regions, every BGC-level metric 1.0 → excluded, not chosen
    sw = pd.DataFrame({"threshold": [0.1, 0.2, 0.3, 0.4], "bgc_precision": [1.0, 0.75, 0.85, 0.9],
                       "f2_macro": [1.0, 0.95, 0.91, 0.90], "n_pred": [10, 2600, 2400, 2200],
                       "n_gt": [2297] * 4})
    assert operating_point(sw) == (0.3, True)
    assert operating_point(sw, min_pred_ratio=0) == (0.1, True)


def test_bootstrap_matches_weighted_aggregate():
    orf_bgc, cls = _contig()
    s = concat([score_contig([(2, 4), (7, 9)], orf_bgc, unit=u, bgc_classes=cls) for u in range(4)]
               + [score_contig([(3, 6)], orf_bgc, unit=9, bgc_classes=cls)])
    units = list(pd.unique(pd.concat([s.gt["unit"], s.pred["unit"]])))
    rng = np.random.default_rng(0)
    counts = np.bincount(rng.integers(0, len(units), len(units)), minlength=len(units))
    ref = aggregate(s, unit_weight=dict(zip(units, counts, strict=True)))
    lo, hi = bootstrap(s, n_boot=1, seed=0)["f2_macro"]
    assert lo == pytest.approx(ref["f2_macro"]) and hi == pytest.approx(ref["f2_macro"])
