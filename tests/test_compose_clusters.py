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

"""Multi-BGC training compositions (compose.clusters)."""

import numpy as np
import pandas as pd

from sanntis.config import ClusterConfig
from sanntis.train.compose import compose


def _entities():
    rows = []
    for i in range(40):
        rows.append({"kind": "positive", "label": 1, "kingdom": "proka", "cds_tokens": [[f"P{i}"]] * 8})
    for i in range(30):
        rows.append({"kind": "background", "label": 0, "kingdom": "proka", "cds_tokens": [[f"B{i}"]] * 60})
    for i in range(20):
        rows.append({"kind": "chunk", "label": 0, "kingdom": "proka", "cds_tokens": [[f"C{i}"]] * 10})
    return pd.DataFrame(rows)


def _runs(lab: np.ndarray, ent: np.ndarray):
    """(entity, label, length) of maximal same-entity runs."""
    out, i = [], 0
    while i < len(ent):
        j = i
        while j < len(ent) and ent[j] == ent[i]:
            j += 1
        out.append((int(ent[i]), int(lab[i]), j - i))
        i = j
    return out


def test_clusters_off_is_unchanged_and_on_keeps_every_cds():
    e = _entities()
    a = compose(e, duplicate_positives=3, seed=0)[0]
    b = compose(e, duplicate_positives=3, seed=0, clusters=None)[0]
    assert a.cds_tokens == b.cds_tokens and (a.orf_entity == b.orf_entity).all()
    cfg = ClusterConfig(share=0.5, k=(2, 4), gap=(2, 10))
    c = compose(e, duplicate_positives=3, seed=0, clusters=cfg)[0]
    assert len(c.cds_tokens) == len(a.cds_tokens) and c.orf_label.sum() == a.orf_label.sum()
    assert sorted(map(tuple, c.cds_tokens)) == sorted(map(tuple, a.cds_tokens))
    assert c.cds_tokens == compose(e, duplicate_positives=3, seed=0, clusters=cfg)[0].cds_tokens  # deterministic


def test_clusters_put_short_background_gaps_between_positives():
    e = _entities()
    cfg = ClusterConfig(share=0.5, k=(2, 4), gap=(2, 10))
    c = compose(e, duplicate_positives=1, seed=1, clusters=cfg)[0]
    runs = _runs(c.orf_label, c.orf_entity)
    gaps = [r[2] for p, r, n in zip(runs, runs[1:], runs[2:], strict=False)
            if p[1] == 1 and n[1] == 1 and r[1] == 0 and e.at[r[0], "kind"] == "background" and r[2] <= 10]
    assert len(gaps) >= 8 and all(g >= 2 for g in gaps)  # 20 clustered positives → >= 8 gaps of 2-10 CDS
    base = _runs(*(lambda s: (s.orf_label, s.orf_entity))(compose(e, duplicate_positives=1, seed=1)[0]))
    short = sum(1 for p, r, n in zip(base, base[1:], base[2:], strict=False)
                if p[1] == 1 and n[1] == 1 and r[1] == 0 and r[2] <= 10)
    assert len(gaps) > short
