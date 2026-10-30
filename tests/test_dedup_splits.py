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

from sanntis.train.dedup import dedup, protein_tokens
from sanntis.train.splits import mc_split


def test_protein_tokens_rule():
    cds = [["G3DSA:1", "IPR1", "IPR2"], [], ["PF1"], ["IPR9"], ["IPR9", "x"]]
    assert protein_tokens(cds) == ["IPR1", "PF1", "IPR9"]


def test_dedup_keeps_longest():
    a = [["IPR1"], ["IPR2"], ["IPR3"], ["IPR4"]]
    loci = pd.DataFrame({"entity_id": ["A", "B", "C"], "length": [100, 200, 50],
                         "cds_tokens": [a, a, [["IPR7"], ["IPR8"]]]})
    d = dedup(loci)
    assert d.set_index("entity_id")["representative"].to_dict() == {"A": False, "B": True, "C": True}
    assert d.loc[0, "cluster"] == d.loc[1, "cluster"] != d.loc[2, "cluster"]


def test_mc_split():
    e = pd.DataFrame({"entity_id": [f"e{i}" for i in range(100)],
                      "kind": ["positive"] * 30 + ["chunk"] * 50 + ["background"] * 20,
                      "kingdom": ["proka"] * 100, "genome": [None] * 80 + [f"g{i % 7}" for i in range(20)]})
    r0, r1 = mc_split(e, 0), mc_split(e, 1)
    assert (r0[:80] == "val").sum() == 16
    assert not r0.equals(r1)
    # background: whole genomes on one side
    bg = e[e.kind == "background"].assign(role=r0)
    assert bg.groupby("genome")["role"].nunique().max() == 1
    np.testing.assert_array_equal(mc_split(e, 0), r0)  # deterministic
