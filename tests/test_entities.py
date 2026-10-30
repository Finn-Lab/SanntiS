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

import pandas as pd

from sanntis.train.entities import entity_table, negative_entities, positive_entities, primary_class


def _pool():
    genomes = pd.DataFrame({"genome": ["g1", "g2"], "kingdom": ["proka", "proka"]})
    rows = []
    for g in ("g1", "g2"):
        for i in range(60):
            t = None if (g == "g2" and i == 30) else ([] if i % 7 == 0 else [f"IPR{i % 13}"])
            rows.append((g, "c1", i, t))
    cds = pd.DataFrame(rows, columns=["genome", "contig", "cds_idx", "tokens"])
    runs = pd.DataFrame({"genome": ["g1", "g2", "g2"], "contig": ["c1"] * 3, "run_id": ["r0", "r0", "r1"],
                         "cds_idx": [list(range(0, 60)), list(range(0, 40)), list(range(40, 60))],
                         "n_cds": [60, 40, 20]})
    backbones = pd.DataFrame({"genome": ["g2"], "contig": ["c1"], "run_id": ["r1"], "kingdom": ["proka"]})
    return {"genomes": genomes, "cds": cds, "runs": runs, "backbones": backbones}


def test_primary_class():
    assert primary_class("terpene;PKS") == "PKS" and primary_class("other;ribosomal") == "other"
    assert primary_class("NRPS;PKS") == "NRPS" and primary_class(None) is None


def test_entities_end_to_end():
    loci = pd.DataFrame({"entity_id": ["BGC1", "BGC2", "BGC3", "BGC4"], "kingdom": "proka",
                         "class_raw": ["NRPS;PKS", "terpene", "terpene", "other"],
                         "length": [1000, 900, 1200, 300_000],
                         "cds_tokens": [[["IPR1"], ["IPR2"], ["IPR3"]], [["IPR5"], ["IPR6"]],
                                        [["IPR5"], ["IPR6"]], [["IPR9"]]]})
    pos, st = positive_entities(loci)
    assert st["excluded_too_long"] == 1 and list(pos["entity_id"]) == ["BGC1", "BGC3"]  # BGC2 dup of longer BGC3
    neg, ns = negative_entities(_pool(), pos)
    ch = neg[neg["kind"] == "chunk"]
    assert ns["chunks_sampled_proka"] == 6 and len(ch) == 6
    bg = neg[neg["kind"] == "background"]
    assert bg["n_cds"].min() >= 5
    assert not bg["entity_id"].str.contains(":r1:").any()  # test backbone excluded
    t = entity_table(pos, neg)
    assert list(t["kind"].iloc[:2]) == ["positive", "positive"]
