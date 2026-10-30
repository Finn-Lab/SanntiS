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

import json

import pandas as pd

from sanntis.io.antismash import write_sideload
from sanntis.predict import REGION_COLUMNS


def _regions(rows):
    base = {c: None for c in REGION_COLUMNS} | {"score": 0.9, "p_non_bgc": 0.1, "n_orfs": 3}
    return pd.DataFrame([base | r for r in rows], columns=REGION_COLUMNS)


def test_sideload(tmp_path):
    regions = _regions([
        {"record_id": "c1", "region": 1, "start": 101, "end": 2000, "bgc_class": "NRPS",
         "npc_pathway": "Polyketides|Terpenoids", "npc_pathway_score": 0.8, "npc_pathway_confident": True,
         "npc_superclass": "unassigned", "npc_superclass_top": "x", "npc_superclass_score": 0.2,
         "npc_pathway_key_domains_support": "Polyketides:IPR1+IPR2|Terpenoids:none",
         "npc_pathway_key_domains_unsupported": True},
        {"record_id": "c1", "region": 2, "start": 1, "end": 50}])
    path = tmp_path / "x.json"
    write_sideload(regions, ["c1", "c2"], path, configuration={"kingdom": "proka", "npc-precision": 0.9})
    d = json.loads(path.read_text())
    assert d["tool"]["name"] == "SanntiS" and d["tool"]["configuration"] == {"kingdom": "proka", "npc-precision": "0.9"}
    assert [r["name"] for r in d["records"]] == ["c1", "c2"] and d["records"][1]["subregions"] == []
    s1, s2 = d["records"][0]["subregions"]
    assert (s1["start"], s1["end"], s1["label"]) == (100, 2000, "NRPS")
    assert s1["details"]["ID"] == "c1_sanntis_1" and s1["details"]["score"] == "0.9000"
    assert s1["details"]["npc_pathway"] == ["Polyketides", "Terpenoids"]
    assert s1["details"]["npc_pathway_key_domains_support"] == ["Polyketides:IPR1+IPR2", "Terpenoids:none"]
    assert s1["details"]["npc_pathway_key_domains_unsupported"] == "true"
    assert (s2["start"], s2["label"]) == (0, "SanntiS") and "npc_pathway" not in s2["details"]


def test_sideload_no_regions(tmp_path):
    path = tmp_path / "x.json"
    write_sideload(_regions([]), ["c1"], path)
    assert json.loads(path.read_text())["records"] == [{"name": "c1", "subregions": []}]
