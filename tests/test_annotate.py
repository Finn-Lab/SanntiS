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
from pathlib import Path

import pandas as pd
import pytest

from sanntis.annotate import ips6
from sanntis.annotate.cache import AnnotationCache
from sanntis.config import NextflowOptions
from sanntis.utils import cache_dir

FIX = Path(__file__).parent / "fixtures"


def _hits(rows):
    return pd.DataFrame(rows, columns=["md5", "app", "signature", "start", "end", "ipr"])


def test_cache_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("SANNTIS_CACHE", str(tmp_path / "c"))
    assert cache_dir() == tmp_path / "c"
    c = AnnotationCache()
    assert "ips6.0.3.0_interpro107.0_" in str(c.dir)
    assert c.lookup(["a", "b"]) == (set(), {"a", "b"})
    c.add(["a", "b"], _hits([("a", "pfam", "PF00001", 1, 50, "IPR000001"), ("a", "coils", "Coil", 60, 80, None)]))
    cached, missing = c.lookup(["a", "b", "c"])
    assert cached == {"a", "b"} and missing == {"c"}  # b had no hit and is still cached
    h = c.hits(["a", "b"])
    assert len(h) == 2 and h["ipr"].isna().sum() == 1
    # the same protein annotated again (concurrent run): the first shard wins, no duplicate rows
    c.add(["a", "c"], _hits([("a", "pfam", "PF00001", 1, 50, "IPR000001"), ("c", "smart", "SM1", 1, 9, None)]))
    assert len(c.hits(["a"])) == 2 and len(c.hits(["a", "c"])) == 3
    assert c.known() == {"a", "b", "c"} and len(c.hits(["a", "b", "c"])) == 3
    with pytest.raises(ValueError):
        c.add(["x"], _hits([("y", "pfam", "PF1", 1, 2, None)]))


def test_app_names_and_pins():
    names = ips6.ips6_app_names()
    assert names == ["pfam", "ncbifam", "sfld", "superfamily", "cathgene3d", "hamap", "coils", "cdd", "prints",
                     "pirsf", "prositeprofiles", "prositepatterns", "mobidblite", "smart"]
    with pytest.raises(ips6.Ips6Error, match="interpro"):
        ips6.check_nf_args(NextflowOptions(nf_args="--interpro 108.0"))
    ips6.check_nf_args(NextflowOptions(nf_args="-with-trace --cpus 4"))


def test_command_is_pinned(tmp_path):
    run = ips6.nextflow_run(tmp_path / "p.faa", tmp_path / "out", tmp_path / "data", tmp_path,
                            apps=["pfam", "gene3d"])
    cmd = run.command(NextflowOptions(container_engine="singularity", nf_args="-with-trace"))
    assert cmd[cmd.index("-r") + 1] == "6.0.3.0"
    assert cmd[cmd.index("--interpro") + 1] == "107.0"
    assert cmd[cmd.index("--applications") + 1] == "pfam,cathgene3d"
    assert "--noMatchesApi" in cmd and cmd[cmd.index("--formats") + 1] == "tsv"
    assert cmd[-1] == "-with-trace"


def test_missing_data(tmp_path):
    assert ips6.missing_data(tmp_path) == ["interpro/107.0/databases.json"]
    dbs = json.loads((FIX / "ips6_databases.json").read_text())
    (tmp_path / "interpro" / "107.0").mkdir(parents=True)
    (tmp_path / "interpro" / "107.0" / "databases.json").write_text(json.dumps(dbs))
    miss = ips6.missing_data(tmp_path)
    assert any(m.startswith("gene3d (cath/4.3.0)") for m in miss) and not any("coils" in m for m in miss)
    for d, v in [("pfam", "38.0"), ("ncbifam", "17.0"), ("sfld", "4"), ("superfamily", "1.75"), ("cath", "4.3.0"),
                 ("hamap", "2025_01"), ("cdd", "3.21"), ("prints", "42.0"), ("pirsf", "3.10"),
                 ("prosite", "2025_01"), ("smart", "9.0")]:
        (tmp_path / d / v).mkdir(parents=True)
        (tmp_path / d / v / "x").write_text("x")
    assert ips6.missing_data(tmp_path) == []
