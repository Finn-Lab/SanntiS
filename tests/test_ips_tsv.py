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

import gzip

import pandas as pd
import pytest

from sanntis.io.ips_tsv import IpsTsvError, canonical_app, check_apps, detect_version, read_ips_tsv, tokens_by_key

IPS5 = [
    "p1\tmd5a\t300\tPfam\tPF00109\tketoacyl-synt\t10\t200\t1E-30\tT\t15-10-2026\tIPR014030\tKS N\t-\t-",
    "p1\tmd5a\t300\tGene3D\tG3DSA:3.40.47.10\t-\t5\t250\t1E-50\tT\t15-10-2026\t-\t-\t-\t-",
    "p1\tmd5a\t300\tPANTHER\tPTHR11712\t-\t1\t290\t1E-90\tT\t15-10-2026\t-\t-\t-\t-",
    "p1\tmd5a\t300\tCoils\tCoil\t-\t260\t280\t-\tT\t15-10-2026\t-\t-\t-\t-",
    "p2\tmd5b\t120\tSUPERFAMILY\tSSF53901\t-\t5\t100\t1E-20\tT\t15-10-2026\tIPR016039\t-\t-\t-",
    "p2\tmd5b\t120\tMobiDBLite\tmobidb-lite\t-\t1\t20\t-\tT\t15-10-2026\t-\t-\t-\t-",
]


def _write(tmp_path, rows, gz=False):
    p = tmp_path / ("x.tsv.gz" if gz else "x.tsv")
    with (gzip.open(p, "wt") if gz else open(p, "w")) as fh:
        fh.write("\n".join(rows) + "\n")
    return p


def test_canonical_names():
    assert canonical_app("CATH-Gene3D") == canonical_app("Gene3D") == "gene3d"
    assert canonical_app("PROSITE profiles") == canonical_app("ProSiteProfiles") == "prositeprofiles"
    assert canonical_app("MobiDB-lite") == canonical_app("MobiDBLite") == canonical_app("MobiDB Lite")
    assert canonical_app("NCBIFAM") == canonical_app("NCBIfam") == "ncbifam"


def test_ips5_parse_filter_tokens(tmp_path):
    h = read_ips_tsv(_write(tmp_path, IPS5, gz=True))
    assert h.version == "5"
    assert h.dropped["PANTHER"] == 1  # never allowed
    assert set(h.hits["app"]) == {"pfam", "gene3d", "coils", "superfamily", "mobidblite"}
    sm = pd.DataFrame({"signature": ["PF00109", "G3DSA:3.40.47.10", "SSF53901"], "app": ["pfam", "gene3d", "sf"],
                       "entry": ["IPR014030", None, "IPR016039"]})
    t = tokens_by_key(h.hits, sm)
    # ordered by hit start; unintegrated / unmapped signatures keep their accession
    assert t == {"p1": ["G3DSA:3.40.47.10", "IPR014030", "Coil"], "p2": ["mobidb-lite", "IPR016039"]}
    assert tokens_by_key(h.hits, sm, key="md5")["md5b"] == t["p2"]


def test_version_detection():
    assert detect_version({"Pfam", "Gene3D"}, {"T"}) == "5"
    assert detect_version({"Pfam", "CATH-Gene3D"}, {"Pfam", "CATH-Gene3D"}) == "6"
    with pytest.raises(IpsTsvError):
        detect_version({"Gene3D", "CATH-Gene3D"}, {"T", "Gene3D"})


def test_missing_required_apps(tmp_path):
    rows = [f"p{i}\tm{i}\t100\tPfam\tPF00001\t-\t1\t50\t1E-5\tT\t-\tIPR000001\t-\t-\t-" for i in range(150)]
    h = read_ips_tsv(_write(tmp_path, rows))
    with pytest.raises(IpsTsvError, match="required"):
        check_apps(h)
    assert "gene3d" in check_apps(h, allow_incomplete=True)


def test_ips6_upper_case_md5(tmp_path):
    rows = ["sp|P1|X\t0001C5D32DCCB8E03267C370872E1694\t100\tCATH-Gene3D\tG3DSA:1.10.10.10\t-\t1\t50\t1E-5\t"
            "CATH-Gene3D\t21-10-2026\t-\t-\t-\t-"]
    h = read_ips_tsv(_write(tmp_path, rows))
    assert h.version == "6" and h.hits["md5"].iloc[0] == "0001c5d32dccb8e03267c370872e1694"
    assert h.hits["app"].iloc[0] == "gene3d"
