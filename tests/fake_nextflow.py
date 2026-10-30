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

"""Stand-in for the `nextflow` executable (tests only): replays a recorded InterProScan 6 TSV.

`-version` prints a version line. `run ...` appends its argv (JSON) to $FAKE_NF_CALLS, creates the IPS6 data layout
under --datadir (from tests/fixtures/ips6_databases.json), and writes <outdir>/<outprefix>.tsv with the rows of
$FAKE_IPS_TSV whose md5 matches an input sequence (col 1 replaced by the FASTA id), restricted to --applications.
"""

import hashlib
import json
import os
import re
import sys
from pathlib import Path

FIX = Path(__file__).parent / "fixtures"
DATA_DIRS = {"Pfam": "pfam", "NCBIFAM": "ncbifam", "SFLD": "sfld", "SUPERFAMILY": "superfamily", "CATH-Gene3D": "cath",
             "HAMAP": "hamap", "CDD": "cdd", "PRINTS": "prints", "PIRSF": "pirsf", "PROSITE profiles": "prosite",
             "SMART": "smart"}


def norm(s: str) -> str:
    return re.sub(r"[-_\s]", "", s).casefold()


def main(argv: list[str]) -> int:
    if "-version" in argv:
        print(f"      version {os.environ.get('FAKE_NF_VERSION', '26.10.3')} build 5000")
        return 0
    with open(os.environ["FAKE_NF_CALLS"], "a") as fh:
        fh.write(json.dumps(argv) + "\n")
    if os.environ.get("FAKE_NF_FAIL"):
        return 1
    p = {argv[i][2:]: argv[i + 1] for i in range(len(argv) - 1) if argv[i].startswith("--")}
    dd = Path(p["datadir"])
    dbs = json.loads((FIX / "ips6_databases.json").read_text())
    (dd / "interpro" / p["interpro"]).mkdir(parents=True, exist_ok=True)
    (dd / "interpro" / p["interpro"] / "databases.json").write_text(json.dumps(dbs))
    for name, d in DATA_DIRS.items():
        (dd / d / dbs[name]).mkdir(parents=True, exist_ok=True)
        (dd / d / dbs[name] / "data").write_text("x")
    seqs, sid = {}, None
    for line in Path(p["input"]).read_text().splitlines():
        if line.startswith(">"):
            sid = line[1:].split()[0]
            seqs[sid] = ""
        else:
            seqs[sid] += line.strip()
    by_md5 = {hashlib.md5(s.upper().encode()).hexdigest(): i for i, s in seqs.items()}
    apps = {norm(a) for a in p["applications"].split(",")}
    out = Path(p["outdir"])
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{p.get('outprefix', Path(p['input']).name)}.tsv", "w") as fh:
        for line in Path(os.environ["FAKE_IPS_TSV"]).read_text().splitlines():
            s = line.split("\t")
            if s[1].lower() in by_md5 and norm(s[3]) in apps:
                fh.write("\t".join([by_md5[s[1].lower()]] + s[1:]) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
