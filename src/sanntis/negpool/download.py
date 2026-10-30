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

"""NCBI assembly download (FTP over https), verified against the directory's ``md5checksums.txt``.

Prokaryotes need the nucleotide FASTA (genes are called with pyrodigal); eukaryotes need the RefSeq GenBank flat
file (the only eukaryote CDS source).
"""

from __future__ import annotations

import logging
import re
import urllib.request
from pathlib import Path

from sanntis.utils import fetch, file_digest, http_request

log = logging.getLogger(__name__)

FTP = "https://ftp.ncbi.nlm.nih.gov/genomes/all/"
_ACC = re.compile(r"^(GC[AF])_(\d{3})(\d{3})(\d{3})\.\d+$")


def _get(url: str, timeout: int = 120) -> str:
    with urllib.request.urlopen(http_request(url), timeout=timeout) as r:
        return r.read().decode()


def assembly_dir(accession: str) -> str:
    """https URL of the assembly directory ``.../GCA/000/220/375/GCA_000220375.1_<name>/``."""
    m = _ACC.match(accession)
    if not m:
        raise ValueError(f"not an assembly accession: {accession!r}")
    parent = f"{FTP}{m[1]}/{m[2]}/{m[3]}/{m[4]}/"
    names = re.findall(rf'href="({re.escape(accession)}_[^"/]+)/"', _get(parent))
    if not names:
        raise FileNotFoundError(f"{accession}: no assembly directory under {parent}")
    return f"{parent}{sorted(names)[-1]}/"


def download_assembly(accession: str, dest_dir: Path, kind: str, url: str = "") -> Path:
    """Download ``<dir>_genomic.{fna,gbff}.gz`` (kind ``fna`` | ``gbff``) into ``dest_dir``; md5-verified."""
    d = url.rstrip("/") + "/" if url else assembly_dir(accession)
    d = d.replace("ftp://", "https://")
    name = f"{d.rstrip('/').rsplit('/', 1)[-1]}_genomic.{kind}.gz"
    sums = dict((line.split()[1].lstrip("./"), line.split()[0]) for line in _get(d + "md5checksums.txt").splitlines()
                if len(line.split()) == 2)
    dest = Path(dest_dir) / name
    if dest.exists() and sums.get(name) == file_digest(dest, "md5"):
        return dest
    fetch(d + name, dest)
    if name in sums and file_digest(dest, "md5") != sums[name]:
        dest.unlink()
        raise OSError(f"{name}: md5 mismatch")
    return dest
