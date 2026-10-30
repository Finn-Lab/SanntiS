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

from sanntis.negpool import sample as S
from sanntis.train.ncbi import lineage_group, lineage_of


def test_gtdb_sample(tmp_path):
    rows = []
    for p, n in (("A", 5), ("B", 1), ("C", 3)):
        for _ in range(n):
            rows.append({"accession": f"RS_GCF_00000{len(rows)}.1", "gtdb_representative": "t",
                         "gtdb_taxonomy": f"d__Bacteria;p__{p};c__x;o__;f__;g__;s__"})
    rows.append({"accession": "GB_GCA_9.1", "gtdb_representative": "f", "gtdb_taxonomy": "d__Bacteria;p__Z"})
    t = tmp_path / "bac.tsv.gz"
    with gzip.open(t, "wt") as fh:
        pd.DataFrame(rows).to_csv(fh, sep="\t", index=False)
    a = S.sample_gtdb([t], n_samples=5, seed=42)
    b = S.sample_gtdb([t], n_samples=5, seed=42)
    assert a.equals(b) and len(a) == 5
    phyla = a["lineage"].str.extract(r"p__(\w)")[0].value_counts()
    assert phyla["B"] == 1 and set(phyla.index) == {"A", "B", "C"}  # 1-2 per phylum
    assert a["accession"].str.startswith("GCF_").all() and (a["kingdom"] == "proka").all()


def test_refseq_sample(tmp_path):
    hdr = "#   See ftp://...\n#assembly_accession\ttaxid\tftp_path\tassembly_level\n"
    lines = [f"GCF_{i:09d}.1\t{100 + i % 3}\thttps://ftp/x{i}\tChromosome" for i in range(9)]
    p = tmp_path / "s.txt"
    p.write_text(hdr + "\n".join(lines) + "\n")
    lin = {100: {"phylum": "Ascomycota", "superkingdom": "Eukaryota"}, 101: {"phylum": "Basidiomycota"}, 102: {}}
    s = S.sample_refseq({"fungi": p}, lin)
    assert s["lineage"].str.contains("Ascomycota").sum() == 3 and len(s) == 6  # 3 per phylum, no-phylum dropped
    assert s.equals(S.sample_refseq({"fungi": p}, lin))


def test_taxonomy_parsing():
    t = {"TaxId": "5599", "Rank": "species", "ScientificName": "Alternaria alternata",
         "LineageEx": [{"Rank": "no rank", "ScientificName": "cellular organisms"},
                       {"Rank": "domain", "ScientificName": "Eukaryota"},
                       {"Rank": "phylum", "ScientificName": "Ascomycota"}]}
    lin = lineage_of(t)
    assert lin["phylum"] == "Ascomycota" and lin["superkingdom"] == "Eukaryota" and lin["species"].startswith("Alt")


def test_lineage_group():
    assert lineage_group({"superkingdom": "Bacteria"}) == "prokaryotes"
    assert lineage_group({"superkingdom": "Eukaryota", "kingdom": "Fungi"}) == "fungi"
    assert lineage_group({"superkingdom": "Eukaryota", "kingdom": "Viridiplantae"}) == "plants"
    assert lineage_group({"superkingdom": "Eukaryota", "phylum": "Apicomplexa"}) == "other eukaryotes"
    assert lineage_group({}) is None
