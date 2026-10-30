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
import shutil
from pathlib import Path

import pytest
from Bio.Seq import Seq
from Bio.SeqFeature import CompoundLocation, FeatureLocation, SeqFeature
from Bio.SeqIO import write as seqio_write
from Bio.SeqRecord import SeqRecord

from sanntis.annotate.genecall import call_genes
from sanntis.data import EXAMPLE_DIR
from sanntis.io.fasta import (
    CDS_TABLE_COLUMNS,
    InputError,
    clean_protein,
    input_kind,
    input_prefix,
    md5,
    read_nucleotide_fasta,
    read_protein_fasta,
)
from sanntis.io.genbank import read_genbank
from sanntis.io.gff_cds import read_faa_gff

FIX = Path(__file__).parent / "fixtures"


def test_input_kind_and_prefix():
    assert input_kind("a/x.fna.gz") == "fasta" and input_kind("x.GBK") == "genbank"
    assert input_kind("x.faa.gz") == "protein"
    assert input_prefix("dir/genome.fasta.gz") == "genome"
    with pytest.raises(InputError):
        input_kind("x.txt")


def test_clean_protein():
    assert clean_protein("mk*") == "MK"
    assert clean_protein("MK*L") == "MKXL"  # internal stop → X (IPS6 alphabet)
    assert md5("MK") == md5(clean_protein("mk*"))


def test_genbank_fixture():
    cds, st = read_genbank(FIX / "BGC0003066.gbk")
    assert list(cds.columns) == CDS_TABLE_COLUMNS
    assert st.records == 1 and len(cds) == st.cds == 19
    assert cds["cds_idx"].tolist() == list(range(19))
    assert (cds["start"].diff().dropna() >= 0).all()
    assert cds["locus_tag"].iloc[0] == "SLNWT_0579" and cds["strand"].iloc[0] == -1
    assert (cds["start"].iloc[0], cds["end"].iloc[0]) == (0, 1146)  # 0-based half-open


def _record(features):
    r = SeqRecord(Seq("ATG" * 400), id="rec1", annotations={"molecule_type": "DNA"})
    r.features = features
    return r


def test_genbank_joined_pseudo_and_gzip(tmp_path):
    joined = SeqFeature(CompoundLocation([FeatureLocation(100, 200, 1), FeatureLocation(300, 400, 1)]), type="CDS",
                        qualifiers={"locus_tag": ["j1"], "translation": ["MKV*"]})
    pseudo = SeqFeature(FeatureLocation(500, 600, -1), type="CDS", qualifiers={"locus_tag": ["p1"], "pseudo": [""]})
    plain = SeqFeature(FeatureLocation(10, 40, -1), type="CDS", qualifiers={"translation": ["MAA"]})
    p = tmp_path / "x.gbk.gz"
    with gzip.open(p, "wt") as fh:
        seqio_write(_record([joined, pseudo, plain]), fh, "genbank")
    cds, st = read_genbank(p)
    assert st.pseudo_skipped == 1 and st.compound == 1
    # a CDS without locus_tag / protein_id / gene is named by its genomic position
    assert cds[["locus_tag", "start", "end"]].values.tolist() == [["rec1_cds1", 10, 40], ["j1", 100, 400]]
    assert cds["protein"].tolist() == ["MAA", "MKV"]


def test_genbank_missing_translation_fails(tmp_path):
    f = SeqFeature(FeatureLocation(10, 40, 1), type="CDS", qualifiers={"locus_tag": ["t1"]})
    p = tmp_path / "x.gbk"
    seqio_write(_record([f]), p, "genbank")
    with pytest.raises(InputError, match="without /translation"):
        read_genbank(p)


def test_fasta_and_genecall(tmp_path):
    gz = tmp_path / "c.fna.gz"
    with open(FIX / "BGC0002866.fna", "rb") as a, gzip.open(gz, "wb") as b:
        shutil.copyfileobj(a, b)
    recs = read_nucleotide_fasta(gz)
    cds = call_genes(recs, "meta")
    assert len(cds) > 5
    assert cds["locus_tag"].iloc[0] == "BGC0002866_1"
    assert not cds["protein"].str.contains(r"\*").any()
    with pytest.raises(InputError, match="20000"):
        call_genes(recs, "single")


# 3 contigs of a combined Pyrodigal + FragGeneScanRS gene call (FAA length-sorted)
SLICE = EXAMPLE_DIR / "ERZ29794225_slice"


def test_faa_gff_slice():
    cds, st = read_faa_gff(SLICE.with_suffix(".faa"), SLICE.with_suffix(".gff"))
    assert list(cds.columns) == CDS_TABLE_COLUMNS
    assert st.records == 3 and len(cds) == st.cds == 24 and st.compound == 0 and st.unmatched_proteins == 0
    for _, g in cds.groupby("record_id"):  # genomic order whatever the file order
        assert g["cds_idx"].tolist() == list(range(len(g))) and (g["start"].diff().dropna() >= 0).all()
    by_tag = cds.set_index("locus_tag")
    assert tuple(by_tag.loc["ERZ29794225_46_1", ["start", "end", "strand"]]) == (237, 633, 1)
    assert tuple(by_tag.loc["ERZ29794225_46_38_226_+", ["start", "end", "strand"]]) == (37, 226, 1)
    ips_md5 = {line.split("\t")[0]: line.split("\t")[1] for line in SLICE.with_suffix(".ips.tsv").open()}
    assert len(ips_md5) == 20 and all(by_tag.loc[k, "md5"] == v for k, v in ips_md5.items())


def _faa_gff(tmp_path, proteins: dict, rows: list[str], tail: str = ""):
    faa, gff = tmp_path / "p.faa", tmp_path / "p.gff"
    faa.write_text("".join(f">{k} some description\n{v}\n" for k, v in proteins.items()))
    gff.write_text("##gff-version 3\n" + "".join(r + "\n" for r in rows) + tail)
    return faa, gff


def test_faa_gff_rules(tmp_path, caplog):
    rows = ["c1\tx\tgene\t1\t90\t.\t+\t.\tID=g1",
            "c1\tx\tCDS\t200\t290\t.\t-\t0\tID=b%3B1;Parent=g2",  # percent-encoded ID
            "c1\tx\tCDS\t1\t30\t.\t+\t.\tID=a1",
            "c1\tx\tCDS\t61\t90\t.\t+\t.\tID=a1"]  # one CDS on two rows
    faa, gff = _faa_gff(tmp_path, {"a1": "MKV*", "b;1": "mll", "extra": "MA"}, rows,
                        tail="##FASTA\n>c1\nACGT\n")
    cds, st = read_faa_gff(faa, gff)
    assert cds[["locus_tag", "start", "end", "strand", "protein"]].values.tolist() == [
        ["a1", 0, 90, 1, "MKV"], ["b;1", 199, 290, -1, "MLL"]]
    assert st.compound == 1 and st.unmatched_proteins == 1 and "1 proteins have no CDS" in caplog.text

    for bad, match in [(["c1\tx\tCDS\t1\t30\t.\t.\t.\tID=a1"], "strand"),
                       (["c1\tx\tCDS\t30\t1\t.\t+\t.\tID=a1"], "coordinates"),
                       (["c1\tx\tCDS\t1\t30\t.\t+\t.\tName=a1"], "without an ID"),
                       (["c1\tx\tCDS\t1\t30\t.\t+\t.\tID=zz"], "no protein"),
                       (["c1\tx\tCDS\t1\t30\t.\t+\t.\tID=a1", "c2\tx\tCDS\t1\t30\t.\t+\t.\tID=a1"], "several"),
                       (["c1\tx\tCDS\t1\t30"], "9 tab-separated"),
                       (["c1\tx\tgene\t1\t30\t.\t+\t.\tID=a1"], "no CDS")]:
        with pytest.raises(InputError, match=match):
            read_faa_gff(*_faa_gff(tmp_path, {"a1": "MK"}, bad))
    faa.write_text(">a1\nMK\n>a1\nMK\n")
    with pytest.raises(InputError, match="duplicate protein id"):
        read_protein_fasta(faa)
