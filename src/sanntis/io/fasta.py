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

"""FASTA I/O: nucleotide records for gene calling, protein FASTA out, and the shared CDS-table helpers.

A *CDS table* (one per input) has the columns ``CDS_TABLE_COLUMNS``: ``start`` / ``end`` are 0-based half-open on the
record, ``strand`` is +1 / -1, ``protein`` the cleaned translation and ``md5`` its lower-case hex md5 (the
annotation cache key, and the IPS TSV fallback join key).
"""

from __future__ import annotations

import gzip
import hashlib
import re
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

from sanntis.io.gff import escape

CDS_TABLE_COLUMNS = ["record_id", "cds_idx", "locus_tag", "start", "end", "strand", "protein", "md5"]
FASTA_SUFFIXES = (".fna", ".fa", ".fasta", ".fas", ".ffn")
GENBANK_SUFFIXES = (".gbk", ".gb", ".gbff", ".genbank")
PROTEIN_SUFFIXES = (".faa",)

# InterProScan 6 accepts only these residues (FastaFile.PROTEIN_ALPHABET); anything else becomes X.
_NOT_PROTEIN = re.compile(r"[^ACDEFGHIKLMNPQRSTVWYUOXBZJ]")


class InputError(ValueError):
    pass


def strip_gz(path: Path | str) -> str:
    name = Path(path).name
    return name[:-3] if name.lower().endswith(".gz") else name


def input_kind(path: Path | str) -> str:
    """``fasta`` | ``genbank`` | ``protein`` from the file name (``.gz`` allowed)."""
    name = strip_gz(path).lower()
    if name.endswith(FASTA_SUFFIXES):
        return "fasta"
    if name.endswith(GENBANK_SUFFIXES):
        return "genbank"
    if name.endswith(PROTEIN_SUFFIXES):
        return "protein"
    raise InputError(f"{path}: unsupported input; expected "
                     f"{'/'.join(FASTA_SUFFIXES + GENBANK_SUFFIXES + PROTEIN_SUFFIXES)} (optionally .gz)")


def input_prefix(path: Path | str) -> str:
    """Output prefix: the file name without ``.gz`` and its last extension."""
    name = strip_gz(path)
    return name.rsplit(".", 1)[0] if "." in name else name


def open_text(path: Path | str):
    return gzip.open(path, "rt") if str(path).lower().endswith(".gz") else open(path)


def read_nucleotide_fasta(path: Path | str) -> list[SeqRecord]:
    with open_text(path) as fh:
        records = list(SeqIO.parse(fh, "fasta"))
    if not records:
        raise InputError(f"{path}: no FASTA records")
    ids = [r.id for r in records]
    if len(set(ids)) != len(ids):
        raise InputError(f"{path}: duplicate sequence ids")
    return records


def read_protein_fasta(path: Path | str) -> dict[str, str]:
    """id (first header token) → sequence, as written (cleaning happens in ``finish_cds_table``)."""
    seqs: dict[str, str] = {}
    for rid, s in iter_fasta(path):
        if rid in seqs:
            raise InputError(f"{path}: duplicate protein id {rid}")
        seqs[rid] = s
    if not seqs:
        raise InputError(f"{path}: no FASTA records")
    return seqs


def clean_protein(seq: str) -> str:
    """Upper case, terminal stop removed, residues outside the InterProScan alphabet → ``X``."""
    s = seq.strip().upper().rstrip("*")
    return _NOT_PROTEIN.sub("X", s)


def md5(seq: str) -> str:
    return hashlib.md5(seq.encode()).hexdigest()


def finish_cds_table(rows: list[dict]) -> pd.DataFrame:
    """Rows (record_id, locus_tag, start, end, strand, protein) → sorted CDS table with ``cds_idx`` and ``md5``."""
    df = pd.DataFrame(rows, columns=[c for c in CDS_TABLE_COLUMNS if c not in ("cds_idx", "md5")])
    if df.empty:
        return pd.DataFrame(columns=CDS_TABLE_COLUMNS)
    df["protein"] = df["protein"].map(clean_protein)
    df["md5"] = df["protein"].map(md5)
    order = {r: i for i, r in enumerate(dict.fromkeys(df["record_id"]))}
    df = (df.assign(_r=df["record_id"].map(order)).sort_values(["_r", "start", "end"], kind="stable")
          .drop(columns="_r").reset_index(drop=True))
    df["cds_idx"] = df.groupby("record_id", sort=False).cumcount()
    return df[CDS_TABLE_COLUMNS]


def iter_fasta(path: Path | str) -> Iterator[tuple[str, str]]:
    with open_text(path) as fh:
        for r in SeqIO.parse(fh, "fasta"):
            yield r.id, str(r.seq)


def write_protein_fasta(items, path: Path, width: int = 60) -> int:
    """``items``: (header, sequence) pairs. Returns the number written."""
    n = 0
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for h, s in items:
            fh.write(f">{h}\n")
            for i in range(0, len(s), width):
                fh.write(s[i:i + width] + "\n")
            n += 1
    return n


def write_cds_faa(cds: pd.DataFrame, path: Path) -> None:
    strand = {1: "+", -1: "-"}
    write_protein_fasta(((f"{r.locus_tag} {r.record_id}:{r.start + 1}-{r.end}({strand.get(r.strand, '.')}) "
                          f"md5={r.md5}", r.protein) for r in cds.itertuples(index=False)), path)


def write_cds_gff(cds: pd.DataFrame, path: Path, source: str) -> None:
    """The CDS used for prediction as GFF3 (``source``: the gene caller or the input GenBank)."""
    strand = {1: "+", -1: "-"}
    with open(path, "w") as fh:
        fh.write("##gff-version 3\n")
        for r in cds.itertuples(index=False):
            fh.write(f"{r.record_id}\t{source}\tCDS\t{r.start + 1}\t{r.end}\t.\t{strand.get(r.strand, '.')}\t0\t"
                     f"ID={escape(r.locus_tag)};md5={r.md5}\n")
