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

"""GenBank → CDS, one reader for inference and training.

- ``read_genbank``: a ``predict`` input → CDS table (the only eukaryote path: CDS must carry ``/translation``).
- ``iter_cds``: the CDS of one record, for training data (MIBiG and external BGCs, negative-pool genomes).

Both read coordinates, ids and translations the same way (:func:`iter_cds`); they differ in two rules only, set by
its arguments: a CDS without ``/translation`` (inference: an error unless it is a pseudo gene; training: translated
from the record, or an error for external data) and ``/pseudo`` CDS (training drops them; inference keeps them when
they carry a translation).
"""

from __future__ import annotations

import io
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd
from Bio import SeqIO
from Bio.SeqFeature import SeqFeature
from Bio.SeqRecord import SeqRecord

from sanntis.io.fasta import InputError, clean_protein, finish_cds_table, open_text
from sanntis.io.fasta import md5 as protein_md5

log = logging.getLogger(__name__)

CDS_COLUMNS = ["contig", "cds_idx", "cds_id", "locus_tag", "protein_id", "start", "end", "strand", "md5", "length"]


@dataclass
class Cds:
    contig: str
    cds_idx: int
    cds_id: str
    locus_tag: str
    protein_id: str
    start: int  # 0-based half-open
    end: int
    strand: int
    sequence: str  # cleaned as for inference (clean_protein); "" without /translation (missing_translation="empty")
    pseudo: bool = False  # /pseudo or /pseudogene
    joined: bool = False  # joined / order location, reported as its span

    @property
    def md5(self) -> str:
        return protein_md5(self.sequence)


def read_records(path: Path | str | bytes) -> list[SeqRecord]:
    if isinstance(path, bytes):
        return list(SeqIO.parse(io.StringIO(path.decode()), "genbank"))
    with open_text(Path(path)) as fh:
        return list(SeqIO.parse(fh, "genbank"))


def _q(f: SeqFeature, key: str) -> str:
    return (f.qualifiers.get(key) or [""])[0].strip()


def _span(f: SeqFeature) -> tuple[int, int, int]:
    """(start, end, strand) of a CDS. A joined location is reported as its span (min start, max end); for a location
    that wraps the origin of a circular record, the span of its longest part."""
    parts = getattr(f.location, "parts", [f.location])
    strand = f.location.strand if f.location.strand in (1, -1) else 1
    if len(parts) > 1:
        starts = [int(p.start) for p in parts][::strand]
        if any(b < a for a, b in zip(starts, starts[1:], strict=False)):  # wraps the origin
            parts = [max(parts, key=len)]
    return min(int(p.start) for p in parts), max(int(p.end) for p in parts), strand


def _translate(f: SeqFeature, rec: SeqRecord) -> str:
    table = int(_q(f, "transl_table") or 11)
    codon_start = int(_q(f, "codon_start") or 1)
    nt = f.location.extract(rec.seq)[codon_start - 1:]
    nt = nt[: len(nt) - len(nt) % 3]
    return str(nt.translate(table=table, to_stop=True))


def iter_cds(rec: SeqRecord, *, missing_translation: Literal["translate", "error", "empty"] = "translate",
             drop_pseudo: bool = True) -> Iterator[Cds]:
    """CDS of one record in genomic order (start, end), with their protein sequences.

    ``missing_translation``: a CDS without ``/translation`` is translated from the record, raises
    (external data), or comes with an empty sequence (the caller decides). ``drop_pseudo``: skip ``/pseudo`` CDS
    (``/pseudogene`` ones are kept).
    """
    feats = [f for f in rec.features if f.type == "CDS" and not (drop_pseudo and "pseudo" in f.qualifiers)]
    spans = sorted(((_span(f), f) for f in feats), key=lambda x: (x[0][0], x[0][1]))
    for i, ((start, end, strand), f) in enumerate(spans):
        seq = _q(f, "translation")
        lt, pid = _q(f, "locus_tag"), _q(f, "protein_id")
        cid = lt or pid or _q(f, "gene") or f"{rec.id}_cds{i + 1}"
        if not seq and missing_translation == "error":
            raise InputError(f"{rec.id}: CDS {cid} has no /translation")
        if not seq and missing_translation == "translate":
            seq = _translate(f, rec)
        yield Cds(rec.id, i, cid, lt, pid, start, end, strand, clean_protein(seq) if seq else "",
                  pseudo="pseudo" in f.qualifiers or "pseudogene" in f.qualifiers,
                  joined=len(getattr(f.location, "parts", [f.location])) > 1)


def cds_rows(cds: list[Cds]) -> list[dict]:
    return [{"contig": c.contig, "cds_idx": c.cds_idx, "cds_id": c.cds_id, "locus_tag": c.locus_tag,
             "protein_id": c.protein_id, "start": c.start, "end": c.end, "strand": c.strand, "md5": c.md5,
             "length": len(c.sequence)} for c in cds]


# ── inference input ──────────────────────────────────────────────────────────


@dataclass
class GenbankStats:
    records: int = 0
    record_lengths: dict[str, int] = field(default_factory=dict)
    cds: int = 0
    pseudo_skipped: int = 0  # pseudo CDS without /translation
    compound: int = 0  # joined / order locations (reported as their span)


def read_genbank(path: Path | str) -> tuple[pd.DataFrame, GenbankStats]:
    """Every CDS of every record (see :func:`iter_cds` for coordinates). A CDS without ``/translation`` is an error
    unless it is a pseudo gene (skipped and counted)."""
    st = GenbankStats()
    rows: list[dict] = []
    missing: list[str] = []
    seen: set[str] = set()
    with open_text(path) as fh:
        for rec in SeqIO.parse(fh, "genbank"):
            if rec.id in seen:
                raise InputError(f"{path}: duplicate record id {rec.id}")
            seen.add(rec.id)
            st.records += 1
            st.record_lengths[rec.id] = len(rec.seq)
            for c in iter_cds(rec, missing_translation="empty", drop_pseudo=False):
                if not c.sequence:
                    if c.pseudo:
                        st.pseudo_skipped += 1
                    else:
                        missing.append(f"{rec.id}:{c.cds_id}")
                    continue
                st.compound += c.joined
                rows.append({"record_id": rec.id, "locus_tag": c.cds_id, "start": c.start, "end": c.end,
                             "strand": c.strand, "protein": c.sequence})
    if st.records == 0:
        raise InputError(f"{path}: no GenBank records")
    if missing:
        raise InputError(f"{path}: {len(missing)} CDS without /translation (e.g. {', '.join(missing[:3])}). "
                         "sanntis needs translated CDS in GenBank inputs: annotate the file first, or pass the "
                         "nucleotide sequence as .fna (prokaryotes)")
    if st.pseudo_skipped:
        log.warning("%s: %d pseudo CDS without /translation skipped", path, st.pseudo_skipped)
    if st.compound:
        log.info("%s: %d CDS with joined locations (reported as their span)", path, st.compound)
    cds = finish_cds_table(rows)
    st.cds = len(cds)
    return cds, st
