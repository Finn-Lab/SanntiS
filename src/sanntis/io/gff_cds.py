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

"""Protein FASTA + GFF3 reader → CDS table: the CDS rows of the GFF, joined to the FASTA on ``ID``."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

import pandas as pd

from sanntis.io.fasta import InputError, finish_cds_table, open_text, read_protein_fasta

log = logging.getLogger(__name__)

_STRAND = {"+": 1, "-": -1}


@dataclass
class GffCdsStats:
    records: int = 0  # seqids with >= 1 CDS
    cds: int = 0
    compound: int = 0  # IDs on several rows (reported as their span)
    unmatched_proteins: int = 0  # FASTA records with no CDS (skipped)


def _attributes(col9: str) -> dict[str, str]:
    out = {}
    for kv in col9.strip().split(";"):
        if "=" in kv:
            k, v = kv.split("=", 1)
            out[unquote(k.strip())] = unquote(v.strip())
    return out


def read_faa_gff(faa: Path | str, gff: Path | str) -> tuple[pd.DataFrame, GffCdsStats]:
    """Every ``CDS`` row of ``gff``; its ``ID`` names the protein in ``faa``. Score and phase are ignored and rows
    may come in any order. An ``ID`` on several rows (same seqid and strand) is reported as its span. A CDS without
    a protein is an error; proteins without a CDS are skipped with a warning."""
    proteins = read_protein_fasta(faa)
    spans: dict[str, list] = {}  # ID → [seqid, start, end, strand, rows]
    with open_text(gff) as fh:
        for n, line in enumerate(fh, 1):
            if line.startswith("##FASTA"):
                break
            if line.startswith("#") or not line.strip():
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) != 9:
                raise InputError(f"{gff}:{n}: expected 9 tab-separated columns, got {len(f)}")
            if f[2] != "CDS":
                continue
            cid = _attributes(f[8]).get("ID")
            if not cid:
                raise InputError(f"{gff}:{n}: CDS without an ID attribute")
            if f[6] not in _STRAND:
                raise InputError(f"{gff}:{n}: strand must be + or -, got {f[6]!r}")
            seqid, start, end, strand = unquote(f[0]), int(f[3]), int(f[4]), _STRAND[f[6]]
            if not 1 <= start <= end:
                raise InputError(f"{gff}:{n}: bad coordinates {start}..{end}")
            s = spans.get(cid)
            if s is None:
                spans[cid] = [seqid, start, end, strand, 1]
            elif (s[0], s[3]) != (seqid, strand):
                raise InputError(f"{gff}:{n}: CDS {cid} spans several sequences or strands")
            else:
                s[1], s[2], s[4] = min(s[1], start), max(s[2], end), s[4] + 1
    if not spans:
        raise InputError(f"{gff}: no CDS features")
    missing = [c for c in spans if c not in proteins]
    if missing:
        raise InputError(f"{len(missing)} CDS of {gff} have no protein in {faa} (e.g. {', '.join(missing[:3])}); "
                         "the GFF ID must equal the FASTA id, and both files must come from the same gene calls")
    st = GffCdsStats(compound=sum(s[4] > 1 for s in spans.values()),
                     unmatched_proteins=len(proteins) - len(spans))
    if st.unmatched_proteins:
        log.warning("%s: %d proteins have no CDS in %s and are skipped", faa, st.unmatched_proteins, gff)
    if st.compound:
        log.info("%s: %d CDS on several rows (reported as their span)", gff, st.compound)
    cds = finish_cds_table([{"record_id": s[0], "locus_tag": cid, "start": s[1] - 1, "end": s[2], "strand": s[3],
                             "protein": proteins[cid]} for cid, s in spans.items()])
    st.records, st.cds = cds["record_id"].nunique(), len(cds)
    return cds, st
