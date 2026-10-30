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

"""Prokaryotic gene calling with pyrodigal 3: ``meta`` (default) or ``single`` (trained on the input genome)."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import pandas as pd
import pyrodigal
from Bio.SeqRecord import SeqRecord

from sanntis.io.fasta import InputError, finish_cds_table

log = logging.getLogger(__name__)

MIN_SINGLE_BP = 20_000  # pyrodigal / Prodigal need >= 20 kb of sequence to train


def gene_finder(records: list[SeqRecord], mode: Literal["meta", "single"] = "meta") -> pyrodigal.GeneFinder:
    if mode == "meta":
        return pyrodigal.GeneFinder(meta=True)
    total = sum(len(r.seq) for r in records)
    if total < MIN_SINGLE_BP:
        raise InputError(f"--genecall single needs >= {MIN_SINGLE_BP} bp to train ({total} bp given); "
                         "use --genecall meta")
    gf = pyrodigal.GeneFinder(meta=False)
    gf.train(*(bytes(r.seq) for r in records))
    return gf


def call_genes(records: list[SeqRecord], mode: Literal["meta", "single"] = "meta", threads: int = 1) -> pd.DataFrame:
    """CDS table of the predicted genes; locus tags follow Prodigal (``<record>_<n>``, 1-based per record)."""
    gf = gene_finder(records, mode)

    def one(rec: SeqRecord) -> list[dict]:
        genes = gf.find_genes(bytes(rec.seq).upper())
        return [{"record_id": rec.id, "locus_tag": f"{rec.id}_{i + 1}", "start": g.begin - 1, "end": g.end,
                 "strand": g.strand, "protein": g.translate(include_stop=False)} for i, g in enumerate(genes)]

    with ThreadPoolExecutor(max(1, threads)) as ex:
        rows = [r for rs in ex.map(one, records) for r in rs]
    log.info("pyrodigal (%s): %d genes on %d records", mode, len(rows), len(records))
    return finish_cds_table(rows)
