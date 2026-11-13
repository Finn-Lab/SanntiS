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

"""Genomes that never enter a negative pool (``--exclude`` of ``plan-negatives`` and ``build-negatives``).

The table (TSV) has the columns ``accession`` and ``species``; either may be empty in a row, other columns (e.g. a
note) are ignored. A genome is excluded when its assembly matches an ``accession`` (GCA_ / GCF_ prefix and version
ignored, so the GenBank and RefSeq copies of an assembly match) or its species matches a ``species`` (case and
brackets ignored). Rows with an accession only exclude that assembly; rows with a species exclude the whole species.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ACC_RE = re.compile(r"^GC[AF]_(\d+)(?:\.\d+)?")


def accession_core(acc: object) -> str:
    """``GCF_000123456.2`` → ``000123456`` (shared by the GenBank / RefSeq pair); other ids unchanged."""
    s = str(acc or "").strip()
    m = ACC_RE.match(s)
    return m.group(1) if m else s


def norm_species(name: object) -> str:
    return " ".join(re.sub(r"[\[\]']", "", str(name or "")).lower().split())


def species_of_lineage(lineage: object) -> str:
    """Species from a lineage string (``...;s__Name`` or ``...;species__Name``)."""
    for seg in str(lineage or "").split(";"):
        seg = seg.strip()
        for p in ("species__", "s__"):
            if seg.startswith(p) and len(seg) > len(p):
                return seg[len(p):]
    return ""


@dataclass(frozen=True)
class Exclusion:
    accessions: frozenset[str]
    species: frozenset[str]

    def __len__(self) -> int:
        return len(self.accessions) + len(self.species)


def read_exclude(path: str | Path) -> Exclusion:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    missing = {"accession", "species"} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: exclusion table needs the columns accession and species (missing {sorted(missing)})")
    acc = frozenset(accession_core(a) for a in df["accession"] if a.strip())
    sp = frozenset(norm_species(s) for s in df["species"] if s.strip())
    return Exclusion(acc, sp)


def excluded(genomes: pd.DataFrame, ex: Exclusion) -> pd.Series:
    """Boolean mask over a genome table (``accession``, else ``genome``; ``species``, else from ``lineage``)."""
    acc = genomes["accession"] if "accession" in genomes else genomes["genome"]
    acc = [a or g for a, g in zip(acc.fillna(""), genomes["genome"], strict=True)]
    if "species" in genomes:
        sp = [s or species_of_lineage(lin) for s, lin in zip(genomes["species"].fillna(""),
                                                             genomes.get("lineage", pd.Series([""] * len(genomes),
                                                                                              index=genomes.index)),
                                                             strict=True)]
    else:
        sp = [species_of_lineage(lin) for lin in genomes.get("lineage", pd.Series([""] * len(genomes)))]
    hit = [accession_core(a) in ex.accessions or (bool(s) and norm_species(s) in ex.species) for a, s in zip(acc, sp,
                                                                                                             strict=True)]
    return pd.Series(hit, index=genomes.index)
