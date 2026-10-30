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

"""antiSMASH sideloader JSON writer: one subregion per kept region (``antismash --sideload``)."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from sanntis._version import __version__
from sanntis.io.gff import SOURCE, _isnull, region_attributes

DESCRIPTION = "BGC detection and classification from InterProScan domain tokens"
LIST_ATTRIBUTES = {"npc_pathway", "npc_pathway_key_domains_support"}  # '|'-joined in the GFF


def subregion(r) -> dict:
    """Region row (1-based, end inclusive) → sideloader subregion (0-based, end exclusive)."""
    details = {"score": f"{r.score:.4f}"}
    for k, v in region_attributes(r):
        details[k] = str(v).split("|") if k in LIST_ATTRIBUTES else str(v)
    return {"start": int(r.start) - 1, "end": int(r.end),
            "label": SOURCE if _isnull(r.bgc_class) else str(r.bgc_class), "details": details}


def sideload(regions: pd.DataFrame, record_ids: Iterable[str], configuration: dict | None = None) -> dict:
    """Every record of ``record_ids`` is listed (antiSMASH ignores records without areas; the schema needs >= 1)."""
    by_record = {str(rid): [] for rid in record_ids}
    for r in regions.itertuples(index=False):
        by_record.setdefault(str(r.record_id), []).append(subregion(r))
    tool = {"name": SOURCE, "version": __version__, "description": DESCRIPTION,
            "configuration": {k: str(v) for k, v in (configuration or {}).items()}}
    return {"tool": tool, "records": [{"name": rid, "subregions": subs} for rid, subs in by_record.items()]}


def write_sideload(regions: pd.DataFrame, record_ids: Iterable[str], path: Path,
                   configuration: dict | None = None) -> None:
    """``configuration``: the same key / value pairs as the GFF ``##`` header lines."""
    Path(path).write_text(json.dumps(sideload(regions, record_ids, configuration), indent=2) + "\n")
