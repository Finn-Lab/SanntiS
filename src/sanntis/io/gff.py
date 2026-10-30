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

"""GFF3 writer: one CLUSTER feature per kept region."""

from __future__ import annotations

import math
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from sanntis._version import __version__

SOURCE = "SanntiS"


def escape(v) -> str:
    return (str(v).replace("%", "%25").replace(";", "%3B").replace("=", "%3D").replace("&", "%26")
            .replace(",", "%2C"))


def _isnull(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def region_attributes(r) -> list[tuple[str, str]]:
    """Unescaped ``(key, value)`` attributes of one region row, shared by the GFF3 and antiSMASH writers."""
    attrs = [("ID", f"{r.record_id}_sanntis_{r.region}")]
    if not _isnull(r.bgc_class):
        attrs += [("nearest_MiBIG_class", r.bgc_class), ("bgc_class", r.bgc_class)]
    attrs += [("p_non_bgc", f"{r.p_non_bgc:.4f}"), ("n_orfs", str(r.n_orfs))]
    if not _isnull(r.npc_pathway):
        attrs += [("npc_pathway", r.npc_pathway), ("npc_pathway_score", f"{r.npc_pathway_score:.4f}"),
                  ("npc_pathway_confident", str(bool(r.npc_pathway_confident)).lower()),
                  ("npc_superclass", r.npc_superclass), ("npc_superclass_top", r.npc_superclass_top),
                  ("npc_superclass_score", f"{r.npc_superclass_score:.4f}")]
        if not _isnull(getattr(r, "npc_pathway_key_domains_support", None)):
            u = r.npc_pathway_key_domains_unsupported
            attrs += [("npc_pathway_key_domains_support", r.npc_pathway_key_domains_support),
                      ("npc_pathway_key_domains_unsupported", "na" if _isnull(u) else str(bool(u)).lower())]
    return attrs


def feature_lines(regions: pd.DataFrame, source: str = SOURCE) -> Iterable[str]:
    for r in regions.itertuples(index=False):
        attrs = ";".join(f"{k}={escape(v)}" for k, v in region_attributes(r))
        yield f"{r.record_id}\t{source}\tCLUSTER\t{r.start}\t{r.end}\t{r.score:.4f}\t.\t.\t{attrs}"


def write_gff(regions: pd.DataFrame, path: Path, header: dict | None = None) -> None:
    """``header``: extra ``##key value`` lines (model, IPS release, QC flags)."""
    with open(path, "w") as fh:
        fh.write("##gff-version 3\n")
        fh.write(f"##sanntis-version {__version__}\n")
        for k, v in (header or {}).items():
            fh.write(f"##{k} {v}\n")
        for line in feature_lines(regions):
            fh.write(line + "\n")
