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

"""Tabular outputs: per-ORF scores and one row per region."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_orf_scores(orfs: pd.DataFrame, path: Path) -> None:
    orfs[["record_id", "cds_idx", "locus_tag", "score"]].to_csv(path, sep="\t", index=False)


def write_regions(regions: pd.DataFrame, path: Path) -> None:
    regions.to_csv(path, sep="\t", index=False, float_format="%.6g")
