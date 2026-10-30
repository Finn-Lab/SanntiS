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

"""Feature-level sequence composition: one shuffled concatenation of whole entities per kingdom."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class OrfSequence:
    """One composed contig in ORF space. ``orf_entity`` = row index into the entity frame, per ORF."""

    kingdom: str
    cds_tokens: list[list[str]]
    orf_label: np.ndarray
    orf_entity: np.ndarray


def compose(entities: pd.DataFrame, *, duplicate_positives: int = 1, seed: int = 0) -> list[OrfSequence]:
    """Per kingdom (sorted): positives repeated ``duplicate_positives`` times, every other entity once, permuted
    with ``default_rng(seed)``. Labels: 1 for every CDS of a positive. Entity rows are indexed by the frame index."""
    rng = np.random.default_rng(seed)
    out: list[OrfSequence] = []
    for kingdom, grp in entities.groupby("kingdom", sort=True):
        order: list[int] = []
        for i, lab in zip(grp.index, grp["label"], strict=True):
            order.extend([i] * (duplicate_positives if lab == 1 else 1))
        order = list(rng.permutation(order))
        cds: list[list[str]] = []
        lab_arr: list[int] = []
        ent_arr: list[int] = []
        for i in order:
            e = entities.loc[i]
            n = len(e["cds_tokens"])
            cds.extend(list(c) for c in e["cds_tokens"])
            lab_arr.extend([int(e["label"])] * n)
            ent_arr.extend([int(i)] * n)
        out.append(OrfSequence(kingdom=str(kingdom), cds_tokens=cds, orf_label=np.asarray(lab_arr, dtype=np.int8),
                               orf_entity=np.asarray(ent_arr, dtype=np.int64)))
    return out
