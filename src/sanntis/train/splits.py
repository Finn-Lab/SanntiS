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

"""Monte Carlo CV splits: per kingdom track, a random 80/20 split over positives + chunks with
seed = split index; background pieces are split by genome, ``crc32(genome) % n_splits``."""

from __future__ import annotations

import zlib

import numpy as np
import pandas as pd


def mc_split(entities: pd.DataFrame, split: int, *, n_splits: int = 5, val_fraction: float = 0.2) -> pd.Series:
    """→ role ('train' | 'val') per entity row. ``entities``: entity_id, kind (positive|chunk|background),
    kingdom, genome (background)."""
    role = pd.Series("train", index=entities.index, dtype=object)
    for _, grp in entities[entities["kind"] != "background"].groupby("kingdom", sort=True):
        rng = np.random.default_rng(split)
        idx = rng.permutation(len(grp))
        n_val = int(round(val_fraction * len(grp)))
        role.loc[grp.index[idx[:n_val]]] = "val"
    bg = entities["kind"] == "background"
    fold = entities.loc[bg, "genome"].map(lambda g: zlib.crc32(str(g).encode()) % n_splits)
    role.loc[fold.index[fold == split % n_splits]] = "val"
    return role
