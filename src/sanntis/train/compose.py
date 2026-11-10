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


def _units(order: list[int], entities: pd.DataFrame) -> list[list[int]]:
    """[entity row, first CDS, end CDS] per composed entity."""
    return [[i, 0, len(entities.at[i, "cds_tokens"])] for i in order]


def _cluster(units: list[list[int]], entities: pd.DataFrame, clusters, rng: np.random.Generator) -> list[list[int]]:
    """Group a share of the positive units into clusters of k positives separated by short real background gaps.

    Gaps of g ~ U{gap} CDS are cut from the front of background units (which keep their remaining CDS); a cluster
    replaces its first member's slot. Stops clustering when the background runs out."""
    kind = entities["kind"]
    pos = [j for j, u in enumerate(units) if kind.at[u[0]] == "positive"]
    donors = [j for j, u in enumerate(units) if kind.at[u[0]] == "background"]
    chosen = sorted(rng.choice(pos, size=int(round(clusters.share * len(pos))), replace=False).tolist()) if pos else []
    groups, i = [], 0
    while i < len(chosen):
        k = int(rng.integers(clusters.k[0], clusters.k[1] + 1))
        if len(chosen[i:i + k]) >= 2:
            groups.append(chosen[i:i + k])
        i += k
    d = 0
    built: dict[int, list[list[int]]] = {}
    removed: set[int] = set()
    for grp in groups:
        seq = [units[grp[0]]]
        ok = True
        for m in grp[1:]:
            g = int(rng.integers(clusters.gap[0], clusters.gap[1] + 1))
            while d < len(donors) and units[donors[d]][2] - units[donors[d]][1] < g:
                d += 1
            if d >= len(donors):
                ok = False
                break
            du = units[donors[d]]
            seq.append([du[0], du[1], du[1] + g])
            du[1] += g
            seq.append(units[m])
        if not ok:
            break
        built[grp[0]] = seq
        removed.update(grp[1:])
    out = []
    for j, u in enumerate(units):
        if j in removed:
            continue
        if j in built:
            out.extend(built[j])
        elif u[2] > u[1]:
            out.append(u)
    return out


def compose(entities: pd.DataFrame, *, duplicate_positives: int = 1, seed: int = 0,
            clusters=None) -> list[OrfSequence]:
    """Per kingdom (sorted): positives repeated ``duplicate_positives`` times, every other entity once, permuted
    with ``default_rng(seed)``. Labels: 1 for every CDS of a positive. Entity rows are indexed by the frame index.

    ``clusters`` (recipe ``compose.clusters``; training compositions only): a share of the positives is placed in
    clusters of k positives separated by gaps of real background, so the detector sees BGC → background → BGC
    transitions at realistic distances (neighbouring BGCs in BGC-rich genomes)."""
    rng = np.random.default_rng(seed)
    out: list[OrfSequence] = []
    for kingdom, grp in entities.groupby("kingdom", sort=True):
        order: list[int] = []
        for i, lab in zip(grp.index, grp["label"], strict=True):
            order.extend([i] * (duplicate_positives if lab == 1 else 1))
        order = list(rng.permutation(order))
        units = _units(order, entities)
        if clusters is not None and "kind" in entities:
            units = _cluster(units, entities, clusters,
                             np.random.default_rng(seed + 7919 + sum(map(ord, str(kingdom)))))
        cds: list[list[str]] = []
        lab_arr: list[int] = []
        ent_arr: list[int] = []
        for i, a, b in units:
            e = entities.loc[i]
            n = b - a
            cds.extend(list(c) for c in e["cds_tokens"][a:b])
            lab_arr.extend([int(e["label"])] * n)
            ent_arr.extend([int(i)] * n)
        out.append(OrfSequence(kingdom=str(kingdom), cds_tokens=cds, orf_label=np.asarray(lab_arr, dtype=np.int8),
                               orf_entity=np.asarray(ent_arr, dtype=np.int64)))
    return out
