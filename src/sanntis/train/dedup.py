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

"""Positive deduplication within a training pool: composite-Dice single linkage, longest kept."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from sanntis.features.tokens import bag_matrix


def protein_tokens(cds_tokens: Sequence[Sequence[str]]) -> list[str]:
    """One token per protein (CDS order): its first InterPro entry, else its first signature; contiguous
    duplicates collapsed. ``cds_tokens`` are the per-CDS token lists (ordered by hit start)."""
    out: list[str] = []
    for toks in cds_tokens:
        if not len(toks):
            continue
        ipr = [t for t in toks if str(t).startswith("IPR")]
        t = ipr[0] if ipr else toks[0]
        if not out or out[-1] != t:
            out.append(t)
    return out


def _dice(M: sp.csr_matrix) -> sp.csr_matrix:
    M = M.astype(np.float64)
    inter = (M @ M.T).tocoo()
    size = np.asarray(M.sum(1)).ravel()
    den = size[inter.row] + size[inter.col]
    val = np.divide(2 * inter.data, den, out=np.zeros_like(inter.data), where=den > 0)
    return sp.csr_matrix((val, (inter.row, inter.col)), shape=inter.shape)


def composite_similarity(token_seqs: Sequence[Sequence[str]], weights: tuple[float, float] = (0.5, 0.5)):
    sets = [set(t) for t in token_seqs]
    pairs = [{(t[i], t[i + 1]) for i in range(len(t) - 1)} for t in token_seqs]
    dv = {t: i for i, t in enumerate(sorted(set().union(*sets)))} if sets else {}
    pv = {p: i for i, p in enumerate(sorted(set().union(*pairs)))} if pairs else {}
    sim = (weights[0] * _dice(bag_matrix(sets, dv, binary=True))
           + weights[1] * _dice(bag_matrix(pairs, pv, binary=True)))
    sim = sim.tocsr()
    sim.setdiag(0)
    sim.eliminate_zeros()
    return sim


def dedup(loci: pd.DataFrame, threshold: float = 0.974) -> pd.DataFrame:
    """``loci``: entity_id, length (bp), cds_tokens. Returns ``loci`` + ``cluster`` + ``representative`` (bool):
    the longest locus of each single-linkage cluster (ties: entity_id order)."""
    seqs = [protein_tokens(c) for c in loci["cds_tokens"]]
    sim = composite_similarity(seqs)
    out = loci.copy().reset_index(drop=True)
    _, out["cluster"] = connected_components(sim >= threshold - 1e-12, directed=False)
    order = out.sort_values(["cluster", "length", "entity_id"], ascending=[True, False, True])
    reps = set(order.groupby("cluster", sort=False).head(1).index)
    out["representative"] = out.index.isin(reps)
    return out
