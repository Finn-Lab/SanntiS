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

"""Native-BGC removal: antiSMASH regions → BGC-free CDS runs, native mask, test backbones."""

from __future__ import annotations

import zlib
from collections.abc import Iterable

import numpy as np
import pandas as pd

TEST_BACKBONES = {"proka": 55, "euk": 15}
BACKBONE_MIN_CDS = 50
BACKBONE_SEED = 0


def overlaps(start: int, end: int, intervals: Iterable[tuple[int, int]]) -> bool:
    """0-based half-open overlap with any interval."""
    return any(start < b and end > a for a, b in intervals)


def clean_runs(cds: pd.DataFrame, regions: dict[str, list[tuple[int, int]]]) -> tuple[list[dict], list[dict]]:
    """Maximal runs of consecutive CDS outside every region, and the CDS inside regions (native mask).

    ``cds``: genome, contig, cds_idx, start, end (one genome). Run ids are ``<contig>_run<k>``.
    """
    runs, mask = [], []
    for (genome, contig), sub in cds.sort_values(["contig", "cds_idx"]).groupby(["genome", "contig"], sort=False):
        iv = regions.get(contig, [])
        k, cur = 0, []
        for idx, s, e in sub[["cds_idx", "start", "end"]].itertuples(index=False):
            if overlaps(int(s), int(e), iv):
                mask.append({"genome": genome, "contig": contig, "cds_idx": int(idx)})
                if cur:
                    runs.append({"genome": genome, "contig": contig, "run_id": f"{contig}_run{k}", "cds_idx": cur,
                                 "n_cds": len(cur)})
                    k, cur = k + 1, []
                continue
            cur.append(int(idx))
        if cur:
            runs.append({"genome": genome, "contig": contig, "run_id": f"{contig}_run{k}", "cds_idx": cur,
                         "n_cds": len(cur)})
    return runs, mask


def test_backbones(runs: pd.DataFrame, genomes: pd.DataFrame, n: dict[str, int] | None = None,
                   min_cds: int = BACKBONE_MIN_CDS, seed: int = BACKBONE_SEED) -> pd.DataFrame:
    """Isolate test backbones: per kingdom, the longest clean run (>= ``min_cds``) of each genome,
    then ``n[kingdom]`` genomes drawn with ``default_rng(seed + crc32(kingdom) % 1000)``."""
    n = n or TEST_BACKBONES
    kingdom = dict(zip(genomes["genome"], genomes["kingdom"], strict=True))
    out = []
    for k, want in n.items():
        rng = np.random.default_rng(seed + zlib.crc32(k.encode()) % 1000)
        df = runs[(runs["genome"].map(kingdom) == k) & (runs["n_cds"] >= min_cds)]
        df = (df.sort_values("n_cds", ascending=False, kind="stable").drop_duplicates("genome")
              .sort_values(["genome", "run_id"]).reset_index(drop=True))
        take = min(want, len(df))
        idx = rng.choice(len(df), size=take, replace=False) if take else []
        sel = df.iloc[list(idx)]
        out += [{"genome": g, "contig": c, "run_id": r, "kingdom": k}
                for g, c, r in sel[["genome", "contig", "run_id"]].itertuples(index=False)]
    return pd.DataFrame(out, columns=["genome", "contig", "run_id", "kingdom"])


def portions(runs: pd.DataFrame, quota: int, *, k: int = 3, seed: int = 0) -> list[dict]:
    """Portions of one genome's BGC-free CDS (lineage-matched pools).

    The genome's clean runs (``runs``: genome, contig, run_id, cds_idx; test-backbone runs already removed) are
    concatenated in contig / run order, so short runs of fragmented assemblies count as much as long ones. ``quota``
    CDS are then drawn as ``k`` windows, one per equal stratum of the concatenation (non-overlapping), with
    ``default_rng(seed + crc32(genome))``. A genome with no more than ``quota`` clean CDS gives one portion with all
    of them. Rows: genome, run_id (``<genome>:portion<j>``), contigs (per CDS), cds_idx, n_cds.
    """
    if runs.empty or quota <= 0:
        return []
    genome = str(runs["genome"].iloc[0])
    members: list[tuple[str, int]] = []
    ordered = runs.assign(_first=runs["cds_idx"].map(lambda x: int(x[0]))).sort_values(["contig", "_first"],
                                                                                        kind="stable")
    for contig, idx in ordered[["contig", "cds_idx"]].itertuples(index=False):
        members += [(str(contig), int(i)) for i in idx]
    n = len(members)
    if n <= quota:
        windows = [(0, n)]
    else:
        rng = np.random.default_rng(seed + zlib.crc32(genome.encode()) % 100_000)
        k = max(1, min(k, quota))
        sizes = [quota // k + (1 if j < quota % k else 0) for j in range(k)]
        bounds = np.linspace(0, n, k + 1).astype(int)
        windows = []
        for j, L in enumerate(sizes):
            lo, hi = int(bounds[j]), int(bounds[j + 1])
            start = lo + int(rng.integers(0, max(hi - lo - L, 0) + 1))
            windows.append((start, min(start + L, hi)))
    out = []
    for j, (a, b) in enumerate(windows):
        seg = members[a:b]
        out.append({"genome": genome, "run_id": f"{genome}:portion{j}", "contigs": [c for c, _ in seg],
                    "cds_idx": [i for _, i in seg], "n_cds": len(seg)})
    return out
