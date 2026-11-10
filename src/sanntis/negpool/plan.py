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

"""Lineage-matched negative pool plan: per-genome CDS quotas that follow the lineage mix of the positives.

The detector learns P(BGC | domain tokens). When a lineage holds many positives and no negatives, "looks like this
lineage" becomes BGC evidence. The plan therefore keeps the negative:positive CDS ratio about constant per lineage:

1. **Floor.** One genome per phylum that has candidates but no positives, with quota ``floor_cds``, so every phylum
   keeps some background.
2. **Matched.** The rest of the track budget is split over the positives' lineage groups in proportion to their
   positive CDS. A group is the positives' family when there are candidates in that family, else their order, else
   their phylum (positives without a family fall back the same way).
3. **Genomes.** Within a group, candidates are taken by ``priority`` (lower first, e.g. genomes that are already
   annotated), then in a seeded random order, one genome per species, skipping ``exclude_species``. Each genome's
   quota is ``min(cap, clean_cds)``; the last genome of a group takes the remainder (at least ``min_quota``).

Inputs (pandas frames):

- positives: track (proka | euk), phylum, order, family, n_cds (one row per positive);
- candidates: genome, track, phylum, order, family, species, clean_cds (estimated BGC-free CDS), priority, plus any
  passthrough columns (kept in the output).

Output: the selected candidates with ``quota_cds``, ``plan_group`` and ``plan_role`` (floor | matched).
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

RANKS = ("family", "order", "phylum")


@dataclass(frozen=True)
class PlanOptions:
    budget: dict[str, int]  # track → total negative CDS (floor + matched)
    cap: int = 1500  # CDS per genome
    floor_cds: int = 500
    min_quota: int = 100
    seed: int = 0
    exclude_species: frozenset[str] = frozenset()


def _key(rank: str, name) -> str | None:
    return f"{rank}:{name}" if isinstance(name, str) and name else None


def group_of(row, cand_keys: set[str]) -> str | None:
    """The finest rank of the positive (family → order → phylum) that has candidates."""
    for r in RANKS:
        k = _key(r, row[r])
        if k and k in cand_keys:
            return k
    return None


def _order(cands: pd.DataFrame, salt: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + zlib.crc32(salt.encode()) % 100_000)
    return cands.assign(_r=rng.random(len(cands))).sort_values(["priority", "_r"], kind="stable").drop(columns="_r")


def plan_track(pos: pd.DataFrame, cands: pd.DataFrame, track: str, o: PlanOptions) -> tuple[pd.DataFrame, dict]:
    cands = cands[~cands["species"].isin(o.exclude_species)].copy()
    keys = {r: cands[r].map(lambda v, r=r: _key(r, v)) for r in RANKS}
    cand_keys = set().union(*(set(k.dropna()) for k in keys.values()))
    pos = pos.assign(group=[group_of(r, cand_keys) for _, r in pos.iterrows()])
    stats = {"positives": len(pos), "positives_without_group": int(pos["group"].isna().sum())}
    used_species: set[str] = set()
    used_genomes: set[str] = set()
    picked: list[dict] = []

    def take(group: str, pool: pd.DataFrame, need: int, role: str) -> int:
        got = 0
        for _, c in _order(pool, f"{track}:{group}", o.seed).iterrows():
            if got >= need:
                break
            if c["species"] in used_species or c["genome"] in used_genomes:
                continue
            q = int(min(o.cap, c["clean_cds"], need - got))
            if q < o.min_quota and got > 0:
                break
            q = max(q, min(o.min_quota, int(c["clean_cds"])))
            used_species.add(c["species"])
            used_genomes.add(c["genome"])
            picked.append({**c.to_dict(), "quota_cds": q, "plan_group": group, "plan_role": role})
            got += q
        return got

    # 1. floor: phyla without positives
    pos_phyla = set(pos["phylum"].dropna())
    floor_total = 0
    for ph in sorted(set(cands["phylum"].dropna()) - pos_phyla):
        floor_total += take(f"phylum:{ph}", cands[cands["phylum"] == ph], o.floor_cds, "floor")
    # 2. matched: the rest in proportion to positive CDS per group
    matched_budget = max(o.budget[track] - floor_total, 0)
    by_group = pos.dropna(subset=["group"]).groupby("group")["n_cds"].sum()
    share = by_group / by_group.sum() if by_group.sum() else by_group
    short = {}
    for g, s in share.sort_values(ascending=False).items():
        need = int(round(s * matched_budget))
        rank, name = g.split(":", 1)
        got = take(g, cands[cands[rank] == name], need, "matched")
        if got < need:
            short[g] = need - got
    out = pd.DataFrame(picked)
    stats |= {"floor_cds": floor_total, "matched_budget": matched_budget,
              "matched_cds": int(out.loc[out["plan_role"] == "matched", "quota_cds"].sum()) if len(out) else 0,
              "groups": int(len(by_group)), "groups_short": len(short), "cds_short": int(sum(short.values())),
              "genomes": len(out), "ratio_matched": round(matched_budget / max(int(pos["n_cds"].sum()), 1), 3)}
    return out, stats


def plan_pool(positives: pd.DataFrame, candidates: pd.DataFrame, o: PlanOptions) -> tuple[pd.DataFrame, dict]:
    parts, stats = [], {}
    for track in sorted(o.budget):
        p, s = plan_track(positives[positives["track"] == track], candidates[candidates["track"] == track], track, o)
        parts.append(p)
        stats[track] = s
    return pd.concat(parts, ignore_index=True), stats
