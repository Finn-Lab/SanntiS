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

"""Entity table: deduplicated positives, length-matched negative chunks and background pieces.

Positives come from ``collect`` + ``annotate`` (one row per locus with its per-CDS token lists). Negatives come
from a negative pool version: ``cds_tokens.parquet`` (every CDS of every genome), ``clean_runs.parquet``
(BGC-free CDS runs) and ``test_backbones.parquet`` (runs reserved for the isolate test sets).

Entity rows: entity_id, kind (positive | chunk | background), label (1 | 0), class (primary MIBiG class, or
``non-BGC``; None for a GBK-only external BGC), classes (tuple, all MIBiG classes), kingdom, genome, n_cds,
cds_tokens (list of per-CDS token lists, genomic order).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from sanntis.model.postfilter import NON_BGC
from sanntis.train.dedup import dedup

log = logging.getLogger(__name__)

KINGDOMS = ("proka", "euk")


def classes_of(raw: object) -> tuple[str, ...]:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return ()
    if isinstance(raw, (list, tuple, np.ndarray)):
        return tuple(sorted(str(c).strip() for c in raw if str(c).strip()))
    return tuple(sorted(c.strip() for c in str(raw).split(";") if c.strip()))


def primary_class(raw: object) -> str | None:
    """ASCII-first class (NRPS > PKS > other > ribosomal > saccharide > terpene); None when unlabelled."""
    c = classes_of(raw)
    return c[0] if c else None


# ── Positives ────────────────────────────────────────────────────────────────


def positive_entities(loci: pd.DataFrame, *, dice_threshold: float = 0.974,
                      max_locus_bp: int = 200_000) -> tuple[pd.DataFrame, dict]:
    """``loci``: entity_id, kingdom, class_raw, length (bp), cds_tokens. Exclusions (> max bp, no CDS) then dedup
    within the pool, longest representative kept."""
    stats = {"loci": len(loci)}
    keep = loci[loci["cds_tokens"].map(len) > 0]
    stats["excluded_no_cds"] = len(loci) - len(keep)
    n = len(keep)
    keep = keep[keep["length"] <= max_locus_bp]
    stats["excluded_too_long"] = n - len(keep)
    d = dedup(keep.reset_index(drop=True), threshold=dice_threshold)
    reps = d[d["representative"]].sort_values("entity_id").reset_index(drop=True)
    stats["representatives"] = len(reps)
    for k in KINGDOMS:
        stats[f"representatives_{k}"] = int((reps["kingdom"] == k).sum())
    out = pd.DataFrame({
        "entity_id": reps["entity_id"], "kind": "positive", "label": 1,
        "class": reps["class_raw"].map(primary_class), "classes": reps["class_raw"].map(classes_of),
        "kingdom": reps["kingdom"], "genome": None, "n_cds": reps["cds_tokens"].map(len),
        "cds_tokens": reps["cds_tokens"].map(lambda c: [list(g) for g in c]),
    })
    return out, stats


# ── Negatives ────────────────────────────────────────────────────────────────


def sample_chunks(runs: pd.DataFrame, *, target: int, mu: float, sigma: float, min_cds: int = 3,
                  max_cds: int = 200, seed: int = 0) -> pd.DataFrame:
    """``target`` chunks: length ~ N(mu, sigma) clipped to [min, max]; run chosen with p ∝ n_cds, start uniform."""
    rng = np.random.default_rng(seed)
    runs = runs[runs["n_cds"] >= min_cds].reset_index(drop=True)
    if runs.empty:
        raise RuntimeError("no clean runs long enough to sample chunks from")
    w = runs["n_cds"].to_numpy(dtype=float)
    w /= w.sum()
    rows: list[dict] = []
    attempts = 0
    while len(rows) < target and attempts < target * 20:
        attempts += 1
        L = int(np.clip(rng.normal(mu, sigma), min_cds, max_cds))
        r = runs.iloc[int(rng.choice(len(runs), p=w))]
        if r["n_cds"] < L:
            continue
        start = int(rng.integers(0, r["n_cds"] - L + 1))
        rows.append({"chunk_id": f"neg_{len(rows):06d}", "genome": r["genome"], "contig": r["contig"],
                     "run_id": r["run_id"], "cds_idx": list(r["cds_idx"][start : start + L]), "n_cds": L})
    if len(rows) < target:
        log.warning("sampled only %d of %d chunks", len(rows), target)
    return pd.DataFrame(rows, columns=["chunk_id", "genome", "contig", "run_id", "cds_idx", "n_cds"])


def length_distribution(n_cds: np.ndarray) -> tuple[float, float]:
    v = np.asarray(n_cds)[np.asarray(n_cds) > 0]
    if v.size == 0:
        raise RuntimeError("no positives with CDS: cannot fit the chunk length distribution")
    return float(v.mean()), float(v.std(ddof=1) if v.size > 1 else 1.0) or 1.0


def pool_frames(pool: Path) -> dict[str, pd.DataFrame]:
    """The tables of a negative pool version that the entities are built from."""
    pool = Path(pool)
    return {"genomes": pd.read_parquet(pool / "genomes.parquet"),
            "cds": pd.read_parquet(pool / "cds_tokens.parquet", columns=["genome", "contig", "cds_idx", "tokens"]),
            "runs": pd.read_parquet(pool / "clean_runs.parquet"),
            "backbones": pd.read_parquet(pool / "test_backbones.parquet")}


def negative_entities(pool: dict[str, pd.DataFrame], positives: pd.DataFrame, *, chunks_per_positive: float = 3.0,
                      clip: tuple[int, int] = (3, 200), background_min_cds: int = 5,
                      seed: int = 0) -> tuple[pd.DataFrame, dict]:
    """Chunks (per kingdom, matched to that kingdom's positive lengths) and background pieces."""
    kingdom_of = dict(zip(pool["genomes"]["genome"], pool["genomes"]["kingdom"], strict=True))
    # tokens None = CDS without a feature row (dev pool): skipped inside chunks, breaks background pieces;
    # [] = CDS without hits.
    tok = {(g, c, i): (None if t is None else list(t))
           for g, c, i, t in pool["cds"].itertuples(index=False, name=None)}
    runs = pool["runs"].assign(kingdom=pool["runs"]["genome"].map(kingdom_of))
    backbone_runs = set(zip(pool["backbones"]["genome"], pool["backbones"]["run_id"], strict=True))
    rows: list[dict] = []
    stats: dict = {}
    for k in KINGDOMS:
        kr = runs[runs["kingdom"] == k].reset_index(drop=True)
        pos = positives[positives["kingdom"] == k]
        if kr.empty or pos.empty:
            continue
        mu, sigma = length_distribution(pos["n_cds"].to_numpy())
        target = int(round(chunks_per_positive * int((pos["n_cds"] > 0).sum())))
        ch = sample_chunks(kr, target=target, mu=mu, sigma=sigma, min_cds=clip[0], max_cds=clip[1], seed=seed)
        stats[f"chunks_sampled_{k}"] = len(ch)
        stats[f"chunk_len_mu_sigma_{k}"] = (round(mu, 2), round(sigma, 2))
        suffix = "" if k == "proka" else f"_{k}"
        for c in ch.itertuples(index=False):
            cds = [t for t in (tok.get((c.genome, c.contig, i)) for i in c.cds_idx) if t is not None]
            if not cds:
                continue
            rows.append({"entity_id": f"{c.chunk_id}{suffix}", "kind": "chunk", "label": 0, "class": NON_BGC,
                         "classes": (), "kingdom": k, "genome": c.genome, "n_cds": len(cds), "cds_tokens": cds})
        # background: clean runs (not test backbones) minus chunk CDS, maximal pieces of >= min CDS
        in_chunk = {(c.genome, c.contig, i) for c in ch.itertuples(index=False) for i in c.cds_idx}
        n_bg = n_bg_cds = 0
        for r in kr.itertuples(index=False):
            if (r.genome, r.run_id) in backbone_runs:
                continue
            piece: list[list[str]] = []
            pieces: list[list[list[str]]] = []
            for i in [*list(r.cds_idx), None]:
                t = None if i is None or (r.genome, r.contig, i) in in_chunk else tok.get((r.genome, r.contig, i))
                if t is None:
                    if len(piece) >= background_min_cds:
                        pieces.append(piece)
                    piece = []
                else:
                    piece.append(t)
            for j, cds in enumerate(pieces):
                rows.append({"entity_id": f"bg:{k}:{r.genome}:{r.run_id}:{j}", "kind": "background", "label": 0,
                             "class": NON_BGC, "classes": (), "kingdom": k, "genome": r.genome,
                             "n_cds": len(cds), "cds_tokens": cds})
                n_bg += 1
                n_bg_cds += len(cds)
        stats[f"background_{k}"] = n_bg
        stats[f"background_cds_{k}"] = n_bg_cds
    return pd.DataFrame(rows), stats


def entity_table(positives: pd.DataFrame, negatives: pd.DataFrame) -> pd.DataFrame:
    """Positives first (sorted by id), then chunks, then background; the row index is the entity index."""
    order = {"positive": 0, "chunk": 1, "background": 2}
    t = pd.concat([positives, negatives], ignore_index=True)
    t = t.assign(_o=t["kind"].map(order)).sort_values(["_o", "kingdom"], kind="stable").drop(columns="_o")
    return t.reset_index(drop=True)
