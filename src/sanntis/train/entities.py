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
    """``target`` chunks: length ~ N(mu, sigma) clipped to [min, max]; unit chosen with p ∝ n_cds, start uniform.

    ``runs``: genome, run_id, n_cds and ``members`` (list of (contig, cds_idx)); a clean run or a portion."""
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
        mem = list(r["members"][start : start + L])
        rows.append({"chunk_id": f"neg_{len(rows):06d}", "genome": r["genome"], "contig": mem[0][0],
                     "run_id": r["run_id"], "cds_idx": [i for _, i in mem], "n_cds": L, "members": mem})
    if len(rows) < target:
        log.warning("sampled only %d of %d chunks", len(rows), target)
    return pd.DataFrame(rows, columns=["chunk_id", "genome", "contig", "run_id", "cds_idx", "n_cds", "members"])


def negative_units(pool: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """The units negatives are cut from: the pool's portions when it has them (lineage-matched pools), else its
    clean runs. Columns: genome, run_id, n_cds, members (list of (contig, cds_idx)), backbone (a test-backbone run:
    chunks may come from it, as in pool v1, but background pieces may not)."""
    p = pool.get("portions")
    if p is not None and len(p):
        return pd.DataFrame({"genome": p["genome"], "run_id": p["run_id"], "n_cds": p["n_cds"],
                             "members": [list(zip(map(str, c), map(int, i), strict=True))
                                         for c, i in zip(p["contigs"], p["cds_idx"], strict=True)],
                             "backbone": False})
    r = pool["runs"]
    bb = set(zip(pool["backbones"]["genome"], pool["backbones"]["run_id"], strict=True))
    return pd.DataFrame({"genome": r["genome"].to_numpy(), "run_id": r["run_id"].to_numpy(),
                         "n_cds": r["n_cds"].to_numpy(),
                         "members": [[(str(c), int(i)) for i in idx]
                                     for c, idx in zip(r["contig"], r["cds_idx"], strict=True)],
                         "backbone": [(g, rid) in bb for g, rid in zip(r["genome"], r["run_id"], strict=True)]})


def length_distribution(n_cds: np.ndarray) -> tuple[float, float]:
    v = np.asarray(n_cds)[np.asarray(n_cds) > 0]
    if v.size == 0:
        raise RuntimeError("no positives with CDS: cannot fit the chunk length distribution")
    return float(v.mean()), float(v.std(ddof=1) if v.size > 1 else 1.0) or 1.0


def pool_frames(pool: Path) -> dict[str, pd.DataFrame]:
    """The tables of a negative pool version that the entities are built from."""
    pool = Path(pool)
    portions = pool / "portions.parquet"
    return {"portions": pd.read_parquet(portions) if portions.is_file() else None,
            "genomes": pd.read_parquet(pool / "genomes.parquet"),
            "cds": pd.read_parquet(pool / "cds_tokens.parquet", columns=["genome", "contig", "cds_idx", "tokens"]),
            "runs": pd.read_parquet(pool / "clean_runs.parquet"),
            "backbones": pd.read_parquet(pool / "test_backbones.parquet")}


def euk_genome_ok(lineage: object, euk_lineages: tuple[str, ...] | None) -> bool:
    """Pool lineage strings carry ``kingdom__<name>`` (RefSeq groups: fungi, protozoa, plants)."""
    if not euk_lineages:
        return True
    lin = str(lineage or "").lower()
    return any(f"kingdom__{x.lower()}" in lin for x in euk_lineages)


def negative_entities(pool: dict[str, pd.DataFrame], positives: pd.DataFrame, *, chunks_per_positive: float = 3.0,
                      clip: tuple[int, int] = (3, 200), background_min_cds: int = 5,
                      euk_lineages: tuple[str, ...] | None = None, seed: int = 0) -> tuple[pd.DataFrame, dict]:
    """Chunks (per kingdom, matched to that kingdom's positive lengths) and background pieces, cut from the pool's
    negative units (:func:`negative_units`). ``euk_lineages`` keeps only euk genomes of those RefSeq groups."""
    g = pool["genomes"]
    kingdom_of = dict(zip(g["genome"], g["kingdom"], strict=True))
    lineage = g["lineage"] if "lineage" in g else pd.Series([""] * len(g), index=g.index)
    euk_ok = {x for x, k, lin in zip(g["genome"], g["kingdom"], lineage, strict=True)
              if k != "euk" or euk_genome_ok(lin, euk_lineages)}
    # tokens None = CDS without a feature row (dev pool) or not annotated (portion pools): skipped inside chunks,
    # breaks background pieces; [] = CDS without hits.
    tok = {(g_, c, i): (None if t is None else list(t))
           for g_, c, i, t in pool["cds"].itertuples(index=False, name=None)}
    units = negative_units(pool)
    units = units[units["genome"].isin(euk_ok)].assign(kingdom=lambda d: d["genome"].map(kingdom_of))
    rows: list[dict] = []
    stats: dict = {"negative_units": "portions" if pool.get("portions") is not None and len(pool["portions"])
                   else "clean_runs"}
    for k in KINGDOMS:
        kr = units[units["kingdom"] == k].reset_index(drop=True)
        pos = positives[positives["kingdom"] == k]
        if kr.empty or pos.empty:
            continue
        mu, sigma = length_distribution(pos["n_cds"].to_numpy())
        target = int(round(chunks_per_positive * int((pos["n_cds"] > 0).sum())))
        ch = sample_chunks(kr, target=target, mu=mu, sigma=sigma, min_cds=clip[0], max_cds=clip[1], seed=seed)
        stats[f"chunks_sampled_{k}"] = len(ch)
        stats[f"chunk_len_mu_sigma_{k}"] = (round(mu, 2), round(sigma, 2))
        stats[f"genomes_{k}"] = int(kr["genome"].nunique())
        suffix = "" if k == "proka" else f"_{k}"
        for c in ch.itertuples(index=False):
            cds = [t for t in (tok.get((c.genome, ct, i)) for ct, i in c.members) if t is not None]
            if not cds:
                continue
            rows.append({"entity_id": f"{c.chunk_id}{suffix}", "kind": "chunk", "label": 0, "class": NON_BGC,
                         "classes": (), "kingdom": k, "genome": c.genome, "n_cds": len(cds), "cds_tokens": cds})
        # background: the units minus chunk CDS, maximal pieces of >= min CDS
        in_chunk = {(c.genome, ct, i) for c in ch.itertuples(index=False) for ct, i in c.members}
        n_bg = n_bg_cds = 0
        for r in kr.itertuples(index=False):
            if r.backbone:
                continue
            piece: list[list[str]] = []
            pieces: list[list[list[str]]] = []
            for m in [*list(r.members), None]:
                key = None if m is None else (r.genome, m[0], m[1])
                t = None if key is None or key in in_chunk else tok.get(key)
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
