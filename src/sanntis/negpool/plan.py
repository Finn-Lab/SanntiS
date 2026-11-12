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

``sanntis plan-negatives`` (:func:`run_cli`) builds both frames from public, pinned inputs and writes the genome
table ``sanntis build-negatives --genomes`` reads:

- positives: the production positives of a ``sanntis train --until annotate`` run (deduplicated, the recipe's
  eukaryote lineages), lineage from their MIBiG NCBI taxid;
- candidates: GTDB species representatives of one release (CheckM2 completeness / contamination thresholds,
  ``clean_cds`` = 0.9 x protein count) and candidate tables (TSV: genome, kingdom, accession, taxid, source,
  priority, clean_cds; optional url, taxon, species, and phylum where the taxdump has none), lineage from their
  NCBI taxid;
- every lineage from one NCBI taxdump; ``--exclude`` removes genomes before planning; a breadth table adds genomes
  that are not planned with a fixed quota (``plan_role`` breadth).
"""

from __future__ import annotations

import json
import logging
import re
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

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


# ── inputs (sanntis plan-negatives) ─────────────────────────────────────────

CANDIDATE_COLUMNS = ("genome", "kingdom", "accession", "taxid", "source", "priority", "clean_cds")
LINEAGE_RANKS = ("domain", "kingdom", "phylum", "class", "order", "family", "genus", "species")
GTDB_CLEAN_FRAC = 0.9  # BGC-free share of a GTDB representative's proteins (quota estimate only)


def binomial(name: object) -> str:
    """Organism name → species binomial (``Streptomyces coelicolor A3(2)`` → ``Streptomyces coelicolor``)."""
    w = re.sub(r"[\[\]]", "", str(name or "")).split()
    if len(w) >= 2 and w[1] not in ("sp.", "sp"):
        return f"{w[0]} {w[1]}"
    return " ".join(w[:3])


def lineage_string(lin: dict[str, str]) -> str:
    return ";".join(f"{r}__{lin[r]}" for r in LINEAGE_RANKS if lin.get(r))


def with_lineage(df: pd.DataFrame, tax, taxid_col: str = "taxid") -> pd.DataFrame:
    lins = [tax.lineage(t) for t in df[taxid_col]]
    out = df.copy()
    for r in ("phylum", "order", "family"):
        out[r] = [lin.get(r) for lin in lins]
    out["lineage"] = [lineage_string(lin) for lin in lins]
    out["_tax_species"] = [lin.get("species", "") for lin in lins]
    return out


def positives_from_run(run: Path, tax) -> pd.DataFrame:
    """Production positives of a train run (``--until annotate``): track, phylum, order, family, n_cds."""
    from sanntis.config import load_recipe
    from sanntis.train.steps import production_positives

    run = Path(run)
    f = run / "annotate" / "positives.parquet"
    if not f.is_file():
        raise FileNotFoundError(f"{f}: run `sanntis train --run {run} ... --until annotate` first")
    pe, _ = production_positives(pd.read_parquet(f), load_recipe())
    km = pd.read_csv(run / "collect" / "kingdom_map.tsv", sep="\t", dtype=str)
    taxid = dict(zip(km["accession"], km["ncbi_tax_id"], strict=True))
    pos = pd.DataFrame({"entity_id": pe["entity_id"], "track": pe["kingdom"], "n_cds": pe["n_cds"].astype(int),
                        "taxid": pe["entity_id"].map(taxid)})
    return with_lineage(pos, tax).drop(columns="_tax_species")


def gtdb_candidates(tables: list[Path], tax, *, priority: int = 1, min_completeness: float = 90.0,
                    max_contamination: float = 5.0, release: str = "") -> pd.DataFrame:
    """GTDB species representatives with CheckM2 completeness >= / contamination <= the thresholds."""
    cols = ["accession", "gtdb_representative", "gtdb_taxonomy", "ncbi_taxid", "ncbi_organism_name", "protein_count",
            "checkm2_completeness", "checkm2_contamination"]
    md = pd.concat([pd.read_csv(t, sep="\t", usecols=cols, dtype=str, low_memory=False) for t in tables],
                   ignore_index=True)
    md = md[md["gtdb_representative"].str.lower().eq("t")]
    num = {c: pd.to_numeric(md[c], errors="coerce") for c in ("checkm2_completeness", "checkm2_contamination",
                                                              "protein_count")}
    md = md[(num["checkm2_completeness"] >= min_completeness) & (num["checkm2_contamination"] <= max_contamination)
            & num["protein_count"].notna()]
    acc = md["accession"].str.replace(r"^(RS_|GB_)", "", regex=True)
    df = pd.DataFrame({"genome": acc.values, "kingdom": "proka", "accession": acc.values,
                       "taxid": md["ncbi_taxid"].values, "source": f"gtdb_r{release}" if release else "gtdb",
                       "priority": priority, "clean_cds": (num["protein_count"][md.index] * GTDB_CLEAN_FRAC).round()
                       .values, "species": md["ncbi_organism_name"].map(binomial).values, "url": "",
                       "taxon": "bacteria"})
    df = with_lineage(df, tax)
    gtdb_phylum = md["gtdb_taxonomy"].str.extract(r"p__([^;]+)")[0].values
    df["phylum"] = df["phylum"].where(df["phylum"].notna(), pd.Series(gtdb_phylum, index=df.index))
    return df


def read_candidates(path: Path, tax) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    missing = set(CANDIDATE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path}: candidate table needs {list(CANDIDATE_COLUMNS)} (missing {sorted(missing)})")
    if not df["kingdom"].isin(["proka", "euk"]).all():
        raise ValueError(f"{path}: kingdom must be proka or euk")
    for c in ("url", "taxon", "species"):
        if c not in df:
            df[c] = ""
    df["priority"] = pd.to_numeric(df["priority"]).astype(int)
    df["clean_cds"] = pd.to_numeric(df["clean_cds"])
    df.loc[df["taxon"] == "", "taxon"] = df["kingdom"].map({"proka": "bacteria", "euk": "fungi"})
    out = with_lineage(df.drop(columns=[c for c in ("phylum", "order", "family", "lineage") if c in df]), tax)
    if "phylum" in df:  # the table's phylum where the taxdump has none (e.g. GTDB phyla of candidate taxa)
        given = df["phylum"].replace("", None)
        out["phylum"] = out["phylum"].where(out["phylum"].notna(), given)
    return out


def _finish_species(df: pd.DataFrame) -> pd.DataFrame:
    sp = [s or t or g for s, t, g in zip(df["species"].fillna(""), df["_tax_species"].fillna(""), df["genome"],
                                         strict=True)]
    return df.assign(species=sp).drop(columns="_tax_species")


def dedup_candidates(cand: pd.DataFrame) -> pd.DataFrame:
    """One row per assembly (GenBank / RefSeq copies count once): the lowest ``priority``, then the first given."""
    from sanntis.negpool.exclude import accession_core

    c = cand.assign(_acc=[accession_core(a) for a in cand["accession"]]).sort_values("priority", kind="stable")
    return c.drop_duplicates("_acc").drop(columns="_acc").sort_index()


def genome_table(plan: pd.DataFrame) -> pd.DataFrame:
    """Plan rows → the ``build-negatives --genomes`` table (plus the plan columns, which the build ignores)."""
    return pd.DataFrame({
        "genome": plan["genome"], "kingdom": plan["track"], "lineage": plan["lineage"], "source": plan["source"],
        "accession": plan["accession"], "group": plan["plan_group"], "taxon": plan["taxon"], "url": plan["url"],
        "path": "", "quota_cds": plan["quota_cds"].astype(int), "species": plan["species"], "taxid": plan["taxid"],
        "phylum": plan["phylum"], "order": plan["order"], "family": plan["family"],
        "clean_cds": plan["clean_cds"].astype(int), "priority": plan["priority"], "plan_role": plan["plan_role"]})


def add_breadth(plan: pd.DataFrame, breadth: pd.DataFrame, cds: int) -> pd.DataFrame:
    """Every breadth genome not in the plan (same assembly) with ``min(cds, clean_cds)`` CDS."""
    from sanntis.negpool.exclude import accession_core

    have = {accession_core(a) for a in plan["accession"]}
    b = breadth[[accession_core(a) not in have for a in breadth["accession"]]].copy()
    b["quota_cds"] = np.minimum(cds, b["clean_cds"]).astype(int)
    b = b[b["quota_cds"] > 0]
    b = b.rename(columns={"kingdom": "track"}).assign(plan_group="breadth", plan_role="breadth")
    return pd.concat([plan, b[plan.columns.intersection(b.columns)]], ignore_index=True)


def parse_budget(items) -> dict[str, int]:
    out = {}
    for it in items:
        k, _, v = str(it).partition("=")
        if k not in ("proka", "euk") or not v.strip().isdigit():
            raise ValueError(f"--budget {it!r}: expected proka=<CDS> or euk=<CDS>")
        out[k] = int(v)
    if not out:
        raise ValueError("--budget: give at least one track (proka=<CDS>, euk=<CDS>)")
    return out


def run_cli(**kw) -> None:
    from sanntis.negpool import sample as S
    from sanntis.negpool.exclude import excluded, read_exclude
    from sanntis.negpool.taxdump import Taxdump
    from sanntis.utils import cache_dir, fetch

    out = Path(kw["out"])
    src = str(kw["taxdump"])
    if "://" in src:
        src = fetch(src, cache_dir("ncbi") / src.rsplit("/", 1)[-1], kw.get("taxdump_sha256") or None)
    tax = Taxdump(src)
    pos = positives_from_run(Path(kw["positives"]), tax)
    cands = [read_candidates(Path(c), tax) for c in kw.get("candidates") or ()]
    if kw.get("gtdb_release"):
        rel = str(kw["gtdb_release"])
        cands.append(gtdb_candidates(S.gtdb_tables(rel), tax, priority=kw.get("gtdb_priority", 1),
                                     min_completeness=kw.get("gtdb_min_completeness", 90.0),
                                     max_contamination=kw.get("gtdb_max_contamination", 5.0), release=rel))
    if not cands:
        raise SystemExit("no candidates: give --candidates and/or --gtdb-release")
    cand = _finish_species(pd.concat(cands, ignore_index=True))
    cand = cand.rename(columns={"kingdom": "track"})
    stats: dict = {"candidates": int(len(cand))}
    if kw.get("exclude"):
        ex = read_exclude(kw["exclude"])
        hit = excluded(cand.rename(columns={"track": "kingdom"}), ex)
        stats["candidates_excluded"] = int(hit.sum())
        cand = cand[~hit]
    cand = dedup_candidates(cand[cand["clean_cds"] >= kw.get("min_quota", 100)])
    o = PlanOptions(budget=parse_budget(kw["budget"]), cap=kw.get("cap", 1500), floor_cds=kw.get("floor_cds", 250),
                    min_quota=kw.get("min_quota", 100), seed=kw.get("seed", 0))
    plan, pstats = plan_pool(pos, cand, o)
    stats |= pstats
    if kw.get("breadth"):
        b = _finish_species(read_candidates(Path(kw["breadth"]), tax))
        if kw.get("exclude"):
            b = b[~excluded(b, read_exclude(kw["exclude"]))]
        n = len(plan)
        plan = add_breadth(plan, b, kw.get("breadth_cds", 500))
        stats["breadth_genomes"] = len(plan) - n
    t = genome_table(plan)
    out.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(out, sep="\t", index=False)
    stats["genomes"] = {f"{k}/{s}": int(v) for (k, s), v in t.groupby(["kingdom", "source"]).size().items()}
    stats["quota_cds"] = {k: int(v) for k, v in t.groupby("kingdom")["quota_cds"].sum().items()}
    stats["options"] = {"budget": o.budget, "cap": o.cap, "floor_cds": o.floor_cds, "min_quota": o.min_quota,
                        "seed": o.seed, "taxdump": str(kw["taxdump"]), "gtdb_release": kw.get("gtdb_release")}
    out.with_suffix(".stats.json").write_text(json.dumps(stats, indent=1, default=int))
    print(json.dumps({k: stats[k] for k in ("genomes", "quota_cds")}, indent=1))
