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

"""Genome sampling for the negative pool: GTDB representatives and RefSeq eukaryotes per phylum.

Output (``genomes.tsv``): genome, kingdom, lineage, source, accession, group, taxon (antiSMASH ``--taxon``),
url (assembly FTP directory, filled by the downloader when empty) and optional local ``path``.
"""

from __future__ import annotations

import logging
import zlib
from pathlib import Path

import pandas as pd

from sanntis.utils import cache_dir, fetch

log = logging.getLogger(__name__)

GTDB_BASE = "https://data.gtdb.ecogenomic.org/releases/release{r}/{r}.0/"
GTDB_TABLES = ("bac120_metadata_r{r}.tsv.gz", "ar53_metadata_r{r}.tsv.gz")
REFSEQ_SUMMARY = "https://ftp.ncbi.nlm.nih.gov/genomes/refseq/{group}/assembly_summary.txt"
# group → (genomes per phylum, RefSeq directory): 2-3 per phylum (fungi, protozoa, plants)
REFSEQ_GROUPS = {"fungi": (3, "fungi"), "protozoa": (2, "protozoa"), "plants": (2, "plant")}
ASSEMBLY_LEVELS = ("Complete Genome", "Chromosome", "Scaffold")
GENOME_COLUMNS = ["genome", "kingdom", "lineage", "source", "accession", "group", "taxon", "url", "path"]
RANKS = {"d": "domain", "p": "phylum", "c": "class", "o": "order", "f": "family", "g": "genus", "s": "species"}


def balanced_sample(df: pd.DataFrame, label_col: str, n_samples: int, seed: int) -> pd.DataFrame:
    """GTDB sampler: ``n_samples`` rows spread evenly over the ``label_col`` classes (the remainder goes to the
    largest classes), then shuffled."""
    from sklearn.utils import resample

    counts = df[label_col].dropna().astype(str).value_counts()
    names = counts.index.tolist()
    base, rem = divmod(n_samples, len(names))
    target = {c: min(base, int(counts[c])) for c in names}
    for c in counts.sort_values(ascending=False).index:
        if rem <= 0:
            break
        if target[c] < int(counts[c]):
            target[c] += 1
            rem -= 1
    parts, used = [], set()
    for c in names:
        if target[c] > 0:
            s = resample(df[df[label_col].astype(str) == c], n_samples=target[c], replace=False, random_state=seed)
            parts.append(s)
            used.update(s.index)
    out = pd.concat(parts) if parts else df.iloc[:0]
    missing = n_samples - len(out)
    if missing > 0:
        rest = df.loc[~df.index.isin(list(used))]
        if len(rest):
            out = pd.concat([out, resample(rest, n_samples=min(missing, len(rest)), replace=False,
                                           random_state=seed + 1)])
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def split_gtdb_taxonomy(tax: pd.Series) -> pd.DataFrame:
    rows = []
    for t in tax.fillna(""):
        r = {v: None for v in RANKS.values()}
        for seg in str(t).split(";"):
            seg = seg.strip()
            if len(seg) > 3 and seg[1:3] == "__" and seg[0] in RANKS:
                r[RANKS[seg[0]]] = seg[3:] or None
        rows.append(r)
    return pd.DataFrame(rows, index=tax.index)


def gtdb_tables(release: str = "226", root: Path | None = None) -> list[Path]:
    root = root or cache_dir(f"gtdb/r{release}")
    return [fetch(GTDB_BASE.format(r=release) + t.format(r=release), root / t.format(r=release)) for t in GTDB_TABLES]


def sample_gtdb(tables: list[Path], n_samples: int = 205, seed: int = 42, rank: str = "phylum") -> pd.DataFrame:
    """GTDB species representatives, balanced over ``rank`` (205 genomes over ~204 phyla: 1-2 per phylum)."""
    df = pd.concat([pd.read_csv(t, sep="\t", dtype=str, low_memory=False) for t in tables], ignore_index=True)
    df = df[df["gtdb_representative"].str.lower().eq("t")].copy()
    df = df.join(split_gtdb_taxonomy(df["gtdb_taxonomy"]))
    df = df[df[rank].notna()]
    s = balanced_sample(df, rank, n_samples, seed)
    acc = s["accession"].str.replace(r"^(RS_|GB_)", "", regex=True)
    return pd.DataFrame({"genome": acc, "kingdom": "proka", "lineage": s["gtdb_taxonomy"], "source": "gtdb",
                         "accession": acc, "group": s["domain"], "taxon": "bacteria", "url": "", "path": ""})


def load_assembly_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=1, dtype=str, low_memory=False, na_values=["", "na"])
    df.columns = [c.lstrip("#").strip() for c in df.columns]
    return df


def sample_refseq(summaries: dict[str, Path], lineages: dict[int, dict[str, str]], seed: int = 0) -> pd.DataFrame:
    """RefSeq eukaryotes: ``REFSEQ_GROUPS[group][0]`` assemblies per NCBI phylum (seeded by crc32 of the phylum).

    ``lineages``: taxid → {rank: name} (see :class:`sanntis.train.ncbi.LineageCache`).
    """
    out = []
    for group, path in summaries.items():
        per = REFSEQ_GROUPS[group][0]
        df = load_assembly_summary(path)
        df = df[df["ftp_path"].fillna("").str.startswith("http") & df["taxid"].fillna("").str.isdigit()
                & df["assembly_level"].isin(ASSEMBLY_LEVELS)].copy()
        df["phylum"] = [lineages.get(int(t), {}).get("phylum", "") for t in df["taxid"]]
        df = df[df["phylum"] != ""]
        for phylum, g in df.sort_values("assembly_accession").groupby("phylum"):
            s = g.sample(n=min(per, len(g)), random_state=seed + zlib.crc32(phylum.encode()) % 10_000)
            for r in s.itertuples(index=False):
                lin = lineages.get(int(r.taxid), {})
                out.append({"genome": r.assembly_accession, "kingdom": "euk",
                            "lineage": ";".join(f"{k}__{lin[k]}" for k in ("superkingdom", "kingdom", "phylum",
                                                                          "class", "order", "family", "genus",
                                                                          "species") if lin.get(k)),
                            "source": "refseq", "accession": r.assembly_accession, "group": group,
                            "taxon": "fungi", "url": r.ftp_path, "path": ""})
    return pd.DataFrame(out, columns=GENOME_COLUMNS)


def read_genomes(path: Path) -> pd.DataFrame:
    """A user / pre-sampled genome table (TSV). Required: genome, kingdom; ``path`` skips the download."""
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    for c in GENOME_COLUMNS:
        if c not in df.columns:
            df[c] = ""
    if df["genome"].duplicated().any() or not df["kingdom"].isin(["proka", "euk"]).all():
        raise ValueError(f"{path}: genome ids must be unique and kingdom proka|euk")
    df.loc[df["taxon"] == "", "taxon"] = df["kingdom"].map({"proka": "bacteria", "euk": "fungi"})
    df.loc[df["accession"] == "", "accession"] = df["genome"]
    base = Path(path).parent
    df["path"] = [str(base / p) if p and not Path(p).is_absolute() else p for p in df["path"]]
    return df[GENOME_COLUMNS]
