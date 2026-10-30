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

"""Negative pool version: layout, writer, verification and resolution.

::

    <pool_version>/
      genomes.parquet         genome, kingdom, lineage, source (gtdb|refseq|custom), accession
      cds_tokens.parquet      genome, contig, cds_idx, cds_id, start, end, strand, md5, tokens, has_features
      clean_runs.parquet      genome, contig, run_id, cds_idx (list[int], consecutive), n_cds
      native_mask.parquet     genome, contig, cds_idx (CDS overlapping an antiSMASH region)
      test_backbones.parquet  genome, contig, run_id, kingdom (isolate test backbones; excluded from background)
      manifest.json           IPS / InterPro / apps, antiSMASH version, GTDB / RefSeq snapshot, counts, coverage
                              (genome-wide QC reference), sha256

``tokens`` is ``None`` for a CDS without annotation (a missing CDS: it breaks background pieces) and ``[]`` for a
CDS without hits. Coordinates are 0-based half-open; ``cds_idx`` is the 0-based genomic position in the contig.
Chunks are not part of the pool: they are cut at train time (they depend on the positives).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from sanntis._version import __version__
from sanntis.utils import ChecksumError, sha256_file, write_json

FILES = ("genomes.parquet", "cds_tokens.parquet", "clean_runs.parquet", "native_mask.parquet",
         "test_backbones.parquet")
MANIFEST = "manifest.json"
DEFAULT_REPO = "microbiome-informatics/sanntis-training-data"
SCHEMA_VERSION = 1

COLUMNS = {
    "genomes.parquet": ["genome", "kingdom", "lineage", "source", "accession"],
    "cds_tokens.parquet": ["genome", "contig", "cds_idx", "cds_id", "start", "end", "strand", "md5", "tokens"],
    "clean_runs.parquet": ["genome", "contig", "run_id", "cds_idx", "n_cds"],
    "native_mask.parquet": ["genome", "contig", "cds_idx"],
    "test_backbones.parquet": ["genome", "contig", "run_id", "kingdom"],
}


class PoolError(ValueError):
    pass


@dataclass
class Pool:
    path: Path
    manifest: dict

    @property
    def version(self) -> str:
        return self.manifest["pool_version"]

    @property
    def manifest_sha256(self) -> str:
        return sha256_file(self.path / MANIFEST)


def counts_of(frames: dict[str, pd.DataFrame]) -> dict:
    g, c, r = frames["genomes.parquet"], frames["cds_tokens.parquet"], frames["clean_runs.parquet"]
    kingdom = dict(zip(g["genome"], g["kingdom"], strict=True))
    out: dict = {}
    for k in ("proka", "euk"):
        gk = set(g.loc[g["kingdom"] == k, "genome"])
        ck = c[c["genome"].isin(gk)]
        rk = r[r["genome"].isin(gk)]
        tok = ck["tokens"]
        out[k] = {"genomes": len(gk), "cds": len(ck), "cds_without_tokens": int(tok.isna().sum()),
                  "cds_with_hits": int(tok.map(lambda t: t is not None and len(t) > 0).sum()),
                  "unique_md5": int(ck["md5"].nunique()), "clean_runs": len(rk),
                  "clean_run_cds": int(rk["n_cds"].sum()),
                  "native_mask_cds": int(frames["native_mask.parquet"]["genome"].map(kingdom).eq(k).sum()),
                  "test_backbones": int((frames["test_backbones.parquet"]["kingdom"] == k).sum())}
    return out


def coverage_of(cds: pd.DataFrame, app_counts: pd.Series, kingdom: dict[str, str]) -> dict:
    """Genome-wide annotation coverage per kingdom (the bundle's QC reference, see ``sanntis.qc``), counted per CDS
    as ``predict`` does. ``app_counts``: hits per (md5, app); ``kingdom``: genome → kingdom. CDS without annotation
    (``tokens`` None) are left out."""
    ac = app_counts.reset_index(name="n")
    out = {}
    for k in ("proka", "euk"):
        ann = cds[cds["tokens"].notna() & cds["genome"].map(kingdom).eq(k)]
        n = len(ann)
        if not n:
            continue
        per_md5 = ann["md5"].value_counts()
        a = ac[ac["md5"].isin(per_md5.index)]
        tot = (a["n"] * a["md5"].map(per_md5)).groupby(a["app"]).sum()
        out[k] = {"n_cds": n, "frac_cds_with_tokens": round(float(ann["tokens"].map(len).gt(0).mean()), 4),
                  "hits_per_cds": {app: round(float(c) / n, 4) for app, c in sorted(tot.items())}}
    return out


def validate(frames: dict[str, pd.DataFrame]) -> None:
    for name, cols in COLUMNS.items():
        if missing := [c for c in cols if c not in frames[name].columns]:
            raise PoolError(f"{name}: missing columns {missing}")
    c = frames["cds_tokens.parquet"]
    if c.duplicated(["genome", "contig", "cds_idx"]).any():
        raise PoolError("cds_tokens: duplicate (genome, contig, cds_idx)")
    sizes = c.groupby(["genome", "contig"]).size()
    for g, ct, idx in frames["clean_runs.parquet"][["genome", "contig", "cds_idx"]].itertuples(index=False):
        idx = list(idx)
        if not idx or idx != list(range(idx[0], idx[0] + len(idx))) or idx[-1] >= sizes.get((g, ct), 0):
            raise PoolError(f"clean run of {g}/{ct} is not a consecutive in-range cds_idx list")
    if not set(frames["test_backbones.parquet"]["run_id"]) <= set(frames["clean_runs.parquet"]["run_id"]):
        raise PoolError("test backbones must be clean runs")


def write_pool(out: Path, frames: dict[str, pd.DataFrame], meta: dict) -> Pool:
    """Validate and write a pool version; the manifest records the sha256 of every file."""
    out = Path(out)
    validate(frames)
    out.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        df = frames[name].reset_index(drop=True)
        df.to_parquet(out / name, index=False, compression="zstd")
    man = {"schema_version": SCHEMA_VERSION, "sanntis_version": __version__,
           "created": datetime.now(UTC).isoformat(timespec="seconds"), **meta,
           "counts": counts_of(frames), "files": {n: sha256_file(out / n) for n in FILES}}
    write_json(out / MANIFEST, man)
    return Pool(out, man)


def load_pool(path: Path, *, verify: bool = True) -> Pool:
    path = Path(path)
    if not (path / MANIFEST).is_file():
        raise PoolError(f"{path}: not a negative pool (no {MANIFEST})")
    man = json.loads((path / MANIFEST).read_text())
    if verify:
        for n, sha in man["files"].items():
            got = sha256_file(path / n)
            if got != sha:
                raise ChecksumError(f"{path / n}: sha256 {got} != manifest {sha}")
    return Pool(path, man)


def hub_dataset_dir(spec: str, kind: str) -> tuple[Path, str]:
    """``[<repo>@]<version>`` of the training-data dataset on the Hub → (local directory, version).

    A version of ``kind`` lives under ``<kind>/<version>/`` and is tagged ``<kind>-<version>``. The repository
    defaults to ``$SANNTIS_DATA_REPO`` or :data:`DEFAULT_REPO`.
    """
    s = str(spec)
    repo, _, version = s.rpartition("@") if "@" in s else (os.environ.get("SANNTIS_DATA_REPO", DEFAULT_REPO), "", s)
    if not version or "/" in version:
        raise PoolError(f"{spec!r}: not a directory and not a dataset version ([<repo>@]<version>)")
    from huggingface_hub import snapshot_download

    local = snapshot_download(repo_id=repo, repo_type="dataset", revision=f"{kind}-{version}",
                              allow_patterns=[f"{kind}/{version}/*"])
    return Path(local) / kind / version, version


def resolve_pool(spec: str | Path) -> Pool:
    """A local pool directory, or ``[<repo>@]<version>`` of the training-data dataset on the Hub
    (``negatives/<version>/``, tag ``negatives-<version>``)."""
    p = Path(spec)
    if p.is_dir():
        return load_pool(p)
    return load_pool(hub_dataset_dir(str(spec), "negatives")[0])
