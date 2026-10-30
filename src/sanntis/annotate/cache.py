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

"""Content-addressed annotation cache: protein md5 × (IPS version, InterPro release, apps hash) → hit rows.

```
$SANNTIS_CACHE/annotations/ips<version>_interpro<release>_<apps hash>/
  hits-<shard>.parquet      md5, app, signature, start, end, ipr   (one row per kept hit)
  proteins-<shard>.parquet  md5                                    (every protein the shard annotated, hits or not)
```

A shard is written once and never modified (hits first, then the protein list, each by atomic rename), so
concurrent writers do not clash and a protein listed in ``proteins-*`` always has its hits on disk. Proteins
without any hit are listed too, so they are never re-annotated.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from sanntis.config import IpsConfig, load_recipe
from sanntis.utils import cache_dir

log = logging.getLogger(__name__)

HIT_COLUMNS = ["md5", "app", "signature", "start", "end", "ipr"]
_SCHEMA = pa.schema([("md5", pa.string()), ("app", pa.string()), ("signature", pa.string()),
                     ("start", pa.int64()), ("end", pa.int64()), ("ipr", pa.string())])


class AnnotationCache:
    def __init__(self, root: Path | None = None, ips: IpsConfig | None = None):
        ips = ips or load_recipe().ips
        self.key = f"ips{ips.version}_interpro{ips.interpro}_{ips.apps_hash}"
        self.dir = Path(root or cache_dir()) / "annotations" / self.key

    def _shards(self, kind: str) -> list[Path]:
        return sorted(self.dir.glob(f"{kind}-*.parquet")) if self.dir.is_dir() else []

    def known(self) -> set[str]:
        """md5 of every annotated protein."""
        out: set[str] = set()
        for p in self._shards("proteins"):
            out.update(pq.read_table(p, columns=["md5"]).column("md5").to_pylist())
        return out

    def lookup(self, md5s: Iterable[str]) -> tuple[set[str], set[str]]:
        """→ (cached, missing)."""
        want = set(md5s)
        hit = want & self.known()
        return hit, want - hit

    def hits(self, md5s: Iterable[str]) -> pd.DataFrame:
        """Hit rows of the given proteins (``HIT_COLUMNS``)."""
        want = set(md5s)
        parts = []
        for p in self._shards("hits"):
            if not want:
                break
            t = pq.read_table(p)
            t = t.filter(pc.is_in(t.column("md5"), value_set=pa.array(sorted(want), pa.string())))
            if t.num_rows:
                df = t.to_pandas()
                parts.append(df)
                want -= set(df["md5"])  # a protein annotated twice (concurrent runs): first shard wins
        if not parts:
            return pd.DataFrame({c: pd.Series(dtype="int64" if c in ("start", "end") else "object")
                                 for c in HIT_COLUMNS})
        return pd.concat(parts, ignore_index=True)[HIT_COLUMNS]

    def add(self, md5s: Iterable[str], hits: pd.DataFrame) -> Path | None:
        """Store one annotation batch: ``md5s`` = every protein that was annotated, ``hits`` their kept rows."""
        md5s = sorted(set(md5s))
        if not md5s:
            return None
        self.dir.mkdir(parents=True, exist_ok=True)
        shard = f"{time.time_ns():020d}-{uuid.uuid4().hex[:8]}"  # sorts by creation time
        h = hits[HIT_COLUMNS].copy()
        extra = set(h["md5"]) - set(md5s)
        if extra:
            raise ValueError(f"{len(extra)} hit rows for proteins outside the batch")
        h["ipr"] = h["ipr"].astype("object").where(h["ipr"].notna(), None)
        self._write(pa.Table.from_pandas(h, schema=_SCHEMA, preserve_index=False), f"hits-{shard}.parquet")
        self._write(pa.table({"md5": md5s}), f"proteins-{shard}.parquet")
        log.info("annotation cache: +%d proteins (%d hits) in %s", len(md5s), len(h), self.dir)
        return self.dir / f"proteins-{shard}.parquet"

    def _write(self, table: pa.Table, name: str) -> None:
        tmp = self.dir / f".{name}.tmp"
        pq.write_table(table, tmp)
        os.replace(tmp, self.dir / name)
