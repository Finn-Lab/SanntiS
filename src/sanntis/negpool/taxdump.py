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

"""NCBI taxonomy from one taxdump (``taxdmp_<date>.zip`` of the NCBI taxdump archive, or its unpacked directory), so
every lineage of a negative-pool plan comes from a single, pinned snapshot."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

RANKS = ("domain", "superkingdom", "kingdom", "phylum", "class", "order", "family", "genus", "species")


def _rows(text: str):
    for line in text.splitlines():
        yield [f.strip() for f in line.rstrip("|\t\n").split("\t|\t")]


class Taxdump:
    def __init__(self, path: str | Path):
        path = Path(path)
        if path.is_dir():
            read = lambda n: (path / n).read_text()  # noqa: E731
        else:
            z = zipfile.ZipFile(path)
            read = lambda n: io.TextIOWrapper(z.open(n), encoding="utf-8").read()  # noqa: E731
        self.parent: dict[int, int] = {}
        self.rank: dict[int, str] = {}
        for f in _rows(read("nodes.dmp")):
            t = int(f[0])
            self.parent[t] = int(f[1])
            self.rank[t] = f[2]
        self.name: dict[int, str] = {int(f[0]): f[1] for f in _rows(read("names.dmp")) if f[3] == "scientific name"}
        self.merged: dict[int, int] = {int(f[0]): int(f[1]) for f in _rows(read("merged.dmp"))}
        self._cache: dict[int, dict[str, str]] = {}

    def lineage(self, taxid: object) -> dict[str, str]:
        """taxid → {rank: name} over :data:`RANKS` (``superkingdom`` filled from ``domain`` and vice versa); {} when
        the taxid is unknown."""
        try:
            t = int(float(str(taxid)))
        except (TypeError, ValueError):
            return {}
        t = self.merged.get(t, t)
        if t in self._cache:
            return self._cache[t]
        out: dict[str, str] = {}
        cur, n = t, 0
        while cur in self.parent and n < 200:
            r = self.rank.get(cur)
            if r in RANKS and r not in out and cur in self.name:
                out[r] = self.name[cur]
            if self.parent[cur] == cur:
                break
            cur, n = self.parent[cur], n + 1
        if "domain" in out:
            out.setdefault("superkingdom", out["domain"])
        elif "superkingdom" in out:
            out["domain"] = out["superkingdom"]
        self._cache[t] = out
        return out
