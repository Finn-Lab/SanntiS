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

"""NCBI Entrez lookups with on-disk caches: taxon → superkingdom / lineage.

``NCBI_API_KEY`` is optional (higher rate limit); ``NCBI_EMAIL`` is passed to Entrez when set.
"""

from __future__ import annotations

import json
import logging
import os
import time
import warnings
from collections.abc import Iterable
from pathlib import Path

from sanntis.utils import cache_dir, write_json

log = logging.getLogger(__name__)


def _entrez():
    from Bio import Entrez

    warnings.filterwarnings("ignore", message=r"\s*Email address is not specified")
    Entrez.tool = "sanntis"
    Entrez.email = os.environ.get("NCBI_EMAIL") or None
    if os.environ.get("NCBI_API_KEY"):
        Entrez.api_key = os.environ["NCBI_API_KEY"]
    return Entrez


def _retry(fn, what: str, retries: int = 4):
    """Retry a whole request + parse: Entrez itself rate-limits and retries opening the URL, not a failed read."""
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:  # network / NCBI throttling
            if attempt == retries - 1:
                raise
            log.warning("%s: %s (retry %d)", what, e, attempt + 1)
            time.sleep(2 * (attempt + 1))


def lineage_of(taxon: dict) -> dict[str, str]:
    skip = (None, "no rank", "clade")
    out = {str(x.get("Rank")): str(x.get("ScientificName")) for x in taxon.get("LineageEx", [])
           if x.get("Rank") not in skip}
    if taxon.get("Rank") not in skip:
        out[str(taxon["Rank"])] = str(taxon.get("ScientificName", ""))
    if "domain" in out and "superkingdom" not in out:  # NCBI renamed the rank to "domain" in 2025
        out["superkingdom"] = out["domain"]
    return out


def lineage_group(lin: dict[str, str]) -> str | None:
    """{rank: name} → report lineage: prokaryotes, fungi, plants, other eukaryotes (None if unresolved)."""
    names = set(lin.values())
    if names & {"Bacteria", "Archaea"}:
        return "prokaryotes"
    if "Fungi" in names:
        return "fungi"
    if "Viridiplantae" in names:
        return "plants"
    return "other eukaryotes" if "Eukaryota" in names else None


class LineageCache:
    """taxid → {rank: name} (kingdom of training loci, report lineage, RefSeq sampling by phylum), persisted as
    JSON."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else cache_dir("ncbi") / "lineage.json"
        self.data: dict[str, dict[str, str]] = json.loads(self.path.read_text()) if self.path.exists() else {}

    def resolve(self, taxids: Iterable[int], batch: int = 200) -> dict[int, dict[str, str]]:
        want = sorted({int(t) for t in taxids if t is not None and int(t) > 0})
        todo = [t for t in want if str(t) not in self.data]
        if todo:
            Entrez = _entrez()
            for i in range(0, len(todo), batch):
                ids = todo[i:i + batch]

                def _fetch(ids=ids):
                    with Entrez.efetch(db="taxonomy", id=",".join(map(str, ids)), retmode="xml") as h:
                        return Entrez.read(h)

                for r in _retry(_fetch, f"taxonomy efetch ({len(ids)} ids)"):
                    lin = lineage_of(r)
                    for t in [r.get("TaxId"), *(r.get("AkaTaxIds") or [])]:
                        if t:
                            self.data[str(int(t))] = lin
                for t in ids:
                    self.data.setdefault(str(t), {})
            write_json(self.path, self.data)
        return {t: dict(self.data.get(str(t)) or {}) for t in want}
