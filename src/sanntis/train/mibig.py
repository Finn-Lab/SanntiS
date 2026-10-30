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

"""MIBiG-format training data: ``*_json*.tar.gz`` (one JSON entry per BGC) + ``*_gbk*.tar.gz`` (one GenBank record
per BGC, member name ``<accession>.gbk``). MIBiG releases are fetched by version; external data uses the same format,
validated and read through the same fields (MIBiG 4 JSON schema):

    accession                       BGC id (member names of both archives)
    status                          MIBiG only: ``active`` entries are used
    biosynthesis.classes[].class    NRPS / PKS / ribosomal / terpene / saccharide / other (at least one)
    taxonomy.ncbiTaxId              NCBI taxon (→ kingdom, lineage)
    compounds[].structure           SMILES (→ NPClassifier labels, :mod:`sanntis.train.labels`)
"""

from __future__ import annotations

import json
import logging
import re
import tarfile
from pathlib import Path

from Bio.SeqRecord import SeqRecord

from sanntis.config import load_recipe
from sanntis.io.genbank import read_records
from sanntis.utils import cache_dir, fetch

log = logging.getLogger(__name__)

BASE_URL = "https://dl.secondarymetabolites.org/mibig/"
ID_RE = re.compile(r"^[A-Za-z0-9_.:-]+$")
CLASSES = tuple(load_recipe().postfilter.classes)  # MIBiG biosynthetic classes


class FormatError(ValueError):
    pass


def archive_names(version: str) -> dict[str, str]:
    v = version.removeprefix("v")
    return {"json": f"mibig_json_{v}.tar.gz", "gbk": f"mibig_gbk_{v}.tar.gz"}


def release_archives(version: str, root: Path | None = None) -> dict[str, Path]:
    """The JSON and GBK archives of a MIBiG release, downloaded once into ``$SANNTIS_CACHE/mibig``."""
    root = root or cache_dir("mibig")
    out = {}
    for kind, name in archive_names(version).items():
        try:
            out[kind] = fetch(BASE_URL + name, Path(root) / name)
        except OSError as e:
            raise FormatError(f"MIBiG {version}: cannot get {BASE_URL + name} ({e}); a release needs both "
                              f"{' and '.join(archive_names(version).values())}") from e
    return out


def _members(path: Path, suffix: str):
    try:
        with tarfile.open(path) as tf:
            for m in tf:
                if m.isfile() and m.name.endswith(suffix) and not Path(m.name).name.startswith("."):
                    yield m.name, tf.extractfile(m).read()  # type: ignore[union-attr]
    except (tarfile.TarError, OSError) as e:
        raise FormatError(f"{path}: not a readable tar archive ({e})") from e


def read_entries(path: Path) -> dict[str, dict]:
    """accession → JSON entry. Unparsable files, entries without an accession and duplicates are errors."""
    out: dict[str, dict] = {}
    errors = []
    for name, raw in _members(Path(path), ".json"):
        try:
            d = json.loads(raw)
        except json.JSONDecodeError as e:
            errors.append(f"{name}: invalid JSON ({e})")
            continue
        acc = d.get("accession") if isinstance(d, dict) else None
        if not isinstance(acc, str) or not acc:
            errors.append(f"{name}: no accession")
        elif acc in out:
            errors.append(f"{name}: duplicate accession {acc}")
        else:
            out[acc] = d
    if errors:
        raise FormatError(f"{path}: {len(errors)} problem(s):\n  " + "\n  ".join(errors))
    if not out:
        raise FormatError(f"{path}: no JSON entries")
    return out


def read_gbk_archive(path: Path, accessions: set[str] | None = None) -> dict[str, bytes]:
    """accession (member file stem) → raw GBK bytes (only ``accessions`` when given)."""
    out: dict[str, bytes] = {}
    for name, raw in _members(Path(path), ".gbk"):
        acc = Path(name).stem
        if acc in out:
            raise FormatError(f"{path}: duplicate GBK member {acc}")
        if accessions is None or acc in accessions:
            out[acc] = raw
    return out


def validate_entry(entry: dict, *, mibig: bool) -> list[str]:
    """Problems with the fields SanntiS reads (empty = valid)."""
    errs = []
    acc = entry.get("accession")
    if not ID_RE.match(str(acc)):
        errs.append(f"accession {acc!r}: only letters, digits and _ . : - are allowed")
    if mibig and not isinstance(entry.get("status"), str):
        errs.append("status: missing")
    cl = (entry.get("biosynthesis") or {}).get("classes")
    if not isinstance(cl, list) or not cl:
        errs.append("biosynthesis.classes: missing or empty")
    else:
        names = [c.get("class") if isinstance(c, dict) else None for c in cl]
        if bad := sorted({str(n) for n in names if n not in CLASSES}):
            errs.append(f"biosynthesis.classes: unknown class {bad} (allowed {list(CLASSES)})")
    tax = entry.get("taxonomy")
    if not isinstance(tax, dict) or "ncbiTaxId" not in tax:
        errs.append("taxonomy.ncbiTaxId: missing")
    elif taxid_of(entry) is None:
        errs.append(f"taxonomy.ncbiTaxId: not an integer ({tax['ncbiTaxId']!r})")
    comps = entry.get("compounds", [])
    if not isinstance(comps, list) or any(not isinstance(c, dict) for c in comps):
        errs.append("compounds: must be a list of objects")
    elif any(not isinstance(c.get("structure", ""), str) for c in comps):
        errs.append("compounds[].structure: must be a SMILES string")
    return errs


def classes_of(entry: dict) -> list[str]:
    return sorted({c["class"] for c in (entry.get("biosynthesis") or {}).get("classes", []) if c.get("class")})


def taxid_of(entry: dict) -> int | None:
    t = (entry.get("taxonomy") or {}).get("ncbiTaxId")
    if isinstance(t, bool):
        return None
    try:
        return int(t) if t not in (None, "") else None
    except (TypeError, ValueError):
        return None


def record_taxid(rec: SeqRecord) -> int | None:
    """``/db_xref="taxon:N"`` of the record's source feature."""
    for f in rec.features:
        if f.type == "source":
            for x in f.qualifiers.get("db_xref", []):
                if x.startswith("taxon:") and x[6:].strip().isdigit():
                    return int(x[6:])
    return None


def archive_record(raw: bytes) -> SeqRecord:
    """The one record of an archive GBK (MIBiG GBKs hold the first locus)."""
    recs = read_records(raw)
    if len(recs) != 1:
        raise FormatError(f"{len(recs)} GenBank records (one record = one BGC)")
    return recs[0]
