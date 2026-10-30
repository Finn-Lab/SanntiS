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

"""``collect`` step: positives from MIBiG-format sources, NPC labels, negative pool.

A source is a GBK archive plus (optionally) a JSON archive in the MIBiG format (:mod:`sanntis.train.mibig`): a MIBiG
release fetched by version (``active`` entries) and / or external archives (every entry). Both go through the same
reading and validation. External data fails early with every problem listed; a MIBiG entry that cannot be used is
excluded (``exclusions.tsv``). External BGCs without a JSON entry train the detector only (no class, no NPC labels).

Writes ``<run>/collect/``::

    loci.parquet        one row per BGC: entity_id, source (mibig|external), kingdom, class_raw (';'-joined),
                        primary_class, status, length, n_cds, ncbi_tax_id, npc_status, npc_pathway, npc_superclass
    cds.parquet         entity_id, cds_idx, cds_id, locus_tag, protein_id, start, end, strand, md5, length
    proteins.faa        unique proteins, header = md5 (the input of `annotate`)
    exclusions.tsv      entity_id, reason (invalid MIBiG entry, no GBK, > max_locus_bp, no CDS)
    kingdom_map.tsv     accession → ncbi_tax_id, superkingdom, kingdom
    npc_compounds.tsv   per compound: normalised structure, skeleton, NPClassifier terms
    provenance.json     MIBiG release + archive sha256, external archives sha256, NPC labelling (and the cache
                        seed), negative pool
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from sanntis.config import load_recipe
from sanntis.io.fasta import write_protein_fasta
from sanntis.io.genbank import cds_rows, iter_cds
from sanntis.negpool.pool import resolve_pool
from sanntis.train import labels as npc_labels
from sanntis.train import mibig
from sanntis.train.entities import primary_class
from sanntis.train.ncbi import LineageCache
from sanntis.utils import sha256_file, write_json

log = logging.getLogger(__name__)


class CollectError(ValueError):
    pass


@dataclass
class CollectOptions:
    mibig: str | None = None  # release, e.g. "4.0"; None = external data only
    external: list[tuple[Path, Path | None]] = field(default_factory=list)  # (GBK archive, JSON archive or None)
    negatives: str | None = None
    npc_cache: str | None = None  # published NPClassifier answers: directory or Hub dataset version
    max_locus_bp: int = field(default_factory=lambda: load_recipe().dedup.max_locus_bp)
    archive_root: Path | None = None  # where MIBiG archives are cached (default $SANNTIS_CACHE/mibig)
    taxonomy_cache: Path | None = None
    npc_client: npc_labels.NPClassifierClient | None = None


@dataclass
class Locus:
    entity_id: str
    source: str
    entry: dict | None  # JSON entry (None: GBK-only external BGC)
    gbk: bytes  # archive GBK
    kingdom: str = "proka"
    ncbi_tax_id: int | None = None
    record: object = None


def _archive(p: Path) -> dict:
    return {"path": str(Path(p).resolve()), "name": Path(p).name, "sha256": sha256_file(p)}


# ── sources ──────────────────────────────────────────────────────────────────


def mibig_loci(opts: CollectOptions, prov: dict, excl: list[dict]) -> list[Locus]:
    """Active entries of a MIBiG release; invalid ones and those missing from the GBK archive are excluded."""
    paths = mibig.release_archives(opts.mibig, opts.archive_root)  # type: ignore[arg-type]
    entries = mibig.read_entries(paths["json"])
    active = {a: e for a, e in entries.items() if e.get("status") == "active"}
    prov["mibig"] = {"release": opts.mibig.removeprefix("v"),  # type: ignore[union-attr]
                     **{k: {"name": p.name, "sha256": sha256_file(p)} for k, p in paths.items()},
                     "statuses": ["active"], "entries": len(entries),
                     "status_counts": dict(Counter(e.get("status") for e in entries.values()))}
    gbk = mibig.read_gbk_archive(paths["gbk"], set(active))
    out = []
    for acc in sorted(active):
        if errs := mibig.validate_entry(active[acc], mibig=True):
            excl.append({"entity_id": acc, "reason": "json: " + "; ".join(errs)[:200]})
            continue
        if acc not in gbk:
            excl.append({"entity_id": acc, "reason": "gbk: not in the MIBiG GBK archive"})
            continue
        out.append(Locus(acc, "mibig", active[acc], gbk[acc]))
    return out


def external_loci(gbk_path: Path, json_path: Path | None, taken: set[str]) -> list[Locus]:
    """Every BGC of one external archive pair, all problems reported at once."""
    entries = mibig.read_entries(json_path) if json_path else {}
    gbk = mibig.read_gbk_archive(gbk_path)
    errors = []
    for acc in sorted(set(entries) - set(gbk)):
        errors.append(f"{acc}: JSON entry without a GBK in {Path(gbk_path).name}")
    if json_path:
        for acc in sorted(set(gbk) - set(entries)):
            errors.append(f"{acc}: GBK without a JSON entry in {Path(json_path).name}")
    out = []
    for acc in sorted(set(gbk) & set(entries) if json_path else gbk):
        where = f"{acc}:"
        if acc in taken:
            errors.append(f"{where} accession already used (MIBiG or another external archive)")
        if not mibig.ID_RE.match(acc):
            errors.append(f"{where} invalid accession (letters, digits and _ . : - only)")
        e = entries.get(acc)
        if e is not None:
            errors += [f"{where} {x}" for x in mibig.validate_entry(e, mibig=False)]
        try:
            rec = mibig.archive_record(gbk[acc])
            if not list(iter_cds(rec, missing_translation="error")):
                raise mibig.FormatError("no CDS")
        except Exception as x:
            errors.append(f"{where} gbk: {x}")
            continue
        out.append(Locus(acc, "external", e, gbk[acc], record=rec))
    if errors:
        name = Path(json_path).name + " + " if json_path else ""
        raise CollectError(f"{name}{Path(gbk_path).name}: {len(errors)} problem(s):\n  " + "\n  ".join(errors))
    if not out:
        raise CollectError(f"{gbk_path}: no BGCs")
    return out


def _record(loc: Locus):
    return loc.record if loc.record is not None else mibig.archive_record(loc.gbk)


def _kingdoms(loci: list[Locus], opts: CollectOptions) -> pd.DataFrame:
    """Kingdom from the NCBI taxon (JSON ``taxonomy.ncbiTaxId``, else the GBK source ``taxon`` xref); no taxon →
    prokaryote."""
    for loc in loci:
        loc.ncbi_tax_id = (mibig.taxid_of(loc.entry) if loc.entry is not None else
                           mibig.record_taxid(loc.record) if loc.record is not None else None)
    lin = LineageCache(opts.taxonomy_cache).resolve(t for t in (x.ncbi_tax_id for x in loci) if t)
    rows = []
    for loc in loci:
        sup = lin.get(loc.ncbi_tax_id, {}).get("superkingdom", "") if loc.ncbi_tax_id else ""
        loc.kingdom = "euk" if sup == "Eukaryota" else "proka"
        rows.append({"accession": loc.entity_id, "ncbi_tax_id": loc.ncbi_tax_id or -1, "superkingdom": sup,
                     "kingdom": loc.kingdom})
    return pd.DataFrame(rows)


# ── step ─────────────────────────────────────────────────────────────────────


def collect(out_dir: Path, opts: CollectOptions) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prov: dict = {"max_locus_bp": opts.max_locus_bp}
    excl: list[dict] = []
    loci: list[Locus] = []
    if opts.mibig:
        loci = mibig_loci(opts, prov, excl)
    if opts.external:
        taken = {loc.entity_id for loc in loci}
        prov["external"] = []
        for gbk, js in opts.external:
            ext = external_loci(Path(gbk), Path(js) if js else None, taken)
            taken |= {loc.entity_id for loc in ext}
            loci += ext
            prov["external"].append({"gbk": _archive(gbk), "json": _archive(js) if js else None, "n": len(ext),
                                     "with_json": sum(loc.entry is not None for loc in ext)})
    if not opts.mibig and not opts.external:
        raise CollectError("no positives: give --mibig <release> and / or --external-gbk <archive>")

    rows, cds_all, seqs, kept = [], [], {}, []
    for loc in loci:
        try:
            rec = loc.record = _record(loc)
            cds = list(iter_cds(rec))
        except Exception as e:
            log.warning("%s: %s", loc.entity_id, e)
            excl.append({"entity_id": loc.entity_id, "reason": f"gbk: {str(e)[:200]}"})
            continue
        length = len(rec.seq)
        if length > opts.max_locus_bp:
            excl.append({"entity_id": loc.entity_id, "reason": f"length {length} > {opts.max_locus_bp}"})
            continue
        if not cds:
            excl.append({"entity_id": loc.entity_id, "reason": "no CDS"})
            continue
        kept.append((loc, rec, cds))
    if not kept:
        raise CollectError("no positive loci collected")
    kmap = _kingdoms([k[0] for k in kept], opts)
    kmap.to_csv(out_dir / "kingdom_map.tsv", sep="\t", index=False)

    labelled = {loc.entity_id: loc.entry for loc, _, _ in kept if loc.entry is not None}
    if labelled:
        client = opts.npc_client or npc_labels.NPClassifierClient()
        seed = npc_labels.seed_cache(opts.npc_cache, getattr(client, "dir", None)) if opts.npc_cache else None
        npc, comps, prov["npc_labels"] = npc_labels.label_entries(labelled, client)
        prov["npc_labels"]["cache_seed"] = seed
        comps.assign(**{c: comps[c].map("|".join) for c in ("pathway", "superclass", "npc_class")}).to_csv(
            out_dir / "npc_compounds.tsv", sep="\t", index=False)
    else:
        npc = pd.DataFrame(columns=["npc_status", "pathway", "superclass"])

    for loc, rec, cds in kept:
        classes = mibig.classes_of(loc.entry) if loc.entry is not None else []
        has = loc.entity_id in npc.index
        rows.append({"entity_id": loc.entity_id, "source": loc.source, "kingdom": loc.kingdom,
                     "class_raw": ";".join(classes), "primary_class": primary_class(classes) or "",
                     "status": (loc.entry or {}).get("status", "") if loc.source == "mibig" else "external",
                     "length": len(rec.seq), "n_cds": len(cds),
                     "ncbi_tax_id": loc.ncbi_tax_id if loc.ncbi_tax_id is not None else -1,
                     "npc_status": npc.at[loc.entity_id, "npc_status"] if has else "unlabelled",
                     "npc_pathway": list(npc.at[loc.entity_id, "pathway"]) if has else [],
                     "npc_superclass": list(npc.at[loc.entity_id, "superclass"]) if has else []})
        for r in cds_rows(cds):
            r.pop("contig")
            cds_all.append({"entity_id": loc.entity_id, **r})
        seqs.update((c.md5, c.sequence) for c in cds)

    loci_df = pd.DataFrame(rows)
    loci_df.to_parquet(out_dir / "loci.parquet", index=False)
    pd.DataFrame(cds_all).to_parquet(out_dir / "cds.parquet", index=False)
    pd.DataFrame(excl, columns=["entity_id", "reason"]).to_csv(out_dir / "exclusions.tsv", sep="\t", index=False)
    write_protein_fasta(sorted(seqs.items()), out_dir / "proteins.faa")
    if opts.negatives:
        pool = resolve_pool(opts.negatives)
        prov["negatives"] = {"path": str(pool.path), "version": pool.version, "manifest_sha256": pool.manifest_sha256}
    counts = {"loci": len(loci_df), "cds": len(cds_all), "proteins": len(seqs), "excluded": len(excl),
              "by_source": loci_df["source"].value_counts().to_dict(),
              "by_kingdom": loci_df["kingdom"].value_counts().to_dict(),
              "by_primary_class": loci_df["primary_class"].replace("", "unlabelled").value_counts().to_dict(),
              "by_npc_status": loci_df["npc_status"].value_counts().to_dict()}
    prov["counts"] = counts
    write_json(out_dir / "provenance.json", prov)
    log.info("collect: %s", counts)
    return counts

