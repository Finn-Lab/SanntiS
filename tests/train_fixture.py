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

"""Synthetic end-to-end training fixture: external BGCs as MIBiG-format archives (``fx_json.tar.gz`` +
``fx_gbk.tar.gz``), a negative pool, an md5-keyed IPS TSV, an InterPro ``entries.json`` and a pre-filled NCBI
taxonomy cache (``cache/``, set ``SANNTIS_CACHE`` to it). Small enough for CI (a full ``sanntis train`` takes ~1-2
min on a CPU)."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.Seq import Seq
from Bio.SeqFeature import FeatureLocation, SeqFeature
from Bio.SeqRecord import SeqRecord

from sanntis.negpool.pool import write_pool

AA = "ACDEFGHIKLMNPQRSTVWY"
CLASSES = ["NRPS", "PKS", "terpene"]
TAXA = {"proka": 1396, "euk": 510516}  # Bacillus cereus, Aspergillus oryzae


def write_tar(path: Path, members: dict[str, bytes]) -> Path:
    with tarfile.open(path, "w:gz") as tf:
        for name, data in members.items():
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
    return path


def entry(acc: str, cls: str, kingdom: str, length: int, compounds: list | None = None) -> dict:
    """A minimal MIBiG 4 entry (the fields SanntiS reads)."""
    return {"accession": acc, "status": "active", "biosynthesis": {"classes": [{"class": cls}]},
            "taxonomy": {"name": "synthetic", "ncbiTaxId": TAXA[kingdom]},
            "loci": [{"accession": "XX000001.1", "location": {"from": 1, "to": length}}],
            "compounds": compounds if compounds is not None else [{"name": f"{acc} compound"}]}


def taxonomy_cache(root: Path) -> Path:
    d = root / "ncbi"
    d.mkdir(parents=True, exist_ok=True)
    (d / "lineage.json").write_text(json.dumps({"1396": {"superkingdom": "Bacteria"},
                                                "510516": {"superkingdom": "Eukaryota", "kingdom": "Fungi"},
                                                "243265": {"superkingdom": "Bacteria"}}))
    return root


def _protein(rng) -> str:
    return "M" + "".join(rng.choice(list(AA), size=int(rng.integers(80, 200))))


def _md5(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()


def build(root: Path, n_bgc: int = 60, seed: int = 0) -> dict[str, Path]:
    from Bio import SeqIO

    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    gbks, jsons, tsv = {}, {}, []
    entries = {f"PF{i:05d}": {"database": "Pfam", "integrated": f"IPR{i:06d}" if i % 2 else None}
               for i in range(200)}
    for b in range(n_bgc):
        cls = CLASSES[b % 3]
        kingdom = "euk" if b % 6 == 5 else "proka"
        n = int(rng.integers(4, 9))
        prots = [_protein(rng) for _ in range(n)]
        dna = "".join(rng.choice(list("ACGT"), size=sum(3 * len(p) + 3 for p in prots) + 200))
        rec = SeqRecord(Seq(dna), id=f"custom{b:03d}", name=f"custom{b:03d}", description="synthetic BGC",
                        annotations={"molecule_type": "DNA"})
        pos = 100
        base = {"NRPS": 0, "PKS": 20, "terpene": 40}[cls]
        for i, p in enumerate(prots):
            L = 3 * len(p) + 3
            rec.features.append(SeqFeature(FeatureLocation(pos, pos + L, strand=1), type="CDS",
                                           qualifiers={"locus_tag": [f"c{b:03d}_{i}"], "translation": [p]}))
            pos += L
            sigs = [f"PF{base + int(rng.integers(0, 20)):05d}" for _ in range(int(rng.integers(1, 4)))]
            for j, s in enumerate(sigs):
                m = _md5(p)
                tsv.append(f"{m}\t{m}\t{len(p)}\tPfam\t{s}\t-\t{1 + 10 * j}\t{9 + 10 * j}\t1E-10\tT\t21-10-2026\t"
                           f"{entries[s]['integrated'] or '-'}\t-\t-\t-")
        fh = io.StringIO()
        SeqIO.write([rec], fh, "genbank")
        gbks[f"fx_gbk/custom{b:03d}.gbk"] = fh.getvalue().encode()
        jsons[f"fx_json/custom{b:03d}.json"] = json.dumps(entry(f"custom{b:03d}", cls, kingdom, len(dna))).encode()
    write_tar(root / "fx_gbk.tar.gz", gbks)
    write_tar(root / "fx_json.tar.gz", jsons)
    (root / "ips.tsv").write_text("\n".join(tsv) + "\n")
    (root / "entries.json").write_text(json.dumps(entries))

    # negative pool: 2 proka + 1 euk genome, one 400-CDS contig each, BGC-free runs of 60 CDS
    genomes = pd.DataFrame({"genome": ["g1", "g2", "g3"], "kingdom": ["proka", "proka", "euk"],
                            "lineage": ["bacteria", "bacteria", "fungi"], "source": "custom", "accession": ""})
    cds, runs, native = [], [], []
    for g in genomes["genome"]:
        for i in range(400):
            t = [] if rng.random() < 0.15 else [f"PF{60 + int(rng.integers(0, 140)):05d}"]
            cds.append({"genome": g, "contig": "c1", "cds_idx": i, "cds_id": f"{g}_{i}", "start": i * 1000,
                        "end": i * 1000 + 900, "strand": 1, "md5": f"{g}{i:032d}"[-32:], "tokens": t})
        for r in range(6):
            a = r * 65
            runs.append({"genome": g, "contig": "c1", "run_id": f"r{r}", "cds_idx": list(range(a, a + 60)),
                         "n_cds": 60})
            native += [{"genome": g, "contig": "c1", "cds_idx": j} for j in range(a + 60, a + 65)]
    frames = {"genomes.parquet": genomes, "cds_tokens.parquet": pd.DataFrame(cds),
              "clean_runs.parquet": pd.DataFrame(runs), "native_mask.parquet": pd.DataFrame(native),
              "test_backbones.parquet": pd.DataFrame({"genome": ["g1", "g3"], "contig": "c1", "run_id": "r5",
                                                      "kingdom": ["proka", "euk"]})}
    write_pool(root / "pool", frames, {"pool_version": "fixture", "ips": {"version": "fixture"}})
    return {"gbk": root / "fx_gbk.tar.gz", "json": root / "fx_json.tar.gz", "tsv": root / "ips.tsv",
            "entries": root / "entries.json", "pool": root / "pool", "cache": taxonomy_cache(root / "cache")}


def train_args(fx: dict[str, Path], run: Path, runner: str = "inline") -> list[str]:
    return ["train", "--run", str(run), "--external-gbk", str(fx["gbk"]), "--external-json", str(fx["json"]),
            "--negatives",
            str(fx["pool"]), "--ips-tsv", str(fx["tsv"]), "--entries-json", str(fx["entries"]), "--runner", runner,
            "--workers", "2"]
