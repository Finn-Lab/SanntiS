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

"""``sanntis predict``: inputs → CDS → (pooled) annotation → tokens → Predictor → GFF3 / JSON / TSV outputs."""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from sanntis._version import __version__
from sanntis.annotate import ips6
from sanntis.annotate.genecall import call_genes
from sanntis.config import NextflowOptions, load_recipe
from sanntis.io.antismash import write_sideload
from sanntis.io.fasta import InputError, input_kind, input_prefix, read_nucleotide_fasta, write_cds_faa, write_cds_gff
from sanntis.io.genbank import read_genbank
from sanntis.io.gff import write_gff
from sanntis.io.gff_cds import read_faa_gff
from sanntis.io.ips_tsv import IpsHits, check_apps, join_tokens, read_ips_tsv, tokens_by_key
from sanntis.io.tables import write_orf_scores, write_regions
from sanntis.model.bundle import SanntisModel, default_model_id
from sanntis.predict import Predictor
from sanntis.qc import coverage, flags

log = logging.getLogger("sanntis.predict")

CDS_SOURCE = {"fasta": "pyrodigal", "genbank": "GenBank", "protein": "GFF"}  # --write-faa GFF3 column 2


@dataclass
class Sample:
    id: str
    input: Path
    ips_tsv: Path | None = None
    gff: Path | None = None
    kind: str = ""
    cds: pd.DataFrame | None = None
    meta: dict = field(default_factory=dict)
    # annotation (annotate_managed / annotate_tsv)
    tokens: list[list[str]] = field(default_factory=list)  # per CDS
    app_hits: Counter = field(default_factory=Counter)
    ips: dict = field(default_factory=dict)  # run.json "ips"


def read_samples(inputs, ips_tsv, gff=None) -> list[Sample]:
    if not inputs:
        raise InputError("no inputs: give .fna / .gbk files, or one .faa with --gff")
    samples = [Sample(input_prefix(p), Path(p), Path(ips_tsv) if ips_tsv else None) for p in inputs]
    ids = Counter(s.id for s in samples)
    dup = [i for i, n in ids.items() if n > 1]
    if dup:
        raise InputError(f"duplicate output prefixes: {dup} (inputs need distinct file names)")
    for s in samples:
        s.kind = input_kind(s.input)
    if gff:
        if len(samples) != 1 or samples[0].kind != "protein":
            raise InputError("--gff goes with exactly one .faa input")
        samples[0].gff = Path(gff)
    elif any(s.kind == "protein" for s in samples):
        raise InputError("a .faa input needs --gff (the GFF3 with its CDS)")
    return samples


def load_cds(s: Sample, kingdom: str, genecall: str, threads: int) -> None:
    if s.kind == "fasta":
        if kingdom == "euk":
            raise InputError(f"{s.input}: eukaryotic nucleotide FASTA is not supported (no eukaryotic gene caller); "
                             "give a GenBank file with CDS translations")
        recs = read_nucleotide_fasta(s.input)
        s.cds = call_genes(recs, genecall, threads=threads)
        s.meta.update(records=len(recs), bp=sum(len(r.seq) for r in recs), genecall=f"pyrodigal {genecall}")
    elif s.kind == "protein":
        s.cds, st = read_faa_gff(s.input, s.gff)
        s.meta.update(records=st.records, bp=None, gff=str(s.gff.resolve()), compound_locations=st.compound,
                      unmatched_proteins=st.unmatched_proteins)
    else:
        s.cds, st = read_genbank(s.input)
        s.meta.update(records=st.records, bp=sum(st.record_lengths.values()), pseudo_skipped=st.pseudo_skipped,
                      compound_locations=st.compound)
    s.meta["cds"] = len(s.cds)
    log.info("%s: %d records, %d CDS", s.id, s.meta["records"], len(s.cds))


def _app_counts(hits: pd.DataFrame, keys) -> Counter:
    """Hits per app over the CDS of one input. ``keys``: per CDS ``(column, value)`` or None; a protein is counted
    once per CDS carrying it."""
    c: Counter = Counter()
    for col, vals in pd.DataFrame([k for k in keys if k], columns=["col", "val"]).groupby("col")["val"]:
        n_cds = vals.value_counts()  # CDS per protein
        per = hits[hits[col].isin(n_cds.index)].groupby([col, "app"]).size().reset_index(name="n")
        c.update({app: int(n) for app, n in (per["n"] * per[col].map(n_cds)).groupby(per["app"]).sum().items()})
    return c


def annotate_managed(samples: list[Sample], m: SanntisModel, opts: NextflowOptions, run_dir: Path, *,
                     datadir: Path, keep_work: bool, allow_incomplete_ips: bool) -> None:
    """The proteins of every input, md5-deduplicated, in one InterProScan 6 run (annotation cache first)."""
    r = load_recipe().ips
    proteins = {md5: p for s in samples for md5, p in zip(s.cds["md5"], s.cds["protein"], strict=True)}
    ann = ips6.annotate(proteins, opts, run_dir, datadir=datadir, keep_work=keep_work)
    pooled = IpsHits(hits=ann.hits.assign(protein_id=ann.hits["md5"]), version="6", n_proteins=len(proteins),
                     per_app=Counter(ann.hits["app"]))
    missing_apps = check_apps(pooled, allow_incomplete=allow_incomplete_ips)
    by_md5 = tokens_by_key(ann.hits, m.signature_map, key="md5")
    meta = {"mode": "managed", "ips_version": r.version, "interpro_release": r.interpro,
            "detected_tsv_version": "6", "pipeline": r.pipeline, "apps": r.app_names,
            "unique_proteins": ann.n_proteins, "cache_hits": ann.n_cached, "sent_to_ips6": ann.n_sent,
            "setup_ran": ann.setup_ran, "missing_apps": missing_apps, "datadir": str(datadir),
            "container_engine": opts.container_engine, "nf_config": [str(Path(c).resolve()) for c in opts.nf_config],
            "nf_profile": list(opts.nf_profile), "nf_args": opts.nf_args}
    log.info("IPS6: %d unique proteins, %d from cache, %d annotated now", ann.n_proteins, ann.n_cached, ann.n_sent)
    for s in samples:
        s.tokens = [by_md5.get(x, []) for x in s.cds["md5"]]
        s.app_hits = _app_counts(ann.hits, [("md5", x) for x in s.cds["md5"]])
        s.ips = meta


def annotate_tsv(samples: list[Sample], m: SanntisModel, *, ips_release: str | None,
                 allow_incomplete_ips: bool) -> None:
    """Tokens from a precomputed InterProScan TSV, joined to the CDS by protein id, else md5."""
    bundle_release = m.manifest.get("ips", {}).get("interpro")
    parsed: dict[Path, tuple[IpsHits, list[str]]] = {}  # TSV → hits, missing apps
    for s in samples:
        if s.ips_tsv not in parsed:
            h = read_ips_tsv(s.ips_tsv)
            parsed[s.ips_tsv] = h, check_apps(h, allow_incomplete=allow_incomplete_ips)
            log.info("%s: InterProScan %s TSV, %d proteins, dropped analyses %s", s.ips_tsv, h.version, h.n_proteins,
                     dict(h.dropped))
        h, missing_apps = parsed[s.ips_tsv]
        release = ips_release or "unknown"
        if release != bundle_release:
            log.warning("--ips-tsv InterPro release %s differs from the bundle's %s (declare it with --ips-release)",
                        release, bundle_release)
        toks, keys, js = join_tokens(s.cds, h.hits, m.signature_map)
        if not h.hits.empty and js.by_id + js.by_md5 == 0:
            raise InputError(f"{s.ips_tsv}: no protein matches a CDS of {s.input} by id or md5 (wrong TSV?)")
        if js.id_md5_conflicts > 0.1 * max(js.by_id, 1):
            log.warning("%s: %d CDS matched by id whose sequence differs from the TSV (different gene calls?)",
                        s.id, js.id_md5_conflicts)
        s.tokens = toks
        s.app_hits = _app_counts(h.hits, keys)
        s.ips = {"mode": "tsv", "tsv": str(s.ips_tsv.resolve()), "detected_tsv_version": h.version,
                 "ips_version": f"{h.version}.x" if h.version in ("5", "6") else "unknown",
                 "interpro_release": release, "tsv_proteins": h.n_proteins,
                 "dropped_analyses": dict(h.dropped), "missing_apps": missing_apps, "join": vars(js)}


def gff_header(s: Sample, m: SanntisModel, qc: dict, qc_flags: list[str], kingdom: str, npc_precision: str) -> dict:
    """``##`` lines of the GFF3 (also the antiSMASH sideload ``configuration``)."""
    b, im = m.manifest.get("ips", {}), s.ips
    return {
        "input": s.input.name,
        "model-source": m.source,
        "model-sanntis-version": m.manifest.get("sanntis_version", "unknown"),
        "model-created": m.manifest.get("created", "unknown"),
        "model-ips": f"{b.get('version', 'unknown')} InterPro {b.get('interpro', 'unknown')}",
        "ips": (f"InterProScan {im['ips_version']} InterPro {im['interpro_release']} ({im['mode']}"
                + (f", TSV v{im['detected_tsv_version']})" if im["mode"] == "tsv" else ")")),
        "kingdom": kingdom,
        "npc-precision": npc_precision,
        "qc-cds-with-tokens": f"{qc['frac_cds_with_tokens']:.4f} ({qc['n_cds_with_tokens']}/{qc['n_cds']})",
        "qc-unk-token-share": f"{qc['frac_unk_tokens']:.4f}",
        "qc-flags": ",".join(qc_flags) or "none",
    }


def predict_sample(s: Sample, m: SanntisModel, predictor: Predictor, out: Path, *, kingdom: str, npc_precision: str,
                   write_faa: bool, timings: dict[str, float], t0: float) -> Path:
    """Predict one input and write its outputs; returns the GFF3 path."""
    t = time.time()
    cds = s.cds.assign(tokens=s.tokens)
    pred = predictor.predict(cds)
    qc = coverage(s.tokens, s.app_hits, m.vocab)
    qc_flags = flags(qc, m.config.coverage_reference, load_recipe().ips.app_names, kingdom)
    if s.ips.get("missing_apps"):
        qc_flags.append("missing_apps:" + "|".join(s.ips["missing_apps"]))
    if qc_flags:
        log.warning("%s: QC flags %s", s.id, qc_flags)
    log.info("%s: %d/%d CDS with tokens (%.1f%%), <unk> share %.3f, %d regions", s.id, qc["n_cds_with_tokens"],
             qc["n_cds"], 100 * qc["frac_cds_with_tokens"], qc["frac_unk_tokens"], len(pred.regions))
    header = gff_header(s, m, qc, qc_flags, kingdom, npc_precision)
    files = {"gff": out / f"{s.id}.sanntis.gff3", "orf_scores": out / f"{s.id}.orf_scores.tsv",
             "regions": out / f"{s.id}.regions.tsv", "antismash": out / f"{s.id}.sanntis.antismash.json",
             "run": out / f"{s.id}.run.json"}
    write_gff(pred.regions, files["gff"], header=header)
    write_sideload(pred.regions, cds["record_id"].unique(), files["antismash"], configuration=header)
    write_orf_scores(pred.orf_scores, files["orf_scores"])
    write_regions(pred.regions, files["regions"])
    if write_faa:
        files["faa"] = out / f"{s.id}.cds.faa"
        files["cds_gff"] = out / f"{s.id}.cds.gff3"
        write_cds_faa(s.cds, files["faa"])
        write_cds_gff(s.cds, files["cds_gff"], CDS_SOURCE[s.kind])
    meta = {
        "sanntis_version": __version__, "created": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "command": sys.argv, "sample": {"id": s.id, "input": str(s.input.resolve()), "kind": s.kind, **s.meta},
        "model": {"source": m.source, "sanntis_version": m.manifest.get("sanntis_version"),
                  "created": m.manifest.get("created"), "ips": m.manifest.get("ips", {}),
                  "decode_threshold": m.config.decode.threshold},
        "kingdom": kingdom, "npc_precision": float(npc_precision), "ips": s.ips,
        "counts": {"cds": len(cds), "regions": len(pred.regions)}, "qc": qc, "qc_flags": qc_flags,
        "outputs": {k: str(v.resolve()) for k, v in files.items()},
        "timings_s": {**{k: round(v, 2) for k, v in timings.items()}, "predict": round(time.time() - t, 2),
                      "total_elapsed": round(time.time() - t0, 2)},
    }
    files["run"].write_text(json.dumps(meta, indent=2, default=str))
    return files["gff"]


def run_cli(inputs=(), gff=None, model=None, outdir=".", ips_tsv=None, ips_release=None, kingdom="proka",
            genecall="meta", npc_precision="0.9", threads=1, allow_ips_mismatch=False, allow_incomplete_ips=False,
            write_faa=False, container_engine=None, nf_config=(), nf_profile=(), nf_args="", ips_datadir=None,
            keep_work=False) -> list[Path]:
    t0 = time.time()
    timings: dict[str, float] = {}
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    samples = read_samples(inputs, ips_tsv, gff)
    opts = NextflowOptions(container_engine=container_engine, nf_config=list(nf_config), nf_profile=list(nf_profile),
                           nf_args=nf_args or "")
    managed = [s for s in samples if s.ips_tsv is None]
    if managed:
        ips6.check_nf_args(opts)

    m = SanntisModel.from_pretrained(model or default_model_id(), allow_ips_mismatch=allow_ips_mismatch)
    timings["load_model"] = time.time() - t0

    t = time.time()
    for s in samples:
        load_cds(s, kingdom, genecall, threads)
    timings["cds"] = time.time() - t

    t = time.time()
    if managed:
        annotate_managed(managed, m, opts, out / ".sanntis" / "ips6_run", keep_work=keep_work,
                         datadir=Path(ips_datadir) if ips_datadir else ips6.default_datadir(),
                         allow_incomplete_ips=allow_incomplete_ips)
    annotate_tsv([s for s in samples if s.ips_tsv is not None], m, ips_release=ips_release,
                 allow_incomplete_ips=allow_incomplete_ips)
    timings["annotation"] = time.time() - t

    predictor = Predictor(m, kingdom=kingdom, npc_precision=float(npc_precision), threads=threads)
    written = [predict_sample(s, m, predictor, out, kingdom=kingdom, npc_precision=npc_precision, write_faa=write_faa,
                              timings=timings, t0=t0) for s in samples]
    log.info("done: %d inputs in %.1f s → %s", len(samples), time.time() - t0, out)
    return written
