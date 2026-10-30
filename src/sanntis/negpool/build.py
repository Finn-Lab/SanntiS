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

"""``sanntis build-negatives``: sample → download + gene calls → antiSMASH 8.0.1 native-BGC
removal → IPS6 on the pool proteins → pool version.

Build directory (``--out``)::

    build.json  genomes.tsv
    negpool_prepare/    fan-out (label cpu_small): download + CDS table per genome → prepared/<genome>/
    negpool_antismash/  fan-out (label antismash): antiSMASH per genome → antismash/<genome>/<genome>.antismash.gbk.gz
    ips6/               unique pool proteins (md5) + the IPS6 run
    <pool_version>/     the pool (sanntis.negpool.pool)

Every stage is resumable: finished units have done markers, IPS6 runs with ``-resume``.
"""

from __future__ import annotations

import json
import logging
import re
import shlex
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from sanntis.annotate.genecall import gene_finder
from sanntis.annotate.ips6 import annotate_proteins
from sanntis.config import NextflowOptions, load_recipe
from sanntis.io.fasta import iter_fasta, read_nucleotide_fasta, write_protein_fasta
from sanntis.io.genbank import cds_rows, iter_cds, read_records
from sanntis.io.ips_tsv import read_ips_tsv, signature_map_from_entries, tokens_by_key
from sanntis.negpool import sample as S
from sanntis.negpool import strip
from sanntis.negpool.download import download_assembly
from sanntis.negpool.pool import Pool, coverage_of, write_pool
from sanntis.runner.tasks import RunnerOptions, TaskError, Unit, dispatch, write_unit_script
from sanntis.train.ncbi import LineageCache
from sanntis.utils import cache_dir, write_json

log = logging.getLogger(__name__)

ANTISMASH_VERSION = "8.0.1"
ANTISMASH_IMAGE = "quay.io/nf-core/antismash:8.0.1--pyhdfd78af_0"  # HMM data hmmpress-ed in the image
SINGLE_MODE_MIN_BP = 100_000  # pyrodigal single mode needs enough sequence to train
GENOME_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")  # ids become directory names and antiSMASH script arguments


class BuildError(RuntimeError):
    pass


@dataclass
class BuildOptions:
    out: Path
    pool_version: str = "v1"
    genomes: Path | None = None  # pre-sampled genome table (skips sampling)
    gtdb_release: str = "226"
    gtdb_n: int = 205
    seed: int = 42
    refseq_groups: tuple[str, ...] = ("fungi", "protozoa", "plants")
    antismash_databases: Path | None = None
    antismash_image: str | None = None  # inline runner + container engine
    ips_tsv: tuple[Path, ...] = ()  # precomputed md5-keyed IPS TSVs (skip the IPS6 run)
    entries_json: Path | None = None  # InterPro entries.json for the signature map (default: IPS6 data dir)
    runner: str = "nextflow"
    workers: int = 1
    dry_run: bool = False
    nextflow: NextflowOptions = field(default_factory=NextflowOptions)


# ── sampling ─────────────────────────────────────────────────────────────────


def sample_genomes(o: BuildOptions) -> pd.DataFrame:
    proka = S.sample_gtdb(S.gtdb_tables(o.gtdb_release), n_samples=o.gtdb_n, seed=o.seed)
    root = cache_dir("refseq") / datetime.now(UTC).strftime("%Y-%m-%d")
    summaries = {g: S.fetch(S.REFSEQ_SUMMARY.format(group=S.REFSEQ_GROUPS[g][1]), root / f"assembly_summary_{g}.txt")
                 for g in o.refseq_groups}
    taxids = set()
    for p in summaries.values():
        taxids |= {int(t) for t in S.load_assembly_summary(p)["taxid"].dropna() if str(t).isdigit()}
    euk = S.sample_refseq(summaries, LineageCache().resolve(taxids), seed=0)
    return pd.concat([proka, euk], ignore_index=True)[S.GENOME_COLUMNS]


# ── units ────────────────────────────────────────────────────────────────────


def _genome_row(out: Path, genome: str) -> dict:
    g = pd.read_csv(out / "genomes.tsv", sep="\t", dtype=str, keep_default_na=False)
    rows = g[g["genome"] == genome]
    if rows.empty:
        raise BuildError(f"unknown genome {genome}")
    return rows.iloc[0].to_dict()


def _call_genes(fna: Path, genome: str) -> list:
    """pyrodigal: single mode when the genome is long enough to train, else meta. Ids ``<contig>_<n>``."""
    from Bio.Seq import Seq
    from Bio.SeqFeature import FeatureLocation, SeqFeature
    from Bio.SeqRecord import SeqRecord


    recs = read_nucleotide_fasta(fna)
    total = sum(len(r.seq) for r in recs)
    gf = gene_finder(recs, "meta" if total < SINGLE_MODE_MIN_BP else "single")
    out = []
    for r in recs:
        rec = SeqRecord(Seq(str(r.seq).upper()), id=r.id, name=r.id[:16], description=genome,
                        annotations={"molecule_type": "DNA"})
        for i, g in enumerate(gf.find_genes(bytes(r.seq)), start=1):
            cid = f"{r.id}_{i}"
            rec.features.append(SeqFeature(FeatureLocation(g.begin - 1, g.end, strand=g.strand), type="CDS",
                                           qualifiers={"locus_tag": [cid], "translation": [g.translate().rstrip("*")],
                                                       "transl_table": [str(g.translation_table)]}))
        out.append(rec)
    return out


def prepare_unit(out: Path, genome: str) -> None:
    """Download (unless a local path is given) and build the CDS table + antiSMASH input of one genome."""
    from Bio import SeqIO


    out = Path(out)
    row = _genome_row(out, genome)
    gdir = out / "genomes" / genome
    gdir.mkdir(parents=True, exist_ok=True)
    kind = "fna" if row["kingdom"] == "proka" else "gbff"
    src = Path(row["path"]) if row.get("path") else download_assembly(row["accession"], gdir, kind, row.get("url", ""))
    if row["kingdom"] == "proka" and not str(src).endswith((".gbk", ".gbk.gz", ".gbff", ".gbff.gz")):
        recs = _call_genes(src, genome)
    else:
        recs = [r for r in read_records(src) if any(f.type == "CDS" for f in r.features)]
    pdir = out / "prepared" / genome
    pdir.mkdir(parents=True, exist_ok=True)
    rows, seqs = [], {}
    for rec in recs:
        cds = list(iter_cds(rec))
        rows += [{"genome": genome, **r} for r in cds_rows(cds)]
        seqs.update((c.md5, c.sequence) for c in cds)
    if not rows:
        raise BuildError(f"{genome}: no CDS")
    with (pdir / "input.gbk").open("w") as fh:
        SeqIO.write(recs, fh, "genbank")
    write_protein_fasta(sorted(seqs.items()), pdir / "proteins.faa")
    pd.DataFrame(rows).to_parquet(pdir / "cds.parquet", index=False)


def antismash_script(out: Path, genome: str, taxon: str, databases: Path | None) -> str:
    d = out / "antismash" / genome
    q = shlex.quote
    db = f" --databases {q(str(databases))}" if databases else ""
    return f"""d={q(str(d))}
rm -rf "$d/res"; mkdir -p "$d"
antismash --taxon {q(taxon)} -c "${{SANNTIS_CPUS:-4}}" --minimal --genefinding-tool none --allow-long-headers{db} \\
    --output-dir "$d/res" --output-basename {q(genome)} {q(str(out / "prepared" / genome / "input.gbk"))} \\
    > "$d/antismash.log" 2>&1
antismash --version > "$d/version.txt"
gzip -c "$d/res/{genome}.gbk" > "$d/{genome}.antismash.gbk.gz.tmp"
mv "$d/{genome}.antismash.gbk.gz.tmp" "$d/{genome}.antismash.gbk.gz"
rm -rf "$d/res"
"""


def check_host_antismash() -> str:
    exe = shutil.which("antismash")
    if exe is None:
        raise BuildError("antismash not on PATH: pass --container-engine to use the pinned antiSMASH "
                         f"{ANTISMASH_VERSION} container, or install it")
    v = subprocess.run([exe, "--version"], capture_output=True, text=True, check=False).stdout.strip()
    if ANTISMASH_VERSION not in v:
        raise BuildError(f"host antiSMASH is '{v}', the pool needs {ANTISMASH_VERSION}")
    return v


def container_prefix(engine: str | None, image: str | None, binds: list[Path]) -> list[str]:
    """Inline runner: wrap the antiSMASH script in the pinned container."""
    if not engine:
        return []
    image = image or ANTISMASH_IMAGE
    if engine in ("singularity", "apptainer"):
        img = image if "://" in image or Path(image).exists() else f"docker://{image}"
        return [engine, "exec", *[x for b in binds for x in ("-B", str(b))], img]
    return [engine, "run", "--rm", "-u", "1000", *[x for b in binds for x in ("-v", f"{b}:{b}")], image]


# ── joining ──────────────────────────────────────────────────────────────────


def antismash_regions(path: Path) -> dict[str, list[tuple[int, int]]]:
    """antiSMASH output GenBank → contig → region intervals (0-based half-open)."""
    out: dict[str, list[tuple[int, int]]] = {}
    for rec in read_records(path):
        for f in rec.features:
            if f.type in ("region", "protocluster"):
                out.setdefault(rec.id, []).append((int(f.location.start), int(f.location.end)))
    return out


def pool_tables(out: Path, genomes: pd.DataFrame) -> dict[str, pd.DataFrame]:
    cds_parts, runs, mask = [], [], []
    for g in genomes["genome"]:
        cds = pd.read_parquet(out / "prepared" / g / "cds.parquet")
        reg = antismash_regions(out / "antismash" / g / f"{g}.antismash.gbk.gz")
        r, m = strip.clean_runs(cds, reg)
        runs += r
        mask += m
        cds_parts.append(cds)
    runs_df = pd.DataFrame(runs, columns=["genome", "contig", "run_id", "cds_idx", "n_cds"])
    return {"cds": pd.concat(cds_parts, ignore_index=True), "clean_runs.parquet": runs_df,
            "native_mask.parquet": pd.DataFrame(mask, columns=["genome", "contig", "cds_idx"]),
            "test_backbones.parquet": strip.test_backbones(runs_df, genomes)}


def unique_proteins(out: Path, genomes: pd.DataFrame, dest: Path) -> int:
    seqs: dict[str, str] = {}
    for g in genomes["genome"]:
        seqs.update(iter_fasta(out / "prepared" / g / "proteins.faa"))
    write_protein_fasta(sorted(seqs.items()), dest)
    return len(seqs)


def md5_tokens(tsvs: list[Path], signature_map: pd.DataFrame) -> tuple[dict, dict]:
    """md5 → tokens over md5-keyed IPS TSVs (one file at a time, so large runs stay in memory). ``meta["app_counts"]``
    holds the hits per (md5, app)."""
    tokens: dict[str, list[str]] = {}
    meta: dict = {"versions": set(), "rows": 0}
    counts = []
    for t in tsvs:
        h = read_ips_tsv(t)
        meta["versions"].add(h.version)
        hits = h.hits
        meta["rows"] += len(hits)
        tokens.update(tokens_by_key(hits, signature_map, key="md5"))
        counts.append(hits.groupby(["md5", "app"]).size())
    meta["versions"] = sorted(meta["versions"])
    meta["app_counts"] = (pd.concat(counts).groupby(level=["md5", "app"]).sum() if counts
                          else pd.Series(dtype=int, index=pd.MultiIndex.from_tuples([], names=["md5", "app"])))
    return tokens, meta


def build_negatives(o: BuildOptions, *, annotate=None) -> Pool | None:
    """Build a pool version. ``annotate(faa, outdir, nextflow_opts) -> tsv`` defaults to the IPS6 run."""
    out = Path(o.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "build.json", {k: str(v) if isinstance(v, Path) else v for k, v in asdict(o).items()})
    gt = out / "genomes.tsv"
    if not gt.exists():
        g = S.read_genomes(o.genomes) if o.genomes else sample_genomes(o)
        g.to_csv(gt, sep="\t", index=False)
    genomes = pd.read_csv(gt, sep="\t", dtype=str, keep_default_na=False)
    if bad := [g for g in genomes["genome"] if not GENOME_ID_RE.match(g)]:
        raise BuildError(f"genome ids are used as file names and in shell scripts: letters, digits and _ . - only "
                         f"(bad: {bad[:5]})")
    ropts = RunnerOptions(runner=o.runner, workers=o.workers, dry_run=o.dry_run, nextflow=o.nextflow)
    dispatch(out, "negpool_prepare", [Unit(g, "cpu_small") for g in genomes["genome"]], ropts)

    if not o.nextflow.container_engine and not o.dry_run:
        log.info("host antiSMASH: %s", check_host_antismash())
    for g, taxon in zip(genomes["genome"], genomes["taxon"], strict=True):
        write_unit_script(out, "negpool_antismash", g, antismash_script(out, g, taxon, o.antismash_databases))
    binds = [Path(o.antismash_databases)] if o.antismash_databases else []
    ropts.binds = [str(b) for b in binds]
    ropts.container_prefix = container_prefix(o.nextflow.container_engine, o.antismash_image, [out, *binds])
    dispatch(out, "negpool_antismash", [Unit(g, "antismash") for g in genomes["genome"]], ropts)
    if o.dry_run:
        return None

    tables = pool_tables(out, genomes)
    ipsdir = out / "ips6"
    faa = ipsdir / "proteins.faa"
    n_prot = unique_proteins(out, genomes, faa)
    rec = load_recipe().ips
    if o.ips_tsv:
        tsvs = [Path(t) for t in o.ips_tsv]
        ips_meta = {"source": "precomputed", "tsv": [str(t) for t in tsvs]}
    else:
        tsvs = [(annotate or annotate_proteins)(faa, ipsdir, o.nextflow)]
        ips_meta = {"source": "ips6 run"}
    entries = Path(o.entries_json) if o.entries_json else cache_dir("ips6") / "interpro" / rec.interpro / "entries.json"
    sigmap = signature_map_from_entries(entries)
    tokens, tmeta = md5_tokens(tsvs, sigmap)
    cds = tables.pop("cds")
    cds["tokens"] = [tokens.get(m, []) for m in cds["md5"]]
    cds["has_features"] = True
    ver = sorted({(out / "antismash" / g / "version.txt").read_text().strip() for g in genomes["genome"]})
    meta = {
        "pool_version": o.pool_version,
        "ips": {"version": rec.version, "interpro": rec.interpro, "apps": rec.app_names, "tsv_versions":
                tmeta["versions"], "signature_map": str(entries), **ips_meta},
        "antismash": {"version": ver, "options": "--minimal --genefinding-tool none --taxon bacteria|fungi"},
        "sources": {"gtdb_release": f"r{o.gtdb_release}", "seed": o.seed,
                    "refseq_groups": list(o.refseq_groups), "genomes_tsv": str(gt)},
        "gene_calling": {"proka": "pyrodigal (single mode; meta below 100 kb)", "euk": "RefSeq GBFF CDS"},
        "unique_proteins": n_prot,
        "coverage": coverage_of(cds, tmeta["app_counts"], dict(zip(genomes["genome"], genomes["kingdom"],
                                                                   strict=True))),
    }
    frames = {"genomes.parquet": genomes[["genome", "kingdom", "lineage", "source", "accession", "group"]],
              "cds_tokens.parquet": cds[["genome", "contig", "cds_idx", "cds_id", "locus_tag", "protein_id", "start",
                                         "end", "strand", "md5", "tokens", "has_features"]], **tables}
    pool = write_pool(out / o.pool_version, frames, meta)
    log.info("pool %s: %s", o.pool_version, json.dumps(pool.manifest["counts"]))
    return pool


def run_cli(**kw) -> None:
    nf = NextflowOptions(container_engine=kw.get("container_engine"), nf_config=list(kw.get("nf_config") or ()),
                         nf_profile=list(kw.get("nf_profile") or ()), nf_args=kw.get("nf_args") or "")
    o = BuildOptions(out=Path(kw["out"]), pool_version=kw.get("pool_version") or "v1",
                     genomes=Path(kw["genomes"]) if kw.get("genomes") else None,
                     antismash_databases=Path(kw["antismash_databases"]) if kw.get("antismash_databases") else None,
                     antismash_image=kw.get("antismash_image"),
                     ips_tsv=tuple(Path(p) for p in kw.get("ips_tsv") or ()),
                     entries_json=Path(kw["entries_json"]) if kw.get("entries_json") else None,
                     runner=kw.get("runner") or "nextflow", workers=kw.get("workers") or 1,
                     dry_run=bool(kw.get("dry_run")), nextflow=nf)
    try:
        pool = build_negatives(o)
    except TaskError as e:
        raise SystemExit(str(e)) from e
    if pool is not None:
        print(json.dumps(pool.manifest["counts"], indent=2))
