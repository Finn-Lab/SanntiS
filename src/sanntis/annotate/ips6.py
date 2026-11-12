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

"""InterProScan 6 runner: md5-dedup, cache lookup, top-level ``nextflow run`` s for the missing proteins, cache fill.

The pinned parts of the command (pipeline revision, InterPro release, the 14 applications, TSV output) come from
``recipe.yaml`` and cannot be overridden; the infrastructure (container engine, ``-c`` configs, profiles, extra args)
is passed through. Run settings (environment, see :class:`Ips6Settings`):

- ``SANNTIS_IPS6_CPUS`` (4): IPS6 ``--cpus``, the cores of each multi-threaded IPS6 task (HMMER, Gene3D, SFLD, ...);
- ``SANNTIS_IPS6_CHUNK`` (20000): proteins per ``nextflow run``;
- ``SANNTIS_IPS6_PARALLEL`` (1): chunks run concurrently, each its own ``nextflow run`` and work dir;
- ``SANNTIS_IPS6_QUEUE_SIZE`` / ``SANNTIS_IPS6_SUBMIT_RATE``: Nextflow ``executor.queueSize`` / ``submitRateLimit``
  of each IPS6 run (so concurrent tasks <= parallel x queue size), set after the user's configs;
- ``SANNTIS_IPS6_LOOKUP`` (auto | off) and ``SANNTIS_IPS6_MATCHES_API``: the precalculated-match lookup (IPS6
  Matches API). ``auto`` uses it only when the API serves the pinned InterPro release (IPS6 itself does not check
  the release: it would mix the API's matches with local ones); otherwise every protein is computed locally.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

import pandas as pd

from sanntis.annotate.cache import HIT_COLUMNS, AnnotationCache
from sanntis.config import NextflowOptions, load_recipe
from sanntis.io.fasta import iter_fasta, write_protein_fasta
from sanntis.io.ips_tsv import normalise_name, read_ips_tsv
from sanntis.runner import nextflow as nf
from sanntis.utils import cache_dir

log = logging.getLogger(__name__)

# canonical app → IPS6 data directory under --datadir (coils and mobidblite ship inside the container)
DATA_DIRS = {"pfam": "pfam", "ncbifam": "ncbifam", "sfld": "sfld", "superfamily": "superfamily", "gene3d": "cath",
             "hamap": "hamap", "cdd": "cdd", "prints": "prints", "pirsf": "pirsf", "prositeprofiles": "prosite",
             "prositepatterns": "prosite", "smart": "smart"}
# Set by sanntis; a --nf-args that tries to change one is refused.
PINNED_ARGS = {"-r", "-revision", "--input", "--datadir", "--interpro", "--applications", "--skipApplications",
               "--noMatchesApi", "--matchesApiUrl", "--formats", "--outdir", "--outprefix", "--nucleic", "--runMl",
               "--skipInterproVersionCheck", "--cpus"}
MATCHES_API = "https://www.ebi.ac.uk/interpro/matches/api"
# First run with no data: PROSITE profiles and patterns share one data dir and the patterns scan can start before
# the shared download has finished (IPS6 6.0.3.0). Setup therefore runs once without it, then with every app.
SETUP_SKIP_FIRST = ("prositepatterns",)


class Ips6Error(RuntimeError):
    pass


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name, "").strip()
    if not v:
        return default
    try:
        n = int(v)
    except ValueError as e:
        raise Ips6Error(f"{name}={v!r}: expected an integer") from e
    if n < 1:
        raise Ips6Error(f"{name}={v!r}: must be >= 1")
    return n


@dataclass(frozen=True)
class Ips6Settings:
    """How IPS6 runs (not what it computes): read from the environment at each call."""

    cpus: int = 4
    chunk: int = 20000
    parallel: int = 1
    queue_size: int | None = None
    submit_rate: str | None = None
    lookup: str = "auto"  # auto | off
    matches_api: str = MATCHES_API

    @classmethod
    def from_env(cls) -> Ips6Settings:
        lookup = os.environ.get("SANNTIS_IPS6_LOOKUP", "auto").strip().lower() or "auto"
        if lookup not in ("auto", "off"):
            raise Ips6Error(f"SANNTIS_IPS6_LOOKUP={lookup!r}: expected auto or off")
        q = os.environ.get("SANNTIS_IPS6_QUEUE_SIZE", "").strip()
        return cls(cpus=_env_int("SANNTIS_IPS6_CPUS", 4), chunk=_env_int("SANNTIS_IPS6_CHUNK", 20000),
                   parallel=_env_int("SANNTIS_IPS6_PARALLEL", 1),
                   queue_size=_env_int("SANNTIS_IPS6_QUEUE_SIZE", 1) if q else None,
                   submit_rate=os.environ.get("SANNTIS_IPS6_SUBMIT_RATE", "").strip() or None, lookup=lookup,
                   matches_api=os.environ.get("SANNTIS_IPS6_MATCHES_API", "").strip() or MATCHES_API)


def matches_api_release(url: str, timeout: int = 30) -> str | None:
    """InterPro release the Matches API at ``url`` serves (its ``/info``), None when it cannot be read."""
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/info", timeout=timeout) as r:
            return str(json.load(r).get("release") or "") or None
    except Exception as e:  # noqa: BLE001 - any failure means: no lookup
        log.warning("Matches API %s: could not read /info (%s)", url, e)
        return None


@dataclass
class Lookup:
    used: bool
    url: str
    api_release: str | None = None
    reason: str = ""

    def as_dict(self) -> dict:
        return {"used": self.used, "url": self.url, "api_release": self.api_release, "reason": self.reason}


def resolve_lookup(settings: Ips6Settings) -> Lookup:
    """Use the Matches API only when it serves the pinned release (IPS6 6.0.3.0 checks the API version only)."""
    pinned = load_recipe().ips.interpro
    if settings.lookup == "off":
        return Lookup(False, settings.matches_api, reason="SANNTIS_IPS6_LOOKUP=off")
    rel = matches_api_release(settings.matches_api)
    if rel == pinned:
        return Lookup(True, settings.matches_api, rel, f"API serves InterPro {rel}")
    reason = (f"API serves InterPro {rel}, the pin is {pinned}: every protein computed locally" if rel else
              "API unavailable: every protein computed locally")
    log.warning("InterProScan 6 lookup off: %s", reason)
    return Lookup(False, settings.matches_api, rel, reason)


def default_datadir() -> Path:
    return cache_dir("ips6")


def ips6_app_names(apps: list[str] | None = None) -> list[str]:
    """``--applications`` values (``CATH-Gene3D`` → ``cathgene3d``), recipe order."""
    r = load_recipe().ips
    keep = set(apps or r.app_names)
    return [normalise_name(a.ips6) for a in r.apps if a.canonical in keep]


def missing_data(datadir: Path) -> list[str]:
    """What a run would still have to download: ``interpro/<release>/databases.json`` and the member-DB dirs."""
    r = load_recipe().ips
    dbj = Path(datadir) / "interpro" / r.interpro / "databases.json"
    if not dbj.is_file():
        return [str(dbj.relative_to(datadir))]
    versions = {normalise_name(k): v for k, v in json.loads(dbj.read_text()).items()}
    out = []
    for a in r.apps:
        d = DATA_DIRS.get(a.canonical)
        if d is None:
            continue
        v = versions.get(normalise_name(a.ips6))
        p = Path(datadir) / d / v if v else Path(datadir) / d
        if not p.is_dir() or not any(p.iterdir()):
            out.append(f"{a.canonical} ({p.relative_to(datadir)})")
    return out


def check_nf_args(opts: NextflowOptions) -> None:
    toks = shlex.split(opts.nf_args or "")
    bad = sorted({t.split("=")[0] for t in toks} & PINNED_ARGS)
    if bad:
        raise Ips6Error(f"--nf-args may not set {bad}: the InterProScan version, InterPro release and applications "
                        "are pinned by sanntis; cores and the Matches API lookup are set with SANNTIS_IPS6_CPUS / "
                        "SANNTIS_IPS6_LOOKUP")


def nextflow_run(faa: Path, outdir: Path, datadir: Path, run_dir: Path, apps: list[str] | None = None,
                 prefix: str = "proteins", *, cpus: int = 4, lookup: Lookup | None = None) -> nf.NextflowRun:
    r = load_recipe().ips
    params = {"input": str(Path(faa).resolve()), "datadir": str(Path(datadir).resolve()), "interpro": r.interpro,
              "applications": ",".join(ips6_app_names(apps)), "formats": "tsv",
              "outdir": str(Path(outdir).resolve()), "outprefix": prefix, "cpus": cpus,
              # 6.0.3.0 + 107.0 is a known-compatible pin; the online check is a network call at start-up that
              # fails transiently ("could not retrieve compatibility information")
              "skipInterproVersionCheck": True}
    if lookup is not None and lookup.used:
        params["matchesApiUrl"] = lookup.url
    else:
        params["noMatchesApi"] = True
    return nf.NextflowRun(r.pipeline, params=params, revision=r.version, work_dir=run_dir / "work",
                          log_file=run_dir / "nextflow.log")


def executor_config(run_dir: Path, settings: Ips6Settings) -> Path | None:
    """``executor`` overrides of the IPS6 runs (queue size, submit rate), or None when none is set."""
    lines = []
    if settings.queue_size:
        lines.append(f"    queueSize = {settings.queue_size}")
    if settings.submit_rate:
        lines.append(f"    submitRateLimit = '{settings.submit_rate}'")
    if not lines:
        return None
    p = Path(run_dir) / "ips6_executor.config"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("// written by sanntis (SANNTIS_IPS6_QUEUE_SIZE / SANNTIS_IPS6_SUBMIT_RATE)\nexecutor {\n"
                 + "\n".join(lines) + "\n}\n")
    return p


@dataclass
class RunPlan:
    """Settings + lookup decision shared by every IPS6 run of one annotation."""

    settings: Ips6Settings = field(default_factory=Ips6Settings)
    lookup: Lookup | None = None
    executor_config: Path | None = None

    def options(self, opts: NextflowOptions) -> NextflowOptions:
        if self.executor_config is None:
            return opts
        return opts.model_copy(update={"nf_config": [*opts.nf_config, self.executor_config]})  # last -c wins


def run_plan(run_dir: Path, settings: Ips6Settings | None = None, *, lookup: bool = True) -> RunPlan:
    settings = settings or Ips6Settings.from_env()
    lk = resolve_lookup(settings) if lookup else Lookup(False, settings.matches_api, reason="setup")
    return RunPlan(settings, lk, executor_config(run_dir, settings))


def run_ips6(faa: Path, run_dir: Path, opts: NextflowOptions, datadir: Path, apps: list[str] | None = None,
             keep_work: bool = False, plan: RunPlan | None = None) -> Path:
    """One top-level IPS6 run in ``run_dir`` → its TSV."""
    check_nf_args(opts)
    plan = plan or RunPlan(lookup=Lookup(False, MATCHES_API, reason="default"))
    run_dir = Path(run_dir).resolve()
    out = run_dir / "results"
    out.mkdir(parents=True, exist_ok=True)
    nf.run(nextflow_run(faa, out, datadir, run_dir, apps, cpus=plan.settings.cpus, lookup=plan.lookup),
           plan.options(opts), cwd=run_dir)
    tsv = out / "proteins.tsv"
    if not tsv.is_file():
        raise Ips6Error(f"InterProScan 6 finished without {tsv}")
    if not keep_work:
        shutil.rmtree(run_dir / "work", ignore_errors=True)
    return tsv


# Nextflow's FASTA splitter caches each split next to its input; IPS6 splits every batch file several times
# concurrently, and on large inputs (many batches) a reader occasionally deserialises a cache file another thread is
# still writing ("KryoException: Buffer underflow", then "Join mismatch"). Seen on 748k proteins (150 batches),
# not on 5k-20k. We therefore send at most SANNTIS_IPS6_CHUNK proteins per IPS6 run and retry a run from scratch on
# that signature.
RACE_SIGNATURES = ("KryoException", "Join mismatch")
RETRIES = 2


def run_ips6_retry(faa: Path, run_dir: Path, opts: NextflowOptions, datadir: Path, apps: list[str] | None = None,
                   keep_work: bool = False, plan: RunPlan | None = None) -> Path:
    run_dir = Path(run_dir).resolve()
    for attempt in range(RETRIES + 1):
        try:
            return run_ips6(faa, run_dir, opts, datadir, apps, keep_work, plan)
        except nf.NextflowError:
            logf = run_dir / "nextflow.log"
            text = logf.read_text(errors="replace") if logf.exists() else ""
            if attempt == RETRIES or not any(sig in text for sig in RACE_SIGNATURES):
                raise
            log.warning("IPS6 run hit the Nextflow splitter-cache race; retrying from scratch (%d/%d)",
                        attempt + 1, RETRIES)
            for sub in ("work", ".nextflow", "results"):
                shutil.rmtree(run_dir / sub, ignore_errors=True)
    raise AssertionError("unreachable")


def run_ips6_chunked(faa: Path, run_dir: Path, opts: NextflowOptions, datadir: Path, apps: list[str] | None = None,
                     keep_work: bool = False, chunk: int | None = None, plan: RunPlan | None = None) -> Path:
    """IPS6 on ``faa`` in runs of at most ``chunk`` proteins, ``plan.settings.parallel`` at a time (finished chunks
    are kept on re-runs) → one concatenated TSV in chunk order. ``run_dir/lookup.json`` records the lookup decision.
    """
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    plan = plan or run_plan(run_dir)
    chunk = chunk or plan.settings.chunk
    (run_dir / "lookup.json").write_text(json.dumps(plan.lookup.as_dict() if plan.lookup else None, indent=1))
    records = list(iter_fasta(faa))
    if len(records) <= chunk:
        return run_ips6_retry(faa, run_dir, opts, datadir, apps, keep_work, plan)
    out = run_dir / "results" / "proteins.tsv"
    if out.is_file():
        return out
    dirs = [run_dir / f"chunk{i // chunk:04d}" for i in range(0, len(records), chunk)]

    def one(k: int) -> Path:
        d = dirs[k]
        done = d / "results" / "proteins.tsv"
        if not done.is_file():
            d.mkdir(parents=True, exist_ok=True)
            write_protein_fasta(records[k * chunk : (k + 1) * chunk], d / "proteins.faa")
            run_ips6_retry(d / "proteins.faa", d, opts, datadir, apps, keep_work, plan)
        return done

    log.info("InterProScan 6: %d proteins in %d chunks of <= %d, %d at a time", len(records), len(dirs), chunk,
             plan.settings.parallel)
    errors = []
    with ThreadPoolExecutor(max_workers=plan.settings.parallel) as ex:
        for k, f in enumerate([ex.submit(one, k) for k in range(len(dirs))]):
            try:
                f.result()
            except Exception as e:  # noqa: BLE001 - let the other chunks finish, then fail
                errors.append(f"{dirs[k].name}: {e}")
    if errors:
        raise Ips6Error(f"{len(errors)} of {len(dirs)} InterProScan 6 chunks failed (finished chunks are kept): "
                        + "; ".join(errors[:3]))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    with open(tmp, "w") as fh:
        for d in dirs:
            fh.write((d / "results" / "proteins.tsv").read_text())
    tmp.replace(out)
    return out


def setup(datadir: str | Path | None = None, container_engine: str | None = None, nf_config=(), nf_profile=(),
          nf_args: str = "", run_dir: Path | None = None, opts: NextflowOptions | None = None) -> Path:
    """Fetch the pinned pipeline and data into ``datadir`` (two runs on a bundled protein, see SETUP_SKIP_FIRST)."""
    datadir = Path(datadir or default_datadir()).expanduser().resolve()
    opts = opts or NextflowOptions(container_engine=container_engine, nf_config=list(nf_config),
                                   nf_profile=list(nf_profile), nf_args=nf_args)
    datadir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(run_dir or datadir / ".setup")
    faa = run_dir / "setup.faa"
    run_dir.mkdir(parents=True, exist_ok=True)
    faa.write_text(resources.files("sanntis").joinpath("data/setup.faa").read_text())
    r = load_recipe().ips
    first = [a for a in r.app_names if a not in SETUP_SKIP_FIRST]
    log.info("InterProScan 6 setup (%s, InterPro %s) into %s", r.version, r.interpro, datadir)
    plan = run_plan(run_dir, lookup=False)  # setup must exercise the local data
    run_ips6(faa, run_dir / "pass1", opts, datadir, apps=first, plan=plan)
    run_ips6(faa, run_dir / "pass2", opts, datadir, plan=plan)
    missing = missing_data(datadir)
    if missing:
        raise Ips6Error(f"InterProScan 6 setup finished but data are still missing in {datadir}: {missing}")
    shutil.rmtree(run_dir, ignore_errors=True)
    log.info("InterProScan 6 data ready in %s", datadir)
    return datadir


def ensure_data(opts: NextflowOptions, datadir: Path, run_dir: Path) -> bool:
    """Run :func:`setup` when data are missing. Returns True when setup ran."""
    missing = missing_data(datadir)
    if not missing:
        return False
    log.warning("InterProScan 6 data missing in %s (%s): running setup first", datadir, ", ".join(missing[:4]))
    setup(datadir, run_dir=run_dir, opts=opts)
    return True


@dataclass
class Annotation:
    hits: pd.DataFrame  # HIT_COLUMNS, keyed by md5
    n_proteins: int
    n_cached: int
    n_sent: int
    setup_ran: bool = False
    lookup: dict | None = None  # Lookup.as_dict() of the run, None when nothing was sent


def annotate(proteins: dict[str, str], opts: NextflowOptions, run_dir: Path, *, datadir: Path | None = None,
             cache: AnnotationCache | None = None, keep_work: bool = False) -> Annotation:
    """``proteins``: md5 → cleaned sequence (already deduplicated). Missing proteins go to one IPS6 run."""
    cache = cache or AnnotationCache()
    datadir = Path(datadir or default_datadir()).expanduser().resolve()
    cached, missing = cache.lookup(proteins)
    log.info("annotation: %d unique proteins, %d cached, %d to annotate", len(proteins), len(cached), len(missing))
    setup_ran = False
    lookup = None
    if missing:
        check_nf_args(opts)
        run_dir = Path(run_dir).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        setup_ran = ensure_data(opts, datadir, run_dir / "setup")
        faa = run_dir / "proteins.faa"
        write_protein_fasta(((m, proteins[m]) for m in sorted(missing)), faa)
        plan = run_plan(run_dir / "ips6")
        lookup = plan.lookup.as_dict()
        tsv = run_ips6_chunked(faa, run_dir / "ips6", opts, datadir, keep_work=keep_work, plan=plan)
        h = read_ips_tsv(tsv)
        if h.version not in ("6", "unknown"):
            raise Ips6Error(f"{tsv}: expected InterProScan 6 output, detected version {h.version}")
        hits = h.hits.assign(md5=h.hits["protein_id"].str.lower())
        stray = set(hits["md5"]) - missing
        if stray:
            raise Ips6Error(f"{tsv}: {len(stray)} protein ids that were not submitted")
        cache.add(missing, hits[HIT_COLUMNS])
    return Annotation(hits=cache.hits(proteins), n_proteins=len(proteins), n_cached=len(cached),
                      n_sent=len(missing), setup_ran=setup_ran, lookup=lookup)


def annotate_proteins(faa: Path, outdir: Path, opts: NextflowOptions, *, datadir: Path | None = None,
                      keep_work: bool = False) -> Path:
    """Run IPS6 on ``faa`` (headers = md5), without the cache; returns the md5-keyed TSV (negative pool builds)."""
    outdir = Path(outdir).resolve()
    datadir = Path(datadir or default_datadir()).expanduser().resolve()
    check_nf_args(opts)
    ensure_data(opts, datadir, outdir / "setup")
    return run_ips6_chunked(Path(faa), outdir / "run", opts, datadir, keep_work=keep_work)
