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

"""Command-line interface: predict, setup, example, train, build-negatives, task."""

from __future__ import annotations

import logging
from pathlib import Path

import click

from sanntis._version import __version__

log = logging.getLogger("sanntis")


def _nextflow_options(f):
    """Infrastructure pass-through, shared by every command that may run Nextflow."""
    opts = [
        click.option("--container-engine", type=click.Choice(["docker", "singularity", "apptainer", "podman"]),
                     default=None, help="Container engine for InterProScan 6 (its -profile) and antiSMASH."),
        click.option("--nf-config", "nf_config", multiple=True, type=click.Path(exists=True, dir_okay=False),
                     help="Nextflow config passed as -c to every nextflow run (repeatable)."),
        click.option("--nf-profile", "nf_profile", multiple=True,
                     help="Extra -profile defined by your --nf-config (repeatable)."),
        click.option("--nf-args", default="", help="Arguments appended verbatim to every nextflow run."),
    ]
    for o in reversed(opts):
        f = o(f)
    return f


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="sanntis")
@click.option("-v", "--verbose", count=True, help="More logging (-vv for debug).")
def main(verbose: int) -> None:
    """SanntiS: SMBGC Annotation using Neural Networks Trained on Interpro Signatures."""
    logging.basicConfig(level=logging.WARNING - 10 * min(verbose + 1, 2),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@main.command()
@click.argument("inputs", nargs=-1, type=click.Path(exists=True, dir_okay=False))
@click.option("--gff", type=click.Path(exists=True, dir_okay=False),
              help="GFF3 with the CDS of a .faa input (CDS ID = protein id).")
@click.option("--model", default=None, help="Hub id[@revision] or local bundle directory.")
@click.option("--outdir", "-o", type=click.Path(file_okay=False), default=".", show_default=True)
@click.option("--ips-tsv", type=click.Path(exists=True, dir_okay=False), help="Precomputed IPS 5.76+ / 6.x TSV.")
@click.option("--ips-release", default=None, help="InterPro release of --ips-tsv (e.g. 107.0).")
@click.option("--kingdom", type=click.Choice(["proka", "euk"]), default="proka", show_default=True)
@click.option("--genecall", type=click.Choice(["meta", "single"]), default="meta", show_default=True)
@click.option("--npc-precision", type=click.Choice(["0.9", "0.8"]), default="0.9", show_default=True)
@click.option("--threads", type=int, default=1, show_default=True)
@click.option("--allow-ips-mismatch", is_flag=True)
@click.option("--allow-incomplete-ips", is_flag=True)
@click.option("--write-faa", is_flag=True, help="Also write the CDS proteins (.cds.faa) and their GFF3 (.cds.gff3).")
@click.option("--ips-datadir", type=click.Path(file_okay=False), default=None,
              help="InterProScan 6 data directory (default: $SANNTIS_CACHE/ips6).")
@click.option("--keep-work", is_flag=True, help="Keep the Nextflow work directory of the IPS6 run.")
@_nextflow_options
def predict(**kw) -> None:
    """Predict BGCs in .fna / .gbk inputs, or a .faa with its --gff (optionally with a precomputed IPS TSV)."""
    from sanntis.frontend import run_cli

    run_cli(**kw)


@main.command()
@_nextflow_options
@click.option("--datadir", type=click.Path(file_okay=False), default=None,
              help="IPS6 data directory (default: $SANNTIS_CACHE/ips6).")
def setup(**kw) -> None:
    """Pre-fetch the InterProScan 6 pipeline and its pinned data."""
    from sanntis.annotate.ips6 import setup as ips_setup

    ips_setup(**kw)


@main.command()
@click.option("--run", "run_dir", type=click.Path(file_okay=False), required=True, help="Run directory.")
@click.option("--mibig", default=None,
              help="MIBiG release to download and train on (active entries), e.g. 4.0. Unset / 'none' / 'false': "
                   "external data only.")
@click.option("--external-gbk", multiple=True, type=click.Path(exists=True, dir_okay=False),
              help="External BGCs: MIBiG-format GBK archive (*_gbk*.tar.gz, one record per BGC; repeatable).")
@click.option("--external-json", multiple=True, type=click.Path(exists=True, dir_okay=False),
              help="MIBiG-format JSON archive (*_json*.tar.gz) of each --external-gbk, in the same order; needed for "
                   "the class and NPC heads (repeatable).")
@click.option("--negatives", default=None, help="Negative pool: Hub dataset revision or local directory.")
@click.option("--npc-cache", default=None,
              help="Published NPClassifier answers to seed $SANNTIS_CACHE/npclassifier with before labelling: Hub "
                   "dataset version (e.g. v1) or a directory of cache files.")
@click.option("--ips-tsv", "train_ips_tsv", multiple=True, type=click.Path(exists=True, dir_okay=False),
              help="Precomputed IPS TSV(s) (5.76+ / 6.x, md5 in col 2) of the collected proteins; skips IPS6.")
@click.option("--ips-datadir", type=click.Path(file_okay=False), default=None,
              help="IPS6 data directory (default: $SANNTIS_CACHE/ips6); also the source of the signature map.")
@click.option("--entries-json", type=click.Path(exists=True, dir_okay=False), default=None,
              help="InterPro entries.json for the signature map (default: <ips-datadir>/interpro/<rel>/entries.json).")
@click.option("--runner", type=click.Choice(["nextflow", "inline"]), default=None, help="[default: nextflow]")
@click.option("--workers", type=int, default=None, help="Process pool size (--runner inline).  [default: 1]")
@click.option("--gpu", is_flag=True, help="Train detectors with label 'gpu'.")
@click.option("--from", "from_step", default=None, help="Force re-running from this step.")
@click.option("--until", "until_step", default=None, hidden=True, help="Stop after this step.")
@click.option("--dry-run", is_flag=True, help="Print the nextflow command lines and exit.")
@click.option("--config", "config_file", type=click.Path(exists=True, dir_okay=False), help="run.yaml")
@_nextflow_options
def train(**kw) -> None:
    """Train a model bundle (collect → annotate → entities → cv → tune → finalize → report)."""
    from sanntis.train.orchestrate import run_cli

    run_cli(**kw)


@main.command("build-negatives")
@click.option("--out", type=click.Path(file_okay=False), required=True, help="Build directory.")
@click.option("--pool-version", default="v1", show_default=True, help="Name of the pool version written in --out.")
@click.option("--genomes", type=click.Path(exists=True, dir_okay=False),
              help="Genome table (TSV: genome, kingdom[, lineage, source, accession, path]); skips sampling.")
@click.option("--antismash-databases", type=click.Path(exists=True, file_okay=False),
              help="antiSMASH 8.0.1 databases directory (needed with the biocontainer image).")
@click.option("--antismash-image", default=None, help="antiSMASH image for --runner inline with --container-engine.")
@click.option("--ips-tsv", multiple=True, type=click.Path(exists=True, dir_okay=False),
              help="Precomputed md5-keyed IPS TSV(s) of the pool proteins (skips the IPS6 run).")
@click.option("--entries-json", type=click.Path(exists=True, dir_okay=False),
              help="InterPro entries.json for the signature map (default: the IPS6 data dir).")
@click.option("--runner", type=click.Choice(["nextflow", "inline"]), default="nextflow", show_default=True)
@click.option("--workers", type=int, default=1, show_default=True, help="Process pool size (--runner inline).")
@click.option("--dry-run", is_flag=True, help="Print the nextflow command lines and exit.")
@_nextflow_options
def build_negatives(**kw) -> None:
    """Rebuild the negative genome pool (GTDB / RefSeq sampling, antiSMASH stripping, IPS6)."""
    from sanntis.negpool.build import run_cli

    run_cli(**kw)


@main.command()
@click.option("--outdir", "-o", type=click.Path(file_okay=False), default="sanntis_example", show_default=True)
def example(outdir: str) -> None:
    """Copy the example input (a protein FASTA, its GFF3 and its InterProScan TSV) to OUTDIR."""
    import shutil

    from sanntis.data import EXAMPLE_DIR

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    for f in sorted(EXAMPLE_DIR.iterdir()):
        shutil.copy(f, out / f.name)
        click.echo(out / f.name)
    click.echo(f"\nsanntis predict {out}/ERZ29794225_slice.faa --gff {out}/ERZ29794225_slice.gff "
               f"--ips-tsv {out}/ERZ29794225_slice.ips.tsv -o {out}/out")


@main.command(hidden=True)
@click.argument("step")
@click.option("--run", "run_dir", type=click.Path(exists=True, file_okay=False), required=True)
@click.option("--unit", required=True)
def task(step: str, run_dir: str, unit: str) -> None:
    """Run one unit of a fan-out step (called by nf/tasks.nf or the inline runner)."""
    from sanntis.runner.tasks import run_unit

    run_unit(step, Path(run_dir), unit)


if __name__ == "__main__":
    main()
