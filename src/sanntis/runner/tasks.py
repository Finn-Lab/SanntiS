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

"""Fan-out units: task manifest, step → unit registry, ``sanntis task`` entry point, dispatch.

A step that fans out writes ``<run>/<step>/tasks.tsv`` (``unit_id``, ``label``) and dispatches it with one of two
runners that execute the *same* unit function:

- ``nextflow``: the bundled ``nf/tasks.nf`` runs one process per row; each executes
  ``<python> -m sanntis task <step> --run <run> --unit <unit_id>`` (``<python>`` = the launcher's interpreter).
  Rows with label ``antismash`` run the unit's pre-rendered shell script ``<run>/<step>/units/<unit_id>.sh``
  instead (a third-party tool, optionally in its container; see :func:`write_unit_script`).
- ``inline``: a local process pool (:mod:`sanntis.runner.inline`).

Units read and write ``<run>/`` directly; a unit is done when ``<run>/<step>/units/<unit_id>.ok`` exists.
"""

from __future__ import annotations

import csv
import importlib
import logging
import os
import shlex
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from sanntis.config import NextflowOptions
from sanntis.runner import nextflow as nf
from sanntis.utils import available_cpus, configure_threads

log = logging.getLogger(__name__)

LABELS = ("cpu_small", "train", "gpu", "antismash")
SCRIPT_LABELS = ("antismash",)  # labels whose unit is a shell script, not a Python call
MANIFEST = "tasks.tsv"

UnitFn = Callable[[Path, str], None]

# step → "module:function" (imported lazily, so `sanntis task` stays cheap)
REGISTRY: dict[str, str] = {
    "negpool_prepare": "sanntis.negpool.build:prepare_unit",
    "cv": "sanntis.train.steps:cv_unit",
    "finalize": "sanntis.train.steps:finalize_unit",
}


class TaskError(RuntimeError):
    pass


# Tests only: one extra ``step=module:function`` (an environment variable, so that it reaches spawned workers and
# ``sanntis task`` under Nextflow).
TEST_TASK_ENV = "_SANNTIS_TEST_TASK"


def resolve(step: str) -> UnitFn:
    registry = dict(REGISTRY)
    if extra := os.environ.get(TEST_TASK_ENV):
        name, _, target = extra.partition("=")
        registry[name] = target
    if step not in registry:
        raise TaskError(f"unknown task step {step!r} (known: {sorted(registry)})")
    mod, _, name = registry[step].partition(":")
    return getattr(importlib.import_module(mod), name)


@dataclass(frozen=True)
class Unit:
    unit_id: str
    label: str


def step_dir(run_dir: Path, step: str) -> Path:
    return Path(run_dir) / step


def unit_dir(run_dir: Path, step: str) -> Path:
    return step_dir(run_dir, step) / "units"


def done_marker(run_dir: Path, step: str, unit_id: str) -> Path:
    return unit_dir(run_dir, step) / f"{unit_id}.ok"


def write_manifest(run_dir: Path, step: str, rows: Sequence[Unit]) -> Path:
    seen: set[str] = set()
    for u in rows:
        if u.label not in LABELS:
            raise TaskError(f"unit {u.unit_id}: unknown label {u.label!r} (one of {LABELS})")
        if not u.unit_id or any(c in u.unit_id for c in "\t\n/ ") or u.unit_id in seen:
            raise TaskError(f"invalid or duplicate unit id {u.unit_id!r}")
        seen.add(u.unit_id)
    path = step_dir(run_dir, step) / MANIFEST
    path.parent.mkdir(parents=True, exist_ok=True)
    unit_dir(run_dir, step).mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["unit_id", "label"])
        w.writerows((u.unit_id, u.label) for u in rows)
    tmp.replace(path)
    return path


def write_unit_script(run_dir: Path, step: str, unit_id: str, body: str) -> Path:
    """Shell script executed for a script-label unit (in the tool's container when an engine is set).

    It must be idempotent and write ``<unit>.ok`` itself on success (the generated footer does that).
    """
    p = unit_dir(run_dir, step) / f"{unit_id}.sh"
    p.parent.mkdir(parents=True, exist_ok=True)
    ok = done_marker(run_dir, step, unit_id)
    p.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + body.rstrip() + f"\ntouch {shlex.quote(str(ok))}\n")
    p.chmod(0o755)
    return p


def run_unit(step: str, run_dir: Path, unit: str, *, force: bool = False) -> None:
    """Entry point of ``sanntis task <step> --run <run> --unit <id>``."""
    run_dir = Path(run_dir).resolve()
    marker = done_marker(run_dir, step, unit)
    if marker.exists() and not force:
        log.info("%s/%s already done", step, unit)
        return
    fn = resolve(step)

    log.info("unit %s/%s (%d threads)", step, unit, configure_threads())
    fn(run_dir, unit)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()


def pending(run_dir: Path, step: str, units: Sequence[Unit]) -> list[Unit]:
    return [u for u in units if not done_marker(run_dir, step, u.unit_id).exists()]


# ── dispatch ─────────────────────────────────────────────────────────────────


@dataclass
class RunnerOptions:
    runner: str = "nextflow"  # nextflow | inline
    workers: int = 1
    dry_run: bool = False
    nextflow: NextflowOptions | None = None
    container_prefix: Sequence[str] = ()  # inline runner: wrapper for script-label units (e.g. singularity exec)
    binds: Sequence[str] = ()  # extra host paths a containerised unit reads (e.g. the antiSMASH databases)


def nf_dir() -> Path:
    return Path(str(resources.files("sanntis").joinpath("nf")))


def local_limits_config(path: Path) -> Path:
    """Cap label requests to this machine (local executor only: written when the user passes no --nf-config)."""
    cpus = available_cpus()
    try:
        mem_gb = max(1, os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // 2**30)
    except (ValueError, OSError, AttributeError):
        mem_gb = 4
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"process.resourceLimits = [cpus: {cpus}, memory: '{mem_gb} GB']\n")
    return path


def nextflow_run(run_dir: Path, step: str, opts: RunnerOptions | None = None):
    """The ``nextflow run nf/tasks.nf`` of one fan-out step (work dir ``<run>/<step>/work``, ``-resume``)."""
    opts = opts or RunnerOptions()
    run_dir = Path(run_dir).resolve()
    sd = step_dir(run_dir, step)
    params: dict[str, object] = {"manifest": sd / MANIFEST, "run": run_dir, "step": step, "python": sys.executable}
    if os.environ.get("PYTHONPATH"):
        params["pythonpath"] = os.environ["PYTHONPATH"]
    if opts.binds:
        params["binds"] = ",".join(str(Path(b).resolve()) for b in opts.binds)
    configs = [nf_dir() / "tasks.config"]
    if not (opts.nextflow and opts.nextflow.nf_config):
        configs.append(local_limits_config(sd / "local.config"))
    return nf.NextflowRun(str(nf_dir() / "tasks.nf"), params=params, work_dir=sd / "work",
                       log_file=sd / "nextflow.log", resume=True, extra_configs=configs)


def dispatch(run_dir: Path, step: str, rows: Sequence[Unit], opts: RunnerOptions) -> list[str]:
    """Write the manifest and run every unit with the selected runner. Returns the nextflow command (if any)."""
    write_manifest(run_dir, step, rows)
    todo = pending(run_dir, step, rows)
    if opts.runner == "inline":
        if opts.dry_run:
            print(f"# inline: {len(todo)}/{len(rows)} units of {step}")
            return []
        from sanntis.runner.inline import run_units

        run_units(run_dir, step, todo, workers=opts.workers, container_prefix=opts.container_prefix)
        return []
    if opts.runner != "nextflow":
        raise TaskError(f"unknown runner {opts.runner!r}")

    nfo = opts.nextflow or NextflowOptions()
    run = nextflow_run(run_dir, step, opts)
    if not todo and not opts.dry_run:
        log.info("%s: all %d units done", step, len(rows))
        return run.command(nfo)
    cmd = nf.run(run, nfo, cwd=step_dir(run_dir, step), dry_run=opts.dry_run)
    if not opts.dry_run:
        missing = pending(run_dir, step, rows)
        if missing:
            raise TaskError(f"{step}: {len(missing)} units finished without a done marker, e.g. {missing[0].unit_id}")
    return cmd
