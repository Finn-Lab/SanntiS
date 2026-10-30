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

"""``--runner inline``: the units of a fan-out step in-process, with a process pool.

The per-unit code path is the one ``nf/tasks.nf`` runs: :func:`sanntis.runner.tasks.run_unit` for Python labels,
the unit's shell script for script labels (``antismash``), wrapped in ``container_prefix`` when given.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import subprocess
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from sanntis.runner.tasks import SCRIPT_LABELS, TaskError, Unit, done_marker, run_unit, unit_dir
from sanntis.utils import available_cpus

log = logging.getLogger(__name__)


def _init_worker(threads: int) -> None:
    os.environ["SANNTIS_THREADS"] = str(threads)


def _run_one(run_dir: str, step: str, unit_id: str, label: str, prefix: tuple[str, ...]) -> str:
    if label in SCRIPT_LABELS:
        script = unit_dir(Path(run_dir), step) / f"{unit_id}.sh"
        if not script.exists():
            raise TaskError(f"missing unit script {script}")
        p = subprocess.run([*prefix, "bash", str(script)], capture_output=True, text=True, check=False)
        if p.returncode != 0 or not done_marker(Path(run_dir), step, unit_id).exists():
            raise TaskError(f"{step}/{unit_id} failed (exit {p.returncode}): {p.stderr[-2000:]}")
    else:
        run_unit(step, Path(run_dir), unit_id)
    return unit_id


def run_units(run_dir: Path, step: str, units: Sequence[Unit], *, workers: int = 1,
              container_prefix: Sequence[str] = ()) -> None:
    run_dir = Path(run_dir).resolve()
    prefix = tuple(container_prefix)
    errors: list[str] = []
    if workers <= 1 or len(units) <= 1:
        for u in units:
            try:
                _run_one(str(run_dir), step, u.unit_id, u.label, prefix)
            except Exception as e:  # collect, report all
                log.error("%s/%s: %s", step, u.unit_id, e)
                errors.append(f"{u.unit_id}: {e}")
    else:
        # spawn: units may use torch / threads; never fork a multi-threaded parent

        threads = max(1, available_cpus() // workers)
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"), initializer=_init_worker,
                                 initargs=(threads,)) as ex:
            futs = {ex.submit(_run_one, str(run_dir), step, u.unit_id, u.label, prefix): u for u in units}
            for f in as_completed(futs):
                try:
                    f.result()
                except Exception as e:
                    log.error("%s/%s: %s", step, futs[f].unit_id, e)
                    errors.append(f"{futs[f].unit_id}: {e}")
    if errors:
        raise TaskError(f"{step}: {len(errors)}/{len(units)} units failed; first: {errors[0]}")
