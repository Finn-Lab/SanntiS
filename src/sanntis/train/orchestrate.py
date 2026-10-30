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

"""``sanntis train``: the Python step graph with a resumable run directory.

Each step writes ``<run>/<step>/`` and, when it finishes, ``<run>/<step>/done.json`` with a key = hash of its
config inputs, its input files and the keys of the steps it depends on. A re-run skips a step whose key is
unchanged; ``--from <step>`` forces that step and every later one. Fan-out steps dispatch their units through
:func:`RunContext.dispatch` (Nextflow ``nf/tasks.nf`` or the inline pool; Nextflow work dir
``<run>/<step>/work`` with ``-resume``).

The steps, in order, are :data:`sanntis.train.steps.STEPS`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from sanntis._version import __version__
from sanntis.config import NextflowOptions, TrainConfig
from sanntis.runner.tasks import RunnerOptions, Unit, dispatch
from sanntis.utils import configure_threads, sha256_file, write_json

log = logging.getLogger(__name__)

DONE = "done.json"


class StepError(RuntimeError):
    pass


@dataclass
class RunContext:
    cfg: TrainConfig
    dry_run: bool = False

    @property
    def run_dir(self) -> Path:
        return Path(self.cfg.run_dir).resolve()

    def step_dir(self, step: str) -> Path:
        d = self.run_dir / step
        d.mkdir(parents=True, exist_ok=True)
        return d

    def dispatch(self, step: str, units: Sequence[Unit]) -> list[str]:
        """Run the units of a fan-out step (the step's own Python then joins their outputs)."""
        opts = RunnerOptions(runner=self.cfg.runner, workers=self.cfg.workers, dry_run=self.dry_run,
                             nextflow=self.cfg.nextflow)
        return dispatch(self.run_dir, step, units, opts)


@dataclass
class Step:
    name: str
    run: Callable[[RunContext], dict]  # returns a JSON-able summary (counts) stored in done.json
    inputs: Callable[[RunContext], dict]  # config values and file paths this step depends on
    deps: tuple[str, ...] = ()
    fanout: bool = False  # dispatches units: still executed under --dry-run (to print its nextflow lines)


def _digest(obj) -> str:
    def norm(v):
        if isinstance(v, Path):
            return {"path": str(v), "sha256": sha256_file(v)} if v.is_file() else str(v)
        if isinstance(v, dict):
            return {str(k): norm(x) for k, x in sorted(v.items())}
        if isinstance(v, (list, tuple)):
            return [norm(x) for x in v]
        return v

    return hashlib.sha256(json.dumps(norm(obj), sort_keys=True, default=str).encode()).hexdigest()


def step_key(step: Step, ctx: RunContext, keys: dict[str, str]) -> str:
    return _digest({"step": step.name, "inputs": step.inputs(ctx), "deps": {d: keys.get(d) for d in step.deps}})


def read_done(ctx: RunContext, name: str) -> dict | None:
    p = ctx.run_dir / name / DONE
    return json.loads(p.read_text()) if p.exists() else None


def run(ctx: RunContext, steps: Sequence[Step], *, from_step: str | None = None,
        until: str | None = None) -> dict[str, dict]:
    names = [s.name for s in steps]
    if from_step and from_step not in names:
        raise StepError(f"--from {from_step!r}: unknown step (steps: {names})")
    forced = set(names[names.index(from_step):]) if from_step else set()
    keys: dict[str, str] = {}
    summary: dict[str, dict] = {}
    ctx.run_dir.mkdir(parents=True, exist_ok=True)
    for s in steps:
        key = step_key(s, ctx, keys)
        keys[s.name] = key
        done = read_done(ctx, s.name)
        if done and done.get("key") == key and s.name not in forced:
            log.info("step %s: done (skipped)", s.name)
            summary[s.name] = {"status": "skipped", **done.get("outputs", {})}
        elif ctx.dry_run and not s.fanout:
            print(f"# step {s.name}: would run")
            summary[s.name] = {"status": "would_run"}
        else:
            log.info("step %s: running", s.name)
            (ctx.run_dir / s.name / DONE).unlink(missing_ok=True)
            t0 = time.time()
            out = s.run(ctx) or {}
            if not ctx.dry_run:
                write_json(ctx.run_dir / s.name / DONE, {
                    "step": s.name, "key": key, "sanntis_version": __version__, "seconds": round(time.time() - t0, 1),
                    "inputs": json.loads(json.dumps(s.inputs(ctx), default=str)),
                    "deps": {d: keys.get(d) for d in s.deps}, "outputs": out})
            summary[s.name] = {"status": "ran", **out}
        if until and s.name == until:
            break
    return summary


# ── CLI ──────────────────────────────────────────────────────────────────────


def build_config(kw: dict) -> TrainConfig:
    """CLI options (+ optional run.yaml; explicit CLI values win) → TrainConfig."""
    base: dict = {}
    if kw.get("config_file"):
        base = yaml.safe_load(Path(kw["config_file"]).read_text()) or {}
    nf = dict(base.pop("nextflow", {}) or {})
    for k in ("container_engine", "nf_config", "nf_profile", "nf_args"):
        v = kw.get(k)
        if v not in (None, (), [], ""):
            nf[k] = list(v) if isinstance(v, tuple) else v
    fields = set(TrainConfig.model_fields) - {"nextflow"}
    if unknown := sorted(set(base) - fields):
        raise StepError(f"run.yaml: unknown option(s) {unknown} (--custom, --labels, --mibig-status and --evaluate "
                        "are not options any more: see docs/train.md)")
    cfg = dict(base)
    for k in fields:
        v = kw.get(k)
        if v is None or v is False or v == ():  # not given on the command line
            continue
        cfg[k] = list(v) if isinstance(v, tuple) else v
    return TrainConfig(**cfg, nextflow=NextflowOptions(**nf))


def run_cli(**kw) -> None:
    from sanntis.train.steps import STEPS
    configure_threads()

    ctx = RunContext(build_config(kw), dry_run=bool(kw.get("dry_run")))
    summary = run(ctx, STEPS, from_step=kw.get("from_step"), until=kw.get("until_step"))
    for name, s in summary.items():
        print(f"{name}: {s.get('status')}")
