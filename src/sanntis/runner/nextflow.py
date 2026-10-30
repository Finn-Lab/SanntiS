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

"""``nextflow run`` command lines with infrastructure pass-through.

sanntis maps exactly one thing itself, the container engine (→ ``-profile <engine>``, which IPS6 requires). Every
other piece of infrastructure (executor, queues, accounts, resources per label, site profiles) comes from the
user's own Nextflow config (``--nf-config``, ``--nf-profile``, ``--nf-args``), forwarded to every run unchanged.
"""

from __future__ import annotations

import logging
import os
import re
import shlex
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from sanntis.config import NextflowOptions

log = logging.getLogger(__name__)

MIN_VERSION = (25, 10, 4)  # IPS6 6.0.3.0 manifest: nextflowVersion '!>=25.10.4'
# 25.10.x needs NXF_SYNTAX_PARSER=v2 for IPS6's typed params, and that parser rejects `-profile <engine>` as soon
# as a -c config is added. 26.x handles both with its default parser.
RECOMMENDED_VERSION = (26, 4, 0)


class NextflowError(RuntimeError):
    pass


@dataclass
class NextflowRun:
    """One top-level ``nextflow run`` (never nested)."""

    pipeline: str  # remote id (owner/repo) or a local .nf path
    params: Mapping[str, object] = field(default_factory=dict)
    revision: str | None = None
    work_dir: Path | None = None
    log_file: Path | None = None
    resume: bool = True
    extra_configs: Sequence[Path] = ()  # sanntis-owned configs (e.g. nf/tasks.config), before the user's

    def command(self, opts: NextflowOptions) -> list[str]:
        cmd = [opts.nextflow]
        if self.log_file:
            cmd += ["-log", str(self.log_file)]
        cmd += ["run", self.pipeline]
        if self.revision:
            cmd += ["-r", self.revision]
        profiles = ([opts.container_engine] if opts.container_engine else []) + list(opts.nf_profile)
        if profiles:
            cmd += ["-profile", ",".join(profiles)]
        for c in list(self.extra_configs) + list(opts.nf_config):
            cmd += ["-c", str(Path(c).resolve())]
        if self.work_dir:
            cmd += ["-work-dir", str(self.work_dir)]
        if self.resume:
            cmd += ["-resume"]
        for k, v in self.params.items():
            if v is None or v is False:
                continue
            cmd += [f"--{k}"] if v is True else [f"--{k}", str(v)]
        cmd += shlex.split(opts.nf_args or "")
        return cmd


def env() -> dict[str, str]:
    """Process environment for nextflow: inherit, but never pass a stale NXF_SYNTAX_PARSER=v1."""
    e = dict(os.environ)
    if e.get("NXF_SYNTAX_PARSER") == "v1":
        e.pop("NXF_SYNTAX_PARSER")
    return e


def parse_version(text: str) -> tuple[int, int, int] | None:
    m = re.search(r"version\s+(\d+)\.(\d+)\.(\d+)", text)
    return tuple(int(x) for x in m.groups()) if m else None  # type: ignore[return-value]


def check_version(opts: NextflowOptions) -> tuple[int, int, int]:
    exe = shutil.which(opts.nextflow) or (opts.nextflow if Path(opts.nextflow).is_file() else None)
    if exe is None:
        raise NextflowError(f"'{opts.nextflow}' not found on PATH: install Nextflow >= "
                            f"{'.'.join(map(str, RECOMMENDED_VERSION))} (or pass --ips-tsv to skip InterProScan)")
    out = subprocess.run([exe, "-version"], capture_output=True, text=True, env=env(), check=False)
    v = parse_version(out.stdout + out.stderr)
    if v is None:
        raise NextflowError(f"cannot read the Nextflow version from `{exe} -version`")
    if v < MIN_VERSION:
        need, rec = ".".join(map(str, MIN_VERSION)), ".".join(map(str, RECOMMENDED_VERSION))
        raise NextflowError(f"Nextflow {'.'.join(map(str, v))} is too old; InterProScan 6 needs >= {need} "
                            f"(recommended {rec})")
    if v < RECOMMENDED_VERSION:
        log.warning("Nextflow %s: InterProScan 6 runs with a -c config are known to fail before %s",
                    ".".join(map(str, v)), ".".join(map(str, RECOMMENDED_VERSION)))
    return v


def run(nf: NextflowRun, opts: NextflowOptions, *, cwd: Path, dry_run: bool = False) -> list[str]:
    """Launch one top-level run in ``cwd`` (its .nextflow/ history lives there). Raises on failure."""
    cmd = nf.command(opts)
    log.info("nextflow: %s", shlex.join(cmd))
    if dry_run:
        print(shlex.join(cmd))
        return cmd
    check_version(opts)
    cwd.mkdir(parents=True, exist_ok=True)
    p = subprocess.run(cmd, cwd=cwd, env=env(), check=False)
    if p.returncode != 0:
        raise NextflowError(f"nextflow run {nf.pipeline} failed (exit {p.returncode}); see "
                            f"{nf.log_file or cwd / '.nextflow.log'}")
    return cmd
