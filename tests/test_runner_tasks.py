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

import shutil
import sys
from pathlib import Path

import pytest

from sanntis.config import NextflowOptions
from sanntis.runner import tasks
from sanntis.runner.tasks import RunnerOptions, TaskError, Unit, dispatch, write_manifest


@pytest.fixture(autouse=True)
def _echo_registry(monkeypatch):
    """A unit function importable by spawned workers and by `python -m sanntis task` (env registry)."""
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv(tasks.TEST_TASK_ENV, "t_echo=tests.unit_fns:echo_unit")
    monkeypatch.setenv("PYTHONPATH", f"{root}:{root / 'src'}")


def test_manifest_roundtrip(tmp_path):
    p = write_manifest(tmp_path, "s", [Unit("a", "cpu_small"), Unit("b", "train")])
    assert p.read_text() == "unit_id\tlabel\na\tcpu_small\nb\ttrain\n"
    with pytest.raises(TaskError):
        write_manifest(tmp_path, "s", [Unit("a", "nope")])
    with pytest.raises(TaskError):
        write_manifest(tmp_path, "s", [Unit("a", "cpu_small"), Unit("a", "train")])


def test_run_unit_idempotent(tmp_path):
    (tmp_path / "t_echo").mkdir()
    tasks.run_unit("t_echo", tmp_path, "u1")
    (tmp_path / "t_echo" / "u1.out").unlink()
    tasks.run_unit("t_echo", tmp_path, "u1")  # done marker present: not run again
    assert not (tmp_path / "t_echo" / "u1.out").exists()
    with pytest.raises(TaskError):
        tasks.run_unit("no_such_step", tmp_path, "x")


@pytest.mark.parametrize("workers", [1, 2])
def test_inline_dispatch(tmp_path, workers):
    units = [Unit(f"u{i}", "cpu_small") for i in range(3)]
    tasks.write_unit_script(tmp_path, "t_echo", "s0", f"echo script > {tmp_path}/t_echo/s0.out")
    dispatch(tmp_path, "t_echo", [*units, Unit("s0", "antismash")], RunnerOptions(runner="inline", workers=workers))
    for i in range(3):
        assert (tmp_path / "t_echo" / f"u{i}.out").read_text() == f"u{i}\n"
    assert (tmp_path / "t_echo" / "s0.out").read_text() == "script\n"


def test_inline_failure_reported(tmp_path):
    with pytest.raises(TaskError, match="1/1 units failed"):
        dispatch(tmp_path, "t_echo", [Unit("fail", "cpu_small")], RunnerOptions(runner="inline"))


def test_nextflow_command(tmp_path, capsys):
    cfg = tmp_path / "site.config"
    cfg.write_text("process.executor = 'slurm'\n")
    opts = RunnerOptions(runner="nextflow", dry_run=True,
                         nextflow=NextflowOptions(container_engine="singularity", nf_config=[cfg]))
    cmd = dispatch(tmp_path, "t_echo", [Unit("u0", "cpu_small")], opts)
    out = capsys.readouterr().out
    assert "tasks.nf" in out and "-resume" in out
    assert cmd[cmd.index("--python") + 1] == sys.executable
    cs = [cmd[i + 1] for i, x in enumerate(cmd) if x == "-c"]
    assert cs[0].endswith("nf/tasks.config") and cs[1] == str(cfg.resolve())  # user config last (wins)
    assert cmd[cmd.index("-profile") + 1] == "singularity"
    assert cmd[cmd.index("-work-dir") + 1] == str((tmp_path / "t_echo" / "work").resolve())


def test_nf_files_packaged():
    d = tasks.nf_dir()
    assert (d / "tasks.nf").is_file() and (d / "tasks.config").is_file()


@pytest.mark.nextflow
@pytest.mark.skipif(shutil.which("nextflow") is None, reason="nextflow not on PATH")
def test_nextflow_local(tmp_path, monkeypatch):
    units = [Unit(f"u{i}", "cpu_small") for i in range(2)]
    tasks.write_unit_script(tmp_path, "t_echo", "s0", f"echo script > {tmp_path}/t_echo/s0.out")
    dispatch(tmp_path, "t_echo", [*units, Unit("s0", "antismash")], RunnerOptions(runner="nextflow"))
    assert (tmp_path / "t_echo" / "u1.out").read_text() == "u1\n"
    assert (tmp_path / "t_echo" / "s0.out").read_text() == "script\n"
