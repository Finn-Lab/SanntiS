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

"""``sanntis train`` end to end on the synthetic fixture, with the inline and the Nextflow runner."""

import json
import shutil

import pytest
from click.testing import CliRunner

from sanntis.cli import main

from .train_fixture import build, train_args


@pytest.fixture(autouse=True)
def _fast(monkeypatch):
    monkeypatch.setenv("SANNTIS_DEBUG_MAX_EPOCHS", "3")


def _build(tmp_path, monkeypatch):
    fx = build(tmp_path / "fx")
    monkeypatch.setenv("SANNTIS_CACHE", str(fx["cache"]))
    return fx


def _train(fx, run, runner):
    r = CliRunner().invoke(main, train_args(fx, run, runner), catch_exceptions=False)
    assert r.exit_code == 0, r.output
    return r.output


def test_train_inline(tmp_path, monkeypatch):
    fx = _build(tmp_path, monkeypatch)
    run = tmp_path / "run"
    out = _train(fx, run, "inline")
    for step in ("collect", "annotate", "entities", "cv", "tune", "finalize", "report", "package"):
        assert f"{step}: ran" in out
    assert "evaluate" not in out and not (run / "eval_bundle").exists()
    pkg = json.loads((run / "package" / "done.json").read_text())["outputs"]
    assert (run / "bundle" / "manifest.json").exists() and (run / "bundle" / "report" / "report.md").exists()
    mf = json.loads((run / "bundle" / "manifest.json").read_text())
    assert mf["tuned"]["t_star"] == pkg["t_star"] and mf["debug_max_epochs"] == 3
    cfg = json.loads((run / "bundle" / "config.json").read_text())
    assert cfg["heads"]["class"] is True and cfg["heads"]["npc"] is False  # 60 labelled BGCs, no structures
    assert cfg["coverage_reference"]["frac_cds_with_tokens"] > 0.9
    # resume: nothing re-runs
    assert "ran" not in _train(fx, run, "inline")


@pytest.mark.nextflow
@pytest.mark.slow
def test_train_runners_agree(tmp_path, monkeypatch):
    if shutil.which("nextflow") is None:
        pytest.skip("nextflow not on PATH")
    fx = _build(tmp_path, monkeypatch)
    _train(fx, tmp_path / "inline", "inline")
    _train(fx, tmp_path / "nf", "nextflow")
    a = json.loads((tmp_path / "inline" / "tune" / "tune.json").read_text())
    b = json.loads((tmp_path / "nf" / "tune" / "tune.json").read_text())
    assert (a["epochs"], a["t_star"], a["best_epochs"]) == (b["epochs"], b["t_star"], b["best_epochs"])
    assert a["val_orf_aucpr"] == pytest.approx(b["val_orf_aucpr"], abs=1e-6)
