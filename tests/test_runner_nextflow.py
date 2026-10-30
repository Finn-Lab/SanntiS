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

from pathlib import Path

from sanntis.config import NextflowOptions
from sanntis.runner.nextflow import NextflowRun, parse_version


def test_command_pass_through(tmp_path):
    cfg = tmp_path / "site.config"
    cfg.write_text("process.executor = 'slurm'\n")
    opts = NextflowOptions(container_engine="singularity", nf_config=[cfg], nf_profile=["mysite"],
                           nf_args="-with-tower --foo 'a b'")
    nf = NextflowRun("ebi-pf-team/interproscan6", revision="6.0.3.0", work_dir=Path("/w"),
                     params={"input": "x.faa", "noMatchesApi": True, "goterms": False, "datadir": None})
    cmd = nf.command(opts)
    assert cmd[:4] == ["nextflow", "run", "ebi-pf-team/interproscan6", "-r"]
    assert cmd[cmd.index("-profile") + 1] == "singularity,mysite"
    assert cmd[cmd.index("-c") + 1] == str(cfg.resolve())
    assert "--noMatchesApi" in cmd and "--goterms" not in cmd and "--datadir" not in cmd
    assert cmd[-3:] == ["-with-tower", "--foo", "a b"]


def test_no_engine_no_profile():
    cmd = NextflowRun("x.nf", resume=False).command(NextflowOptions())
    assert "-profile" not in cmd and "-resume" not in cmd


def test_parse_version():
    assert parse_version("      version 26.10.3 build 12345") == (26, 10, 3)
    assert parse_version("nope") is None
