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

from click.testing import CliRunner

from sanntis import __version__
from sanntis.cli import main


def test_version():
    r = CliRunner().invoke(main, ["--version"])
    assert r.exit_code == 0 and __version__ in r.output


def test_help_lists_commands():
    r = CliRunner().invoke(main, ["--help"])
    for cmd in ("predict", "setup", "example", "train", "build-negatives"):
        assert cmd in r.output


def test_example_copies_inputs(tmp_path):
    r = CliRunner().invoke(main, ["example", "-o", str(tmp_path / "ex")])
    assert r.exit_code == 0, r.output
    names = sorted(p.name for p in (tmp_path / "ex").iterdir())
    assert names == ["ERZ29794225_slice.faa", "ERZ29794225_slice.gff", "ERZ29794225_slice.ips.tsv"]
    assert "sanntis predict" in r.output
