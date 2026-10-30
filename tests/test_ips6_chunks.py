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

import pytest

from sanntis.annotate import ips6
from sanntis.config import NextflowOptions
from sanntis.io.fasta import iter_fasta, write_protein_fasta
from sanntis.runner.nextflow import NextflowError


def _fake_run(calls, fail_first_with=None):
    def run(faa, run_dir, opts, datadir, apps=None, keep_work=False):
        calls.append(str(faa))
        if fail_first_with and len(calls) == 1:
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "nextflow.log").write_text(fail_first_with)
            raise NextflowError("boom")
        out = run_dir / "results" / "proteins.tsv"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(f"{h}\t{h}\t10\tPfam\tPF00001\t-\t1\t9\t-\tPfam\t-\t-\t-\t-\t-\n"
                               for h, _ in iter_fasta(faa)))
        return out
    return run


def test_chunked_concatenates_and_resumes(tmp_path, monkeypatch):
    faa = tmp_path / "p.faa"
    write_protein_fasta([(f"{i:032x}", "MKV") for i in range(5)], faa)
    calls = []
    monkeypatch.setattr(ips6, "run_ips6", _fake_run(calls))
    tsv = ips6.run_ips6_chunked(faa, tmp_path / "run", NextflowOptions(), tmp_path, chunk=2)
    assert len(calls) == 3 and len(tsv.read_text().splitlines()) == 5
    calls.clear()
    ips6.run_ips6_chunked(faa, tmp_path / "run", NextflowOptions(), tmp_path, chunk=2)
    assert calls == []  # finished run: nothing re-annotated


@pytest.mark.parametrize("log,retried", [("x KryoException: Buffer underflow y", True), ("other error", False)])
def test_race_retry(tmp_path, monkeypatch, log, retried):
    faa = tmp_path / "p.faa"
    write_protein_fasta([("a" * 32, "MKV")], faa)
    calls = []
    monkeypatch.setattr(ips6, "run_ips6", _fake_run(calls, fail_first_with=log))
    if retried:
        ips6.run_ips6_retry(faa, tmp_path / "run", NextflowOptions(), tmp_path)
        assert len(calls) == 2
    else:
        with pytest.raises(NextflowError):
            ips6.run_ips6_retry(faa, tmp_path / "run", NextflowOptions(), tmp_path)
