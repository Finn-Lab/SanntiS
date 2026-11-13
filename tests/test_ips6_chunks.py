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

import threading
import time

import pytest

from sanntis.annotate import ips6
from sanntis.config import NextflowOptions
from sanntis.io.fasta import iter_fasta, write_protein_fasta
from sanntis.runner.nextflow import NextflowError


def _fake_run(calls, fail_first_with=None, delay=0.0, active=None):
    def run(faa, run_dir, opts, datadir, apps=None, keep_work=False, plan=None):
        calls.append(str(faa))
        if active is not None:
            with active["lock"]:
                active["now"] += 1
                active["max"] = max(active["max"], active["now"])
            time.sleep(delay)
            with active["lock"]:
                active["now"] -= 1
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


def test_parallel_chunks_keep_order(tmp_path, monkeypatch):
    faa = tmp_path / "p.faa"
    write_protein_fasta([(f"{i:032x}", "MKV") for i in range(9)], faa)
    calls, active = [], {"lock": threading.Lock(), "now": 0, "max": 0}
    monkeypatch.setattr(ips6, "run_ips6", _fake_run(calls, delay=0.2, active=active))
    plan = ips6.RunPlan(ips6.Ips6Settings(parallel=3, chunk=2), ips6.Lookup(False, ips6.MATCHES_API))
    tsv = ips6.run_ips6_chunked(faa, tmp_path / "run", NextflowOptions(), tmp_path, plan=plan)
    assert len(calls) == 5 and active["max"] == 3
    assert [line.split("\t")[0] for line in tsv.read_text().splitlines()] == [f"{i:032x}" for i in range(9)]
    assert (tmp_path / "run" / "lookup.json").is_file()


def test_failed_chunk_keeps_the_others(tmp_path, monkeypatch):
    faa = tmp_path / "p.faa"
    write_protein_fasta([(f"{i:032x}", "MKV") for i in range(4)], faa)
    ok = _fake_run([])

    def run(faa, run_dir, *a, **k):
        if run_dir.name == "chunk0001":
            raise NextflowError("boom")
        return ok(faa, run_dir, *a, **k)

    monkeypatch.setattr(ips6, "run_ips6", run)
    plan = ips6.RunPlan(ips6.Ips6Settings(parallel=2, chunk=2), ips6.Lookup(False, ips6.MATCHES_API))
    with pytest.raises(ips6.Ips6Error, match="1 of 2"):
        ips6.run_ips6_chunked(faa, tmp_path / "run", NextflowOptions(), tmp_path, plan=plan)
    assert (tmp_path / "run" / "chunk0000" / "results" / "proteins.tsv").is_file()


def test_settings_from_env(monkeypatch):
    for k, v in {"SANNTIS_IPS6_CPUS": "4", "SANNTIS_IPS6_PARALLEL": "16", "SANNTIS_IPS6_QUEUE_SIZE": "40",
                 "SANNTIS_IPS6_SUBMIT_RATE": "100/1min", "SANNTIS_IPS6_LOOKUP": "auto"}.items():
        monkeypatch.setenv(k, v)
    s = ips6.Ips6Settings.from_env()
    assert (s.cpus, s.parallel, s.queue_size, s.submit_rate, s.lookup, s.chunk) == (4, 16, 40, "100/1min", "auto",
                                                                                     20000)
    monkeypatch.setenv("SANNTIS_IPS6_LOOKUP", "on")
    with pytest.raises(ips6.Ips6Error):
        ips6.Ips6Settings.from_env()
    monkeypatch.setenv("SANNTIS_IPS6_LOOKUP", "off")
    monkeypatch.setenv("SANNTIS_IPS6_PARALLEL", "0")
    with pytest.raises(ips6.Ips6Error):
        ips6.Ips6Settings.from_env()


@pytest.mark.parametrize("api,used", [("107.0", True), ("111.0", False), (None, False)])
def test_lookup_only_for_the_pinned_release(monkeypatch, api, used):
    monkeypatch.setattr(ips6, "matches_api_release", lambda url, timeout=30: api)
    lk = ips6.resolve_lookup(ips6.Ips6Settings(lookup="auto"))
    assert lk.used is used and lk.api_release == api
    params = ips6.nextflow_run("p.faa", "out", "data", ips6.Path("run"), cpus=4, lookup=lk).params
    assert params["cpus"] == 4
    if used:
        assert "noMatchesApi" not in params and params["matchesApiUrl"] == ips6.MATCHES_API
    else:
        assert params["noMatchesApi"] is True and "matchesApiUrl" not in params


def test_lookup_off_never_queries(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("no /info request with SANNTIS_IPS6_LOOKUP=off")

    monkeypatch.setattr(ips6, "matches_api_release", boom)
    assert ips6.resolve_lookup(ips6.Ips6Settings(lookup="off")).used is False


def test_executor_config_goes_last(tmp_path):
    user = tmp_path / "site.config"
    user.write_text("executor { queueSize = 200 }")
    plan = ips6.run_plan(tmp_path, ips6.Ips6Settings(queue_size=40, submit_rate="100/1min", lookup="off"))
    cfg = plan.executor_config.read_text()
    assert "queueSize = 40" in cfg and "submitRateLimit = '100/1min'" in cfg
    opts = plan.options(NextflowOptions(nf_config=[user]))
    assert opts.nf_config == [user, plan.executor_config]
    assert ips6.run_plan(tmp_path / "x", ips6.Ips6Settings(lookup="off")).executor_config is None


@pytest.mark.parametrize("arg", ["--cpus 8", "--matchesApiUrl http://x", "--noMatchesApi"])
def test_nf_args_cannot_override_lookup_or_cpus(arg):
    with pytest.raises(ips6.Ips6Error):
        ips6.check_nf_args(NextflowOptions(nf_args=arg))
