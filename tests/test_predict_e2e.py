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

"""`sanntis predict` end to end, with InterProScan 6 stubbed by a fake `nextflow` on PATH that replays the recorded
IPS6 6.0.3.0 / 107.0 output of the fixture proteins (tests/fixtures/ips6.tsv)."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

from sanntis.cli import main
from sanntis.data import EXAMPLE_DIR
from sanntis.io.fasta import InputError
from sanntis.io.ips_tsv import IpsTsvError, read_ips_tsv

FIX = Path(__file__).parent / "fixtures"
GBK, FNA = FIX / "BGC0003066.gbk", FIX / "BGC0002866.fna"


@pytest.fixture
def fake_nf(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    exe = bin_dir / "nextflow"
    script = str(Path(__file__).parent / "fake_nextflow.py")
    exe.write_text(f"#!{sys.executable}\nimport runpy\nrunpy.run_path({script!r}, run_name='__main__')\n")
    exe.chmod(0o755)
    calls = tmp_path / "calls.jsonl"
    monkeypatch.setenv("PATH", f"{bin_dir}:{__import__('os').environ['PATH']}")
    monkeypatch.setenv("FAKE_NF_CALLS", str(calls))
    monkeypatch.setenv("FAKE_IPS_TSV", str(FIX / "ips6.tsv"))
    monkeypatch.setenv("SANNTIS_CACHE", str(tmp_path / "cache"))

    def n_calls():
        return len(calls.read_text().splitlines()) if calls.exists() else 0

    return n_calls


def predict(*args):
    r = CliRunner().invoke(main, ["predict", *map(str, args)], catch_exceptions=False)
    assert r.exit_code == 0, r.output
    return r


def gff_header(path: Path) -> dict:
    return {line[2:].split(" ", 1)[0]: line[2:].rstrip("\n").split(" ", 1)[1]
            for line in path.read_text().splitlines() if line.startswith("##") and " " in line}


def features(path: Path) -> list[str]:
    return [x for x in path.read_text().splitlines() if not x.startswith("#")]


def test_recorded_ips6_fixture_layout():
    h = read_ips_tsv(FIX / "ips6.tsv")
    assert h.version == "6" and not h.dropped
    assert h.hits["md5"].str.fullmatch(r"[0-9a-f]{32}").all()  # IPS6 writes upper case; parser lower-cases


def test_predict_managed_ips6_batch_and_cache(tmp_path, fake_nf, dummy_bundle):
    out = tmp_path / "out"
    predict(GBK, FNA, "--model", dummy_bundle, "-o", out, "--container-engine", "docker", "--write-faa")
    assert fake_nf() == 3  # two-step data setup (no data yet) + ONE pooled annotation run
    for p in ("BGC0003066", "BGC0002866"):
        for suffix in ("sanntis.gff3", "sanntis.antismash.json", "orf_scores.tsv", "regions.tsv", "run.json", "cds.faa",
                       "cds.gff3"):
            assert (out / f"{p}.{suffix}").is_file()
    side = json.loads((out / "BGC0003066.sanntis.antismash.json").read_text())
    gff = [x.split("\t") for x in features(out / "BGC0003066.sanntis.gff3")]
    assert [(r["name"], s["start"] + 1, s["end"]) for r in side["records"] for s in r["subregions"]] == \
        [(f[0], int(f[3]), int(f[4])) for f in gff]
    meta = json.loads((out / "BGC0002866.run.json").read_text())
    assert meta["ips"]["mode"] == "managed" and meta["ips"]["sent_to_ips6"] == meta["ips"]["unique_proteins"]
    assert meta["ips"]["setup_ran"] and meta["sample"]["genecall"] == "pyrodigal meta"
    assert meta["qc"]["n_cds"] == len(pd.read_csv(out / "BGC0002866.orf_scores.tsv", sep="\t"))
    h = gff_header(out / "BGC0003066.sanntis.gff3")
    assert h["ips"] == "InterProScan 6.0.3.0 InterPro 107.0 (managed)"
    assert h["kingdom"] == "proka" and "qc-flags" in h and h["model-source"] == str(Path(dummy_bundle).resolve())
    assert float(h["qc-cds-with-tokens"].split()[0]) > 0.5

    # re-run (other outdir): every protein comes from the cache, nextflow is not called, same calls
    out2 = tmp_path / "out2"
    predict(GBK, "--model", dummy_bundle, "-o", out2, "--container-engine", "docker")
    assert fake_nf() == 3
    meta2 = json.loads((out2 / "BGC0003066.run.json").read_text())
    assert meta2["ips"]["sent_to_ips6"] == 0 and meta2["ips"]["cache_hits"] == meta2["ips"]["unique_proteins"]
    assert features(out2 / "BGC0003066.sanntis.gff3") == features(out / "BGC0003066.sanntis.gff3")
    o1 = pd.read_csv(out / "BGC0003066.orf_scores.tsv", sep="\t")
    pd.testing.assert_frame_equal(o1, pd.read_csv(out2 / "BGC0003066.orf_scores.tsv", sep="\t"))


def test_batch_partial_cache(tmp_path, fake_nf, dummy_bundle):
    predict(GBK, "--model", dummy_bundle, "-o", tmp_path / "a")
    n = fake_nf()
    predict(GBK, FNA, "--model", dummy_bundle, "-o", tmp_path / "b")
    assert fake_nf() == n + 1  # only the .fna proteins are new
    meta = json.loads((tmp_path / "b" / "BGC0002866.run.json").read_text())
    assert 0 < meta["ips"]["sent_to_ips6"] < meta["ips"]["unique_proteins"]
    assert (tmp_path / "b" / "BGC0003066.sanntis.gff3").is_file()


def _ips6_by_locus_tag(tmp_path) -> Path:
    """The recorded IPS6 TSV with protein ids = the GBK locus tags (as if IPS6 ran on the GBK's own FASTA)."""
    from sanntis.io.genbank import read_genbank

    cds, _ = read_genbank(GBK)
    tag = dict(zip(cds["md5"], cds["locus_tag"], strict=True))
    rows = []
    for line in (FIX / "ips6.tsv").read_text().splitlines():
        s = line.split("\t")
        if s[1].lower() in tag:
            rows.append("\t".join([tag[s[1].lower()]] + s[1:]))
    p = tmp_path / "gbk.ips6.tsv"
    p.write_text("\n".join(rows) + "\n")
    return p


@pytest.mark.parametrize("version", ["5", "6"])
def test_ips_tsv(tmp_path, fake_nf, dummy_bundle, version):
    tsv = FIX / "ips576.tsv" if version == "5" else _ips6_by_locus_tag(tmp_path)
    out = tmp_path / "out"
    predict(GBK, "--ips-tsv", tsv, "--model", dummy_bundle, "-o", out)
    assert fake_nf() == 0  # no InterProScan run
    meta = json.loads((out / "BGC0003066.run.json").read_text())
    ips = meta["ips"]
    assert ips["mode"] == "tsv" and ips["detected_tsv_version"] == version and ips["interpro_release"] == "unknown"
    if version == "5":  # protein ids are md5s → md5 fallback join
        assert ips["join"]["by_md5"] > 0 and ips["join"]["by_id"] == 0
    else:
        assert ips["join"]["by_id"] > 0 and ips["join"]["by_md5"] == 0
    assert meta["qc"]["n_cds_with_tokens"] > 10
    h = gff_header(out / "BGC0003066.sanntis.gff3")
    assert h["ips"] == f"InterProScan {version}.x InterPro unknown (tsv, TSV v{version})"


def test_ips_tsv_and_managed_agree(tmp_path, fake_nf, dummy_bundle):
    """IPS6 TSV given by hand == the managed IPS6 path (same recorded output)."""
    predict(GBK, "--model", dummy_bundle, "-o", tmp_path / "m")
    predict(GBK, "--ips-tsv", _ips6_by_locus_tag(tmp_path), "--model", dummy_bundle, "-o", tmp_path / "t")
    assert features(tmp_path / "m" / "BGC0003066.sanntis.gff3") == features(tmp_path / "t" / "BGC0003066.sanntis.gff3")


def test_ips_tsv_wrong_file(tmp_path, dummy_bundle):
    bad = tmp_path / "bad.tsv"
    bad.write_text("x1\tmd5x\t100\tPfam\tPF00001\t-\t1\t50\t1E-5\tT\t-\tIPR000001\t-\t-\t-\n")
    with pytest.raises(InputError, match="no protein matches"):
        CliRunner().invoke(main, ["predict", str(GBK), "--ips-tsv", str(bad), "--model", str(dummy_bundle), "-o",
                                  str(tmp_path / "o")], catch_exceptions=False)


def test_ips_tsv_required_apps(tmp_path, dummy_bundle):
    rows = [f"p{i}\tm{i}\t100\tPfam\tPF00001\t-\t1\t50\t1E-5\tT\t-\tIPR000001\t-\t-\t-" for i in range(120)]
    rows += (FIX / "ips576.tsv").read_text().splitlines()
    rows = [r for r in rows if "\tGene3D\t" not in r]
    tsv = tmp_path / "x.tsv"
    tsv.write_text("\n".join(rows) + "\n")
    args = ["predict", str(GBK), "--ips-tsv", str(tsv), "--model", str(dummy_bundle), "-o", str(tmp_path / "o")]
    with pytest.raises(IpsTsvError, match="required"):
        CliRunner().invoke(main, args, catch_exceptions=False)
    predict(*args[1:], "--allow-incomplete-ips")
    h = gff_header(tmp_path / "o" / "BGC0003066.sanntis.gff3")
    assert "missing_apps:" in h["qc-flags"] and "gene3d" in h["qc-flags"]


def test_qc_reference_flags(tmp_path, fake_nf, dummy_model, monkeypatch):
    monkeypatch.setattr("sanntis.qc.APP_MIN_CDS", 1)  # the fixture GenBank has 19 CDS
    dummy_model.config.coverage_reference = {"frac_cds_with_tokens": 0.99, "frac_unk_tokens": 0.0,
                                             "hits_per_cds": {"pfam": 50.0}}
    b = dummy_model.save_pretrained(tmp_path / "bundle")
    predict(GBK, "--ips-tsv", FIX / "ips576.tsv", "--model", b, "-o", tmp_path / "o")
    h = gff_header(tmp_path / "o" / "BGC0003066.sanntis.gff3")
    assert set(h["qc-flags"].split(",")) >= {"high_unk_share", "low_pfam_hits"}


def test_input_errors(tmp_path, fake_nf, dummy_bundle):
    with pytest.raises(InputError, match="eukaryotic"):
        CliRunner().invoke(main, ["predict", str(FNA), "--kingdom", "euk", "--ips-tsv", str(FIX / "ips576.tsv"),
                                  "--model", str(dummy_bundle)], catch_exceptions=False)
    with pytest.raises(InputError, match="duplicate"):
        CliRunner().invoke(main, ["predict", str(GBK), str(GBK), "--model", str(dummy_bundle)],
                           catch_exceptions=False)
    faa, gff = str(EXAMPLE_DIR / "ERZ29794225_slice.faa"), str(EXAMPLE_DIR / "ERZ29794225_slice.gff")
    for args, match in [([faa], "needs --gff"), ([str(GBK), "--gff", gff], "exactly one .faa"),
                        ([faa, str(GBK), "--gff", gff], "exactly one .faa")]:
        with pytest.raises(InputError, match=match):
            CliRunner().invoke(main, ["predict", *args, "--model", str(dummy_bundle)], catch_exceptions=False)
    r = CliRunner().invoke(main, ["predict", str(GBK), "--model", str(dummy_bundle), "--nf-args", "--interpro 99"])
    assert r.exit_code != 0 and "pinned" in str(r.exception)


def test_setup_command(tmp_path, fake_nf):
    r = CliRunner().invoke(main, ["setup", "--container-engine", "docker", "--datadir", str(tmp_path / "d")],
                           catch_exceptions=False)
    assert r.exit_code == 0 and fake_nf() == 2
    calls = [json.loads(x) for x in (tmp_path / "calls.jsonl").read_text().splitlines()]
    apps = [c[c.index("--applications") + 1].split(",") for c in calls]
    assert "prositepatterns" not in apps[0] and "prositepatterns" in apps[1] and len(apps[1]) == 14
    assert all(c[c.index("-profile") + 1] == "docker" for c in calls)


SLICE = EXAMPLE_DIR / "ERZ29794225_slice"


def test_faa_gff_with_ips_tsv(tmp_path, fake_nf, dummy_bundle):
    out = tmp_path / "o"
    predict(SLICE.with_suffix(".faa"), "--gff", SLICE.with_suffix(".gff"), "--ips-tsv", SLICE.with_suffix(".ips.tsv"),
            "--ips-release", "107.0", "--model", dummy_bundle, "-o", out)
    assert fake_nf() == 0  # no InterProScan run
    meta = json.loads((out / "ERZ29794225_slice.run.json").read_text())
    assert meta["sample"]["kind"] == "protein" and meta["sample"]["bp"] is None
    assert meta["sample"]["records"] == 3 and meta["counts"]["cds"] == 24
    assert meta["sample"]["gff"] == str(SLICE.with_suffix(".gff").resolve())
    join = meta["ips"]["join"]
    # TSV ids = FAA ids; 20 proteins in the TSV, one (ERZ29794225_46_7) with only a SignalP hit, which is dropped
    assert join["by_id"] == 19 and join["by_md5"] == 0 and join["id_md5_conflicts"] == 0
    assert gff_header(out / "ERZ29794225_slice.sanntis.gff3")["ips"] == "InterProScan 5.x InterPro 107.0 (tsv, TSV v5)"


def test_faa_gff_order_invariant(tmp_path, fake_nf, dummy_bundle):
    """Gene order comes from the GFF coordinates, never from the file order of proteins or rows (records are
    reported in order of first appearance in the GFF, so compare per record)."""
    import random

    rng = random.Random(0)
    recs = SLICE.with_suffix(".faa").read_text().split(">")[1:]
    rows = SLICE.with_suffix(".gff").read_text().splitlines()[1:]
    rng.shuffle(recs)
    rng.shuffle(rows)
    d = tmp_path / "shuffled"
    d.mkdir()
    (d / "ERZ29794225_slice.faa").write_text("".join(">" + r for r in recs))
    (d / "ERZ29794225_slice.gff").write_text("##gff-version 3\n" + "\n".join(rows) + "\n")
    tsv = SLICE.with_suffix(".ips.tsv")
    for name, faa, gff in [("a", SLICE.with_suffix(".faa"), SLICE.with_suffix(".gff")),
                           ("b", d / "ERZ29794225_slice.faa", d / "ERZ29794225_slice.gff")]:
        predict(faa, "--gff", gff, "--ips-tsv", tsv, "--model", dummy_bundle, "-o", tmp_path / name)
    for suffix in ("orf_scores.tsv", "regions.tsv"):
        a, b = (pd.read_csv(tmp_path / x / f"ERZ29794225_slice.{suffix}", sep="\t") for x in "ab")
        key = list(a.columns[:2])
        pd.testing.assert_frame_equal(a.sort_values(key, ignore_index=True), b.sort_values(key, ignore_index=True))


def test_faa_gff_reads_write_faa_output(tmp_path, fake_nf, dummy_bundle):
    from sanntis.io.genbank import read_genbank
    from sanntis.io.gff_cds import read_faa_gff

    predict(GBK, "--ips-tsv", FIX / "ips576.tsv", "--model", dummy_bundle, "-o", tmp_path, "--write-faa")
    cds, _ = read_faa_gff(tmp_path / "BGC0003066.cds.faa", tmp_path / "BGC0003066.cds.gff3")
    ref, _ = read_genbank(GBK)
    cols = ["record_id", "locus_tag", "start", "end", "strand", "md5"]
    pd.testing.assert_frame_equal(cds[cols], ref[cols])
