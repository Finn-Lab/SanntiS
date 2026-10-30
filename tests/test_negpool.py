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

"""build-negatives from scratch on a tiny genome, antiSMASH and IPS6 stubbed; both runners must agree."""

import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

from sanntis.config import NextflowOptions
from sanntis.negpool import build as B
from sanntis.negpool import strip
from sanntis.negpool.pool import FILES, PoolError, coverage_of, load_pool, validate

FIX = Path(__file__).parent / "fixtures" / "negpool"

FAKE_ANTISMASH = '''#!{python}
"""antiSMASH stand-in: copies the input GenBank and adds one region over CDS 5-9 of the first record."""
import sys
from pathlib import Path
from Bio import SeqIO
from Bio.SeqFeature import FeatureLocation, SeqFeature
a = sys.argv[1:]
if "--version" in a:
    print("antiSMASH 8.0.1")
    raise SystemExit(0)
out = Path(a[a.index("--output-dir") + 1]); base = a[a.index("--output-basename") + 1]
recs = list(SeqIO.parse(a[-1], "genbank"))
cds = sorted((f for f in recs[0].features if f.type == "CDS"), key=lambda f: int(f.location.start))
recs[0].features.append(SeqFeature(FeatureLocation(int(cds[5].location.start), int(cds[9].location.end)),
                                   type="region", qualifiers={{"product": ["terpene"]}}))
out.mkdir(parents=True, exist_ok=True)
SeqIO.write(recs, out / f"{{base}}.gbk", "genbank")
'''


def stub_ips6(faa: Path, outdir: Path, opts) -> Path:
    """IPS6-format TSV: every 3rd protein gets a Pfam + a Gene3D hit, every 5th an unintegrated Pfam."""
    md5s = [line[1:].strip() for line in Path(faa).read_text().splitlines() if line.startswith(">")]
    rows = []
    for i, m in enumerate(md5s):
        if i % 3 == 0:
            rows.append([m, m, "100", "Pfam", "PF00005", "d", "10", "80", "1e-5", "Pfam", "-", "IPR003439", "-", "-",
                         "-"])
            rows.append([m, m, "100", "CATH-Gene3D", "G3DSA:3.50.70.10", "d", "5", "90", "1e-5", "CATH-Gene3D", "-",
                         "-", "-", "-", "-"])
        if i % 5 == 0:
            rows.append([m, m, "100", "Pfam", "PF99999", "d", "1", "50", "1e-5", "Pfam", "-", "-", "-", "-", "-"])
    out = Path(outdir) / "results" / f"{Path(faa).name}.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join("\t".join(r) + "\n" for r in rows))
    return out


@pytest.fixture
def env(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("PYTHONPATH", f"{root / 'src'}")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    fake = bindir / "antismash"
    fake.write_text(FAKE_ANTISMASH.format(python=sys.executable))
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}:{__import__('os').environ['PATH']}")
    genomes = tmp_path / "genomes.tsv"
    genomes.write_text("genome\tkingdom\tlineage\tsource\taccession\tpath\n"
                       f"tiny\tproka\td__Bacteria;p__Test\tgtdb\tGCA_000220375.1\t{FIX / 'tiny_genome.fna.gz'}\n")
    entries = tmp_path / "entries.json"
    entries.write_text(json.dumps({"PF00005": {"database": "Pfam", "integrated": "IPR003439"},
                                   "G3DSA:3.50.70.10": {"database": "CATH-Gene3D", "integrated": "IPR016088"}}))
    return genomes, entries


def _build(tmp_path, env, runner: str, name: str):
    genomes, entries = env
    o = B.BuildOptions(out=tmp_path / name, pool_version="vtest", genomes=genomes, entries_json=entries,
                       runner=runner, workers=2, nextflow=NextflowOptions())
    return B.build_negatives(o, annotate=stub_ips6)


def _frames(pool):
    return {f: pd.read_parquet(pool.path / f) for f in FILES}


def test_build_inline(tmp_path, env):
    pool = _build(tmp_path, env, "inline", "inline")
    c = pd.read_parquet(pool.path / "cds_tokens.parquet")
    runs = pd.read_parquet(pool.path / "clean_runs.parquet")
    mask = pd.read_parquet(pool.path / "native_mask.parquet")
    assert len(c) > 100 and set(c["contig"]) == {"GL982570.1", "GL982571.1"}
    assert list(mask["cds_idx"]) == [5, 6, 7, 8, 9] and set(mask["contig"]) == {"GL982570.1"}
    assert runs["n_cds"].sum() == len(c) - len(mask)
    tok = c.set_index(["contig", "cds_idx"])["tokens"]
    assert any(list(t) == ["IPR016088", "IPR003439"] for t in tok) and any(list(t) == ["PF99999"] for t in tok)
    assert all(t is not None for t in tok)  # every CDS annotated
    bb = pd.read_parquet(pool.path / "test_backbones.parquet")
    assert len(bb) == 1 and bb.loc[0, "kingdom"] == "proka"  # only one genome: 1 backbone
    m = pool.manifest
    assert m["counts"]["proka"]["genomes"] == 1 and m["ips"]["interpro"] == "107.0"
    cov = m["coverage"]["proka"]
    assert cov["n_cds"] == len(c) and 0 < cov["frac_cds_with_tokens"] <= 1 and cov["hits_per_cds"]
    load_pool(pool.path)  # sha256s verify


@pytest.mark.nextflow
@pytest.mark.skipif(shutil.which("nextflow") is None, reason="nextflow not on PATH")
def test_runners_identical(tmp_path, env):
    a = _frames(_build(tmp_path, env, "inline", "inline"))
    nf_pool = _build(tmp_path, env, "nextflow", "nextflow")
    b = _frames(nf_pool)
    for f in FILES:
        pd.testing.assert_frame_equal(a[f], b[f], obj=f)
    shas = [json.loads((tmp_path / d / "vtest" / "manifest.json").read_text())["files"] for d in ("inline", "nextflow")]
    assert shas[0] == shas[1]
    assert (tmp_path / "nextflow" / "negpool_antismash" / "work").is_dir()


def test_clean_runs_and_backbones():
    cds = pd.DataFrame({"genome": "g", "contig": ["c1"] * 6 + ["c2"] * 3, "cds_idx": [0, 1, 2, 3, 4, 5, 0, 1, 2],
                        "start": [0, 100, 200, 300, 400, 500, 0, 100, 200],
                        "end": [90, 190, 290, 390, 490, 590, 90, 190, 290]})
    runs, mask = strip.clean_runs(cds, {"c1": [(150, 250)]})  # overlaps cds 1 and 2
    assert [r["cds_idx"] for r in runs] == [[0], [3, 4, 5], [0, 1, 2]]
    assert [r["run_id"] for r in runs] == ["c1_run0", "c1_run1", "c2_run0"]
    assert [m["cds_idx"] for m in mask] == [1, 2]
    rdf = pd.DataFrame(runs)
    g = pd.DataFrame({"genome": ["g"], "kingdom": ["proka"]})
    assert strip.test_backbones(rdf, g, min_cds=3).shape[0] == 1
    assert strip.test_backbones(rdf, g, min_cds=4).empty


def test_validate_rejects_gaps():
    frames = {
        "genomes.parquet": pd.DataFrame({"genome": ["g"], "kingdom": ["proka"], "lineage": [""], "source": ["x"],
                                         "accession": ["g"]}),
        "cds_tokens.parquet": pd.DataFrame({"genome": "g", "contig": "c", "cds_idx": [0, 1, 2], "cds_id": "x",
                                            "start": 0, "end": 1, "strand": 1, "md5": "m", "tokens": [[], [], []]}),
        "clean_runs.parquet": pd.DataFrame({"genome": ["g"], "contig": ["c"], "run_id": ["r"], "cds_idx": [[0, 2]],
                                            "n_cds": [2]}),
        "native_mask.parquet": pd.DataFrame(columns=["genome", "contig", "cds_idx"]),
        "test_backbones.parquet": pd.DataFrame(columns=["genome", "contig", "run_id", "kingdom"]),
    }
    with pytest.raises(PoolError, match="consecutive"):
        validate(frames)


def test_dry_run_prints_commands(tmp_path, env, capsys):
    genomes, entries = env
    o = B.BuildOptions(out=tmp_path / "dry", genomes=genomes, entries_json=entries, runner="nextflow", dry_run=True,
                       nextflow=NextflowOptions(container_engine="singularity"))
    assert B.build_negatives(o) is None
    out = capsys.readouterr().out
    assert out.count("tasks.nf") == 2 and "-profile singularity" in out


def test_coverage_counts_per_cds():
    cds = pd.DataFrame({"genome": ["g1", "g1", "g1", "g1", "g1", "g2"], "md5": ["a", "a", "b", "c", "d", "a"],
                        "tokens": [["X"], ["X"], [], ["Y"], None, ["X"]]})
    hits = pd.Series({("a", "pfam"): 2, ("a", "smart"): 1, ("c", "pfam"): 1, ("z", "pfam"): 5})
    hits.index.names = ["md5", "app"]
    cov = coverage_of(cds, hits, {"g1": "proka", "g2": "euk"})
    # d is unannotated (left out); a counts twice (two CDS share it); z is not in the pool
    assert cov["proka"] == {"n_cds": 4, "frac_cds_with_tokens": 0.75, "hits_per_cds": {"pfam": 1.25, "smart": 0.5}}
    assert cov["euk"] == {"n_cds": 1, "frac_cds_with_tokens": 1.0, "hits_per_cds": {"pfam": 2.0, "smart": 1.0}}


def test_qc_flags_per_kingdom():
    from sanntis.qc import flags

    qc = {"n_cds": 100, "n_cds_with_tokens": 90, "frac_cds_with_tokens": 0.9, "frac_unk_tokens": 0.1,
          "hits_per_cds": {"mobidblite": 0.08}}
    ref = {"source": "negative pool", "proka": {"hits_per_cds": {"mobidblite": 0.1}},
           "euk": {"hits_per_cds": {"mobidblite": 1.5}}}
    assert flags(qc, ref, ["mobidblite"], "proka") == []
    assert flags(qc, ref, ["mobidblite"], "euk") == ["low_mobidblite_hits"]
    assert flags(qc, {"hits_per_cds": {"mobidblite": 1.5}}, ["mobidblite"]) == ["low_mobidblite_hits"]  # flat
    assert flags({**qc, "n_cds": 25, "n_cds_with_tokens": 24}, ref, ["mobidblite"], "euk") == []  # one BGC


def test_bad_genome_id_rejected(tmp_path):
    gt = tmp_path / "genomes.tsv"
    gt.write_text("genome\tkingdom\nok_1\tproka\nbad id$(x)\tproka\n")
    with pytest.raises(B.BuildError, match="genome ids"):
        B.build_negatives(B.BuildOptions(out=tmp_path / "out", genomes=gt, dry_run=True))
