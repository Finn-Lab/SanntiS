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

import io
import json
from pathlib import Path

import pandas as pd
import pytest
from Bio import SeqIO
from Bio.SeqFeature import FeatureLocation, SeqFeature

from sanntis.train import collect as C
from sanntis.train import labels, mibig
from sanntis.train.orchestrate import RunContext, StepError, build_config, run
from sanntis.train.steps import STEPS

from .train_fixture import entry, taxonomy_cache, write_tar

FIX = Path(__file__).parent / "fixtures" / "mibig"
ACCS = ["BGC0002020", "BGC0002195", "BGC0002899"]
PK = {"pathway_results": ["Polyketides"], "superclass_results": ["Macrolides"], "class_results": ["Not a term"]}


class FakeNPC:
    """NPClassifier stand-in: every structure is a polyketide macrolide."""

    def __init__(self, answer=PK):
        self.answer, self.requests, self.seen = answer, 0, []

    def classify(self, smiles):
        self.requests += 1
        self.seen.append(smiles)
        return self.answer

    def model_metadata(self):
        return "models fake"


def _entries() -> dict[str, dict]:
    return {a: json.loads((FIX / f"{a}.json").read_text()) for a in ACCS}


def _archives(d: Path, stem: str, entries: dict[str, dict], gbks: dict[str, bytes]) -> tuple[Path, Path]:
    d.mkdir(parents=True, exist_ok=True)
    j = write_tar(d / f"{stem}_json.tar.gz",
                  {f"{stem}_json/{a}.json": json.dumps(e).encode() for a, e in entries.items()})
    g = write_tar(d / f"{stem}_gbk.tar.gz", {f"{stem}_gbk/{a}.gbk": b for a, b in gbks.items()})
    return g, j


def _opts(tmp_path, **kw) -> C.CollectOptions:
    cache = taxonomy_cache(tmp_path / "cache")
    return C.CollectOptions(taxonomy_cache=cache / "ncbi" / "lineage.json",
                            npc_client=FakeNPC(), **kw)


@pytest.fixture
def release(tmp_path):
    """MIBiG release 'test' already in the archive cache: 3 real loci + 1 active entry without a GBK."""
    entries = _entries()
    extra = json.loads((FIX / "BGC0002020.json").read_text())
    extra.update(accession="BGC0009999", status="active",
                 loci=[{"accession": "XX000001.1", "location": {"from": 1, "to": 4504}}])
    entries["BGC0009999"] = extra
    _archives(tmp_path / "archives", "mibig", entries, {a: (FIX / f"{a}.gbk").read_bytes() for a in ACCS})
    for k in ("json", "gbk"):
        (tmp_path / "archives" / f"mibig_{k}.tar.gz").rename(tmp_path / "archives" / f"mibig_{k}_test.tar.gz")
    return _opts(tmp_path, mibig="test", archive_root=tmp_path / "archives")


def test_collect_mibig(tmp_path, release):
    counts = C.collect(tmp_path / "out", release)
    loci = pd.read_parquet(tmp_path / "out" / "loci.parquet").set_index("entity_id")
    assert sorted(loci.index) == ["BGC0002020", "BGC0002195"]  # pending BGC0002899 filtered, BGC0009999 no GBK
    assert loci.loc["BGC0002195", "kingdom"] == "euk" and loci.loc["BGC0002020", "kingdom"] == "proka"
    assert loci.loc["BGC0002020", "class_raw"] == "NRPS;saccharide"
    assert loci.loc["BGC0002020", "primary_class"] == "NRPS"
    ex = pd.read_csv(tmp_path / "out" / "exclusions.tsv", sep="\t").set_index("entity_id")
    assert ex.loc["BGC0009999", "reason"] == "gbk: not in the MIBiG GBK archive"
    assert set(loci["source"]) == {"mibig"} and set(loci["status"]) == {"active"}
    cds = pd.read_parquet(tmp_path / "out" / "cds.parquet")
    assert (cds.groupby("entity_id").size() == loci["n_cds"]).all()
    faa = (tmp_path / "out" / "proteins.faa").read_text()
    assert faa.count(">") == counts["proteins"] == cds["md5"].nunique()
    assert len(pd.read_csv(tmp_path / "out" / "kingdom_map.tsv", sep="\t")) == 2
    # NPC labels from the compounds; unknown API terms dropped
    assert loci.loc["BGC0002195", "npc_status"] == "classified"
    assert list(loci.loc["BGC0002195", "npc_pathway"]) == ["Polyketides"]
    assert list(loci.loc["BGC0002195", "npc_superclass"]) == ["Macrolides"]
    prov = json.loads((tmp_path / "out" / "provenance.json").read_text())
    assert prov["mibig"]["release"] == "test" and prov["mibig"]["statuses"] == ["active"]
    assert prov["mibig"]["json"]["name"] == "mibig_json_test.tar.gz"
    assert prov["npc_labels"]["unmapped_terms"] == {"class:Not a term": prov["npc_labels"]["skeletons"]}
    comps = pd.read_csv(tmp_path / "out" / "npc_compounds.tsv", sep="\t")
    assert set(comps["accession"]) == {"BGC0002020", "BGC0002195"}


def test_collect_mibig_exclusions(tmp_path, release):
    entries = _entries()
    entries["BGC0009999"] = {**entries["BGC0002020"], "accession": "BGC0009999"}
    del entries["BGC0002020"]["taxonomy"]
    gbks = {a: (FIX / f"{a}.gbk").read_bytes() for a in ACCS}
    _archives(tmp_path / "archives", "mibig", entries, {**gbks, "BGC0009999": gbks["BGC0002020"]})
    for k in ("json", "gbk"):
        (tmp_path / "archives" / f"mibig_{k}.tar.gz").replace(tmp_path / "archives" / f"mibig_{k}_test.tar.gz")
    release.max_locus_bp = 5000
    C.collect(tmp_path / "out", release)
    ex = pd.read_csv(tmp_path / "out" / "exclusions.tsv", sep="\t").set_index("entity_id")
    assert "taxonomy.ncbiTaxId: missing" in ex.loc["BGC0002020", "reason"]  # invalid MIBiG entry: excluded
    assert "6136 > 5000" in ex.loc["BGC0002195", "reason"] and "BGC0002899" not in ex.index  # pending: not used
    assert pd.read_parquet(tmp_path / "out" / "loci.parquet")["entity_id"].tolist() == ["BGC0009999"]
    release.max_locus_bp = 10
    with pytest.raises(C.CollectError, match="no positive loci"):
        C.collect(tmp_path / "out", release)


def test_release_url(tmp_path, monkeypatch):
    calls = []

    def fake_fetch(url, dest, sha256=None):
        calls.append(url)
        if "gbk" in url:
            raise OSError("HTTP Error 404")
        return dest

    monkeypatch.setattr(mibig, "fetch", fake_fetch)
    with pytest.raises(mibig.FormatError, match="mibig_gbk_5.0.tar.gz"):
        mibig.release_archives("v5.0", tmp_path)
    assert calls == [mibig.BASE_URL + "mibig_json_5.0.tar.gz", mibig.BASE_URL + "mibig_gbk_5.0.tar.gz"]


def _gbk_with_taxon(taxid: int) -> bytes:
    rec = SeqIO.read(FIX / "BGC0002195.gbk", "genbank")
    rec.features.insert(0, SeqFeature(FeatureLocation(0, len(rec.seq)), type="source",
                                      qualifiers={"db_xref": [f"taxon:{taxid}"]}))
    fh = io.StringIO()
    SeqIO.write([rec], fh, "genbank")
    return fh.getvalue().encode()


def test_collect_external(tmp_path):
    e = _entries()["BGC0002195"]
    raw = (FIX / "BGC0002195.gbk").read_bytes()
    g, j = _archives(tmp_path / "ext", "mine", {"BGC0002195": e, "my1": {**e, "accession": "my1"}},
                     {"BGC0002195": raw, "my1": raw})
    g2 = write_tar(tmp_path / "ext" / "only_gbk.tar.gz", {"only/gbkonly.gbk": _gbk_with_taxon(510516)})
    opts = _opts(tmp_path, external=[(g, j), (g2, None)])
    counts = C.collect(tmp_path / "out", opts)
    loci = pd.read_parquet(tmp_path / "out" / "loci.parquet").set_index("entity_id")
    assert sorted(loci.index) == ["BGC0002195", "gbkonly", "my1"] and set(loci["source"]) == {"external"}
    assert loci.loc["my1", "kingdom"] == "euk" and loci.loc["my1", "class_raw"] == "other"
    assert loci.loc["my1", "npc_status"] == "classified"
    # GBK only: detector only (no class, no NPC labels); kingdom from the source taxon xref
    assert loci.loc["gbkonly", "class_raw"] == "" and loci.loc["gbkonly", "npc_status"] == "unlabelled"
    assert loci.loc["gbkonly", "kingdom"] == "euk" and loci.loc["gbkonly", "ncbi_tax_id"] == 510516
    assert opts.npc_client.requests == 1  # same skeleton for both labelled BGCs
    prov = json.loads((tmp_path / "out" / "provenance.json").read_text())
    assert "mibig" not in prov and [x["n"] for x in prov["external"]] == [2, 1]
    assert prov["external"][1]["json"] is None and counts["by_source"] == {"external": 3}


def test_collect_external_problems(tmp_path, release):
    es = _entries()
    bad = {**es["BGC0002195"], "accession": "bad1", "biosynthesis": {"classes": [{"class": "lasso"}]}}
    del bad["taxonomy"]
    entries = {"BGC0002020": es["BGC0002020"], "bad1": bad, "nojson_gbk": None, "nogbk": {**bad, "accession": "nogbk"},
               "notr": {**es["BGC0002195"], "accession": "notr"}}
    gbk = (FIX / "BGC0002195.gbk").read_bytes()
    g, j = _archives(tmp_path / "ext", "x", {a: e for a, e in entries.items() if e},
                     {"BGC0002020": (FIX / "BGC0002020.gbk").read_bytes(), "bad1": gbk, "nojson_gbk": gbk,
                      "notr": _strip_translations(gbk.decode()).encode()})
    release.external = [(g, j)]
    with pytest.raises(C.CollectError) as e:
        C.collect(tmp_path / "out", release)
    msg = str(e.value)
    for s in ("nogbk: JSON entry without a GBK", "nojson_gbk: GBK without a JSON entry",
              "BGC0002020: accession already used", "bad1: biosynthesis.classes: unknown class ['lasso']",
              "bad1: taxonomy.ncbiTaxId: missing", "notr: gbk:", "has no /translation"):
        assert s in msg, s


def test_collect_needs_positives(tmp_path):
    with pytest.raises(C.CollectError, match="no positives"):
        C.collect(tmp_path / "out", _opts(tmp_path))


def _strip_translations(text: str) -> str:
    out, skip = [], False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("/translation="):
            skip = not s.endswith('"') or s.count('"') < 2
            continue
        if skip:
            skip = not s.endswith('"')
            continue
        out.append(line)
    return "\n".join(out) + "\n"


# ── NPC labels ───────────────────────────────────────────────────────────────


def test_npc_labels_from_compounds():
    salt = "[Na+].[O-]C(=O)c1ccccc1O"  # salicylate salt → same skeleton as salicylic acid
    entries = {"a": entry("a", "PKS", "proka", 100, [{"name": "sal", "structure": "OC(=O)c1ccccc1O"},
                                                      {"name": "salt", "structure": salt}]),
               "b": entry("b", "PKS", "proka", 100, [{"name": "x"}, {"name": "y", "structure": "not smiles"}]),
               "c": entry("c", "PKS", "proka", 100, [{"name": "z", "structure": "CCO"}])}
    client = FakeNPC()
    client.answer = None
    bgc, comps, prov = labels.label_entries(entries, client)
    assert bgc.loc["a", "npc_status"] == "no_npc_term" and bgc.loc["b", "npc_status"] == "no_structure"
    assert client.requests == 2 and prov["skeletons"] == 2  # one request per skeleton
    client = FakeNPC()
    bgc, comps, prov = labels.label_entries(entries, client)
    assert bgc.loc["a", "npc_status"] == "classified" and bgc.loc["a", "pathway"] == ["Polyketides"]
    assert bgc.loc["a", "npc_class"] == []  # "Not a term" is not in the frozen vocabulary
    st = comps.set_index("compound_name")["status"]
    assert st["salt"] == "classified" and st["y"] == "smiles_parse_failed" and st["x"] == "no_structure"
    assert prov["classifier_version"] == "models fake" and prov["ontology"]["version"] == labels.ONTOLOGY_VERSION


def _cache_record(d: Path, key: str, body: str) -> Path:
    import hashlib

    f = d / f"{hashlib.sha1(key.encode()).hexdigest()}.json"
    f.write_text(json.dumps({"key": key, "status": 200, "body": body, "fetched_at": "2026-01-01T00:00:00+00:00"}))
    return f


def test_seed_cache_dir(tmp_path):
    src, dst = tmp_path / "published", tmp_path / "cache"
    src.mkdir()
    _cache_record(src, "CCO", '{"pathway_results": ["A"]}')
    _cache_record(src, "CCN", '{"pathway_results": ["B"]}')
    dst.mkdir()
    kept = _cache_record(dst, "CCO", '{"pathway_results": ["local"]}')  # an existing answer is never replaced
    prov = labels.seed_cache(src, dst)
    assert prov == {"source": str(src), "version": None, "files": 2, "added": 1}
    assert json.loads(kept.read_text())["body"] == '{"pathway_results": ["local"]}'
    c = labels.NPClassifierClient(dst)
    assert c.classify("CCN") == {"pathway_results": ["B"]} and c.requests == 0
    assert labels.seed_cache(src, dst)["added"] == 0
    # only genuine cache records enter the cache
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "x.json").write_text(json.dumps({"key": "CCO", "status": 200, "body": "{}"}))
    with pytest.raises(labels.LabelError, match="sha1"):
        labels.seed_cache(bad, dst)
    (bad / "x.json").write_text("[]")
    with pytest.raises(labels.LabelError, match="not a cache record"):
        labels.seed_cache(bad, dst)
    with pytest.raises(labels.LabelError, match="no NPClassifier cache files"):
        labels.seed_cache(tmp_path / "empty", dst) if (tmp_path / "empty").mkdir() is None else None


def test_seed_cache_hub(tmp_path, monkeypatch):
    import huggingface_hub

    from sanntis.negpool.pool import PoolError, hub_dataset_dir

    snap = tmp_path / "snapshot"
    (snap / "npclassifier" / "v1").mkdir(parents=True)
    _cache_record(snap / "npclassifier" / "v1", "CCO", "{}")
    calls = []

    def fake_snapshot_download(**kw):
        calls.append(kw)
        return str(snap)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    monkeypatch.delenv("SANNTIS_DATA_REPO", raising=False)
    prov = labels.seed_cache("v1", tmp_path / "cache")
    assert prov["version"] == "v1" and prov["added"] == 1
    assert calls[-1] == {"repo_id": "microbiome-informatics/sanntis-training-data", "repo_type": "dataset",
                         "revision": "npclassifier-v1", "allow_patterns": ["npclassifier/v1/*"]}
    assert hub_dataset_dir("org/data@v2", "negatives")[1] == "v2"
    assert calls[-1]["repo_id"] == "org/data" and calls[-1]["revision"] == "negatives-v2"
    with pytest.raises(PoolError, match="not a directory"):
        hub_dataset_dir("no/such/dir", "negatives")


def test_collect_external_seeded_cache(tmp_path):
    """--npc-cache: the published answers are copied into the client's cache before labelling."""
    src = tmp_path / "published"
    src.mkdir()
    _cache_record(src, "CCO", "{}")
    e = _entries()["BGC0002195"]
    g, j = _archives(tmp_path / "ext", "mine", {"BGC0002195": e}, {"BGC0002195": (FIX / "BGC0002195.gbk").read_bytes()})
    opts = _opts(tmp_path, external=[(g, j)], npc_cache=str(src))
    opts.npc_client.dir = tmp_path / "npc"
    C.collect(tmp_path / "out", opts)
    prov = json.loads((tmp_path / "out" / "provenance.json").read_text())
    assert prov["npc_labels"]["cache_seed"] == {"source": str(src), "version": None, "files": 1, "added": 1}
    assert (tmp_path / "npc").is_dir() and len(list((tmp_path / "npc").glob("*.json"))) == 1


def test_npc_ontology():
    parents = labels.superclass_parents()
    assert "Polyketides" in parents["Macrolides"] and len(parents) > 50
    assert ("pathway", "Terpenoids") in labels.vocabulary()


def test_npc_client_cache(tmp_path, monkeypatch):
    import urllib.error
    import urllib.request

    calls = []

    class Resp(io.BytesIO):
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None):
        calls.append(req.full_url)
        if "boom" in req.full_url:
            raise urllib.error.HTTPError(req.full_url, 500, "err", {}, io.BytesIO(b"server error"))
        return Resp(json.dumps(PK).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(labels.time, "sleep", lambda s: None)
    c = labels.NPClassifierClient(tmp_path / "npc", min_interval=0)
    assert c.classify("CCO")["pathway_results"] == ["Polyketides"]
    assert labels.NPClassifierClient(tmp_path / "npc").classify("CCO") == c.classify("CCO")
    assert len(calls) == 1  # cached
    assert c.classify("boom") is None and len(calls) == 4  # 500: retried, then cached as a failure
    assert c.classify("boom") is None and len(calls) == 4


# ── config / orchestration ───────────────────────────────────────────────────


def test_train_config():
    from pydantic import ValidationError

    from sanntis.config import TrainConfig

    for v in (None, False, "none", "False", ""):
        assert TrainConfig(run_dir="r", mibig=v).mibig is None
    assert TrainConfig(run_dir="r", mibig=4.0).mibig == "4.0"
    c = TrainConfig(run_dir="r", external_gbk=["a.tar.gz", "b.tar.gz"])
    assert c.external == [(Path("a.tar.gz"), None), (Path("b.tar.gz"), None)]
    assert TrainConfig(run_dir="r", npc_cache="v1").npc_cache == "v1" and c.npc_cache is None
    with pytest.raises(ValidationError, match="one per GBK archive"):
        TrainConfig(run_dir="r", external_gbk=["a.tar.gz", "b.tar.gz"], external_json=["a.tar.gz"])


def test_run_yaml_removed_options(tmp_path):
    y = tmp_path / "run.yaml"
    y.write_text("run_dir: r\nmibig: '4.0'\ncustom: sheet.csv\n")
    with pytest.raises(StepError, match="custom"):
        build_config({"config_file": str(y)})
    y.write_text("run_dir: r\nmibig: '4.0'\n")
    cfg = build_config({"config_file": str(y), "mibig": "none"})
    assert cfg.mibig is None  # explicit CLI value wins
    y.write_text("run_dir: r\nrunner: inline\nworkers: 4\nentries_json: e.json\n")
    cfg = build_config({"config_file": str(y), "runner": None, "workers": None, "gpu": False, "train_ips_tsv": ()})
    assert (cfg.runner, cfg.workers, cfg.entries_json) == ("inline", 4, Path("e.json"))  # CLI unset: run.yaml kept
    assert build_config({"config_file": str(y), "workers": 2}).workers == 2


def _fixture_archives(tmp_path, cls="PKS"):
    e = entry("my1", cls, "euk", 100, [])
    return _archives(tmp_path / "ext", "fx", {"my1": e}, {"my1": (FIX / "BGC0002195.gbk").read_bytes()})


def test_orchestrate_resume(tmp_path, monkeypatch):
    from sanntis.config import TrainConfig

    monkeypatch.setenv("SANNTIS_CACHE", str(taxonomy_cache(tmp_path / "cache")))
    g, j = _fixture_archives(tmp_path)
    ctx = RunContext(TrainConfig(run_dir=tmp_path / "run", external_gbk=[g], external_json=[j], runner="inline"))
    assert run(ctx, STEPS, until="collect")["collect"]["status"] == "ran"
    assert run(ctx, STEPS, until="collect")["collect"]["status"] == "skipped"
    assert run(ctx, STEPS, from_step="collect", until="collect")["collect"]["status"] == "ran"
    _fixture_archives(tmp_path, "NRPS")  # input changed → re-run
    assert run(ctx, STEPS, until="collect")["collect"]["status"] == "ran"
    done = json.loads((tmp_path / "run" / "collect" / "done.json").read_text())
    assert done["outputs"]["loci"] == 1
    loci = pd.read_parquet(tmp_path / "run" / "collect" / "loci.parquet")
    assert loci["class_raw"].tolist() == ["NRPS"]


def test_train_cli(tmp_path, monkeypatch):
    from click.testing import CliRunner

    from sanntis.cli import main

    monkeypatch.setenv("SANNTIS_CACHE", str(taxonomy_cache(tmp_path / "cache")))
    g, j = _fixture_archives(tmp_path)
    args = ["train", "--run", str(tmp_path / "run"), "--external-gbk", str(g), "--external-json", str(j),
            "--runner", "inline", "--until", "collect"]
    r = CliRunner().invoke(main, [*args, "--dry-run"])
    assert r.exit_code == 0, r.output
    assert "would run" in r.output and not (tmp_path / "run" / "collect" / "done.json").exists()
    r = CliRunner().invoke(main, args)
    assert r.exit_code == 0, r.output
    assert "collect: ran" in r.output
    assert "collect: skipped" in CliRunner().invoke(main, args).output
    for removed in ("--custom", "--labels", "--evaluate", "--mibig-status"):
        assert CliRunner().invoke(main, ["train", "--run", "r", removed, "x"]).exit_code == 2
