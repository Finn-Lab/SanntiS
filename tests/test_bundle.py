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

import json
import shutil

import numpy as np
import pytest
import torch

from sanntis.model import bundle as bundle_mod
from sanntis.model.bundle import BundleError, IpsMismatchError, SanntisModel
from sanntis.model.detector import predict_tokens

from .conftest import random_regions


def _outputs(m: SanntisModel):
    rng = np.random.default_rng(1)
    toks = rng.integers(0, len(m.vocab), size=450)
    det = [predict_tokens(d, toks, m.config.detector.window) for d in m.detectors()]
    regions = random_regions(rng, m.vocab.tokens + ["X"], 7)
    pf = m.postfilter.classify(regions, m.config.postfilter_drop_threshold)
    npc = m.npc.predict(regions, "proka") if m.npc is not None else None
    return det, pf, npc


def test_local_round_trip(dummy_model, dummy_bundle):
    loaded = SanntisModel.from_pretrained(dummy_bundle)
    assert loaded.vocab.itos == dummy_model.vocab.itos
    assert loaded.config.to_json() == dummy_model.config.to_json()
    a, b = _outputs(dummy_model), _outputs(loaded)
    for x, y in zip(a[0], b[0], strict=True):
        np.testing.assert_array_equal(x, y)
    np.testing.assert_array_equal(a[1][2], b[1][2])
    assert a[1][1] == b[1][1]
    assert a[2].equals(b[2])
    # saving the loaded model again gives identical files
    again = loaded.save_pretrained(dummy_bundle.parent / "again")
    m1 = json.loads((dummy_bundle / "manifest.json").read_text())["files"]
    m2 = json.loads((again / "manifest.json").read_text())["files"]
    # skops (zip metadata) and safetensors (header metadata order) are not byte-stable across saves
    stable = (".skops", ".safetensors")
    assert {k: v for k, v in m1.items() if not k.endswith(stable)} == {
        k: v for k, v in m2.items() if not k.endswith(stable)}
    assert set(m1) == set(m2)
    c = _outputs(SanntisModel.from_pretrained(again))
    assert a[2].equals(c[2]) and a[1][1] == c[1][1]


def test_manifest_lists_every_file(dummy_bundle):
    files = json.loads((dummy_bundle / "manifest.json").read_text())["files"]
    on_disk = {str(p.relative_to(dummy_bundle)) for p in dummy_bundle.rglob("*") if p.is_file()}
    assert on_disk - set(files) == {"manifest.json", "README.md"}
    assert not any(n.endswith((".pt", ".pkl", ".joblib")) for n in on_disk)


def test_tampered_file_fails(dummy_bundle):
    with open(dummy_bundle / "vocab.json", "a") as fh:
        fh.write(" ")
    with pytest.raises(BundleError, match="sha256"):
        SanntisModel.from_pretrained(dummy_bundle)


def test_missing_file_fails(dummy_bundle):
    (dummy_bundle / "detector-seed1.safetensors").unlink()
    with pytest.raises(BundleError, match="missing"):
        SanntisModel.from_pretrained(dummy_bundle)


def test_schema_mismatch_fails(dummy_bundle):
    mf = dummy_bundle / "manifest.json"
    d = json.loads(mf.read_text())
    d["schema_version"] = 99
    mf.write_text(json.dumps(d))
    with pytest.raises(BundleError, match="schema"):
        SanntisModel.from_pretrained(dummy_bundle)


def test_ips_mismatch(dummy_bundle, caplog):
    mf = dummy_bundle / "manifest.json"
    d = json.loads(mf.read_text())
    d["ips"]["version"] = "5.76-107.0"
    mf.write_text(json.dumps(d))
    with pytest.raises(IpsMismatchError):
        SanntisModel.from_pretrained(dummy_bundle)
    SanntisModel.from_pretrained(dummy_bundle, allow_ips_mismatch=True)
    assert "differs" in caplog.text


def test_hub_requires_revision():
    with pytest.raises(BundleError, match="revision"):
        SanntisModel.from_pretrained("someorg/somerepo")


def test_mocked_hub_round_trip(dummy_bundle, tmp_path, monkeypatch):
    """Hub path: snapshot_download is mocked by a local HF-style cache (no network, nothing published)."""
    import huggingface_hub

    calls = []

    def fake_snapshot_download(repo_id, revision=None, cache_dir=None, **kw):
        calls.append((repo_id, revision))
        snap = tmp_path / "hf" / f"models--{repo_id.replace('/', '--')}" / "snapshots" / revision
        if not snap.exists():
            shutil.copytree(dummy_bundle, snap)
        return str(snap)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    m = SanntisModel.from_pretrained("microbiome-informatics/sanntis@v0.0.1-test")
    assert calls == [("microbiome-informatics/sanntis", "v0.0.1-test")]
    assert m.source == "microbiome-informatics/sanntis@v0.0.1-test"
    m2 = SanntisModel.from_pretrained("microbiome-informatics/sanntis", revision="v0.0.1-test")
    assert m2.vocab.itos == m.vocab.itos
    assert all(torch.equal(m.detector_states[0][k], v) for k, v in m2.detector_states[0].items())


def test_default_model_id():
    """The default model tag names the installed release (v1.0.0-beta.1 ↔ 1.0.0b1[.devN])."""
    from packaging.version import Version

    from sanntis import __version__

    tag = Version(bundle_mod.DEFAULT_REVISION.removeprefix("v"))
    assert bundle_mod.default_model_id() == f"microbiome-informatics/sanntis@{bundle_mod.DEFAULT_REVISION}"
    assert tag.release == Version(__version__).release and tag.pre == Version(__version__).pre


def test_bundle_without_npc(tmp_path):
    from .conftest import make_dummy_model

    m = make_dummy_model(npc=False)
    out = m.save_pretrained(tmp_path / "b")
    assert not (out / "npc.skops").exists()
    assert SanntisModel.from_pretrained(out).npc is None


def test_report_replaced_not_merged(dummy_model, tmp_path):
    rep = tmp_path / "rep"
    (rep / "figures").mkdir(parents=True)
    (rep / "figures" / "old.png").write_bytes(b"old")
    dummy_model.report_dir = rep
    out = dummy_model.save_pretrained(tmp_path / "b")
    shutil.rmtree(rep / "figures")
    (rep / "report.md").write_text("new")
    dummy_model.save_pretrained(out)
    files = json.loads((out / "manifest.json").read_text())["files"]
    assert not (out / "report" / "figures").exists() and "report/report.md" in files
    assert not any("old.png" in f for f in files)


def test_unlisted_detector_file_fails(dummy_bundle):
    """A weights file that the manifest does not list is not loaded silently."""
    shutil.copy(dummy_bundle / "detector-seed0.safetensors", dummy_bundle / "detector-seed9.safetensors")
    with pytest.raises(BundleError, match="not listed"):
        SanntisModel.from_pretrained(dummy_bundle)
