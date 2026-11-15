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

"""No-domain filter: regions whose tokens are only COILS / MobiDB-lite hits (or none) are dropped."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from sanntis.cli import main
from sanntis.features.vocab import Vocab
from sanntis.predict import (
    DROP_NO_DOMAIN,
    DROPPED_COLUMNS,
    NON_DOMAIN_TOKENS,
    REGION_COLUMNS,
    Predictor,
    has_domain_token,
    non_domain_tokens,
)

FIX = Path(__file__).parent / "fixtures"


def test_has_domain_token():
    stoi = Vocab(["IPR000001", "mobidb-lite", "Coil"]).stoi
    nd = NON_DOMAIN_TOKENS
    assert has_domain_token([["mobidb-lite"], ["IPR000001"]], stoi, nd)
    assert not has_domain_token([["mobidb-lite"], ["mobidb-lite", "Coil"], []], stoi, nd)
    assert not has_domain_token([["IPR999999"]], stoi, nd)  # out of the vocabulary: the model never sees it
    assert not has_domain_token([[], []], stoi, nd)


def test_non_domain_tokens_from_signature_map():
    sm = pd.DataFrame({"signature": ["mobidb-lite", "Coil", "PF00001", "MOBIDB-X"],
                       "app": ["mobidblite", "coils", "pfam", "mobidblite"], "entry": [None, None, "IPR000001", None]})
    assert non_domain_tokens(sm) == NON_DOMAIN_TOKENS | {"MOBIDB-X"}
    assert non_domain_tokens(None) == NON_DOMAIN_TOKENS


def _cds(records: dict[str, list[list[str]]]) -> pd.DataFrame:
    rows = [{"record_id": rid, "cds_idx": i, "locus_tag": f"{rid}_{i}", "start": i * 1000, "end": i * 1000 + 900,
             "tokens": t} for rid, toks in records.items() for i, t in enumerate(toks)]
    return pd.DataFrame(rows)


@pytest.fixture
def whole_record_regions(monkeypatch):
    """Every record is one region covering all its CDS, kept by the post-filter."""
    def predict_record(self, cds):
        n = len(cds)
        return np.full(n, 0.9), [(0, n, 0.9, "PKS", 0.758)]
    monkeypatch.setattr(Predictor, "predict_record", predict_record)


def _predictor(model, **kw):
    """Predictor on the dummy bundle, whose last two vocabulary tokens stand in for the COILS / MobiDB-lite tokens."""
    p = Predictor(model, **kw)
    dis, coil = model.vocab.tokens[-2:]
    p.non_domain = p.non_domain | {dis, coil}
    return p, dis, coil


def test_predictor_drops_and_renumbers(dummy_model, whole_record_regions):
    t = dummy_model.vocab.tokens
    p, dis, coil = _predictor(dummy_model)
    cds = _cds({"disorder": [[dis], [dis, coil]], "real": [[t[0], dis], [t[2]]], "unknown": [["UNSEEN1"], []],
                "real2": [[t[3]]]})
    pred = p.predict(cds)
    assert pred.regions["record_id"].tolist() == ["real", "real2"] and pred.regions["region"].tolist() == [1, 1]
    assert list(pred.regions.columns) == REGION_COLUMNS and list(pred.dropped.columns) == DROPPED_COLUMNS
    assert pred.dropped["record_id"].tolist() == ["disorder", "unknown"]
    assert set(pred.dropped["drop_reason"]) == {DROP_NO_DOMAIN}
    assert pred.filters == {"domain_filter": True, "regions_before": 4, DROP_NO_DOMAIN: 2}


def test_region_numbers_restart_per_record(dummy_model, monkeypatch):
    t = dummy_model.vocab.tokens
    p, dis, _ = _predictor(dummy_model)

    def predict_record(self, cds):  # three regions per record: domain, disorder only, domain
        return np.full(len(cds), 0.9), [(0, 1, 0.9, "PKS", 0.1), (1, 2, 0.9, "PKS", 0.758), (2, 3, 0.9, "PKS", 0.2)]
    monkeypatch.setattr(Predictor, "predict_record", predict_record)
    pred = p.predict(_cds({"r": [[t[0]], [dis], [t[1]]]}))
    assert pred.regions["region"].tolist() == [1, 2] and pred.regions["start_orf"].tolist() == [0, 2]


def test_filter_off_keeps_every_region(dummy_model, whole_record_regions):
    p, dis, _ = _predictor(dummy_model, domain_filter=False)
    pred = p.predict(_cds({"disorder": [[dis]], "empty": [[]]}))
    assert len(pred.regions) == 2 and pred.dropped.empty and pred.filters["domain_filter"] is False


def _run(*args):
    r = CliRunner().invoke(main, ["predict", *map(str, args)], catch_exceptions=False)
    assert r.exit_code == 0, r.output


def test_cli_outputs(tmp_path, dummy_bundle):
    gbk, tsv = FIX / "BGC0003066.gbk", FIX / "ips576.tsv"
    on, off = tmp_path / "on", tmp_path / "off"
    _run(gbk, "--ips-tsv", tsv, "--model", dummy_bundle, "-o", on)
    _run(gbk, "--ips-tsv", tsv, "--model", dummy_bundle, "-o", off, "--no-domain-filter")
    meta = json.loads((on / "BGC0003066.run.json").read_text())
    kept = pd.read_csv(on / "BGC0003066.regions.tsv", sep="\t")
    dropped = pd.read_csv(on / "BGC0003066.dropped_regions.tsv", sep="\t")
    allr = pd.read_csv(off / "BGC0003066.regions.tsv", sep="\t")
    assert list(dropped.columns) == DROPPED_COLUMNS
    assert meta["counts"]["regions"] == len(kept) and meta["counts"]["dropped_regions"] == len(dropped)
    assert meta["filters"]["regions_before"] == len(allr) == len(kept) + len(dropped)
    assert sorted(zip(kept["start"], kept["end"], strict=True)) == sorted(
        set(zip(allr["start"], allr["end"], strict=True)) - set(zip(dropped["start"], dropped["end"], strict=True)))
    header = [ln for ln in (on / "BGC0003066.sanntis.gff3").read_text().splitlines() if ln.startswith("##")]
    assert any(ln.startswith("##domain-filter on dropped=") for ln in header)
    assert "##domain-filter off" in (off / "BGC0003066.sanntis.gff3").read_text()
    assert json.loads((off / "BGC0003066.run.json").read_text())["filters"]["domain_filter"] is False
    # consumers glob *.regions.tsv for the kept regions: the dropped table must not match
    assert [p.name for p in on.glob("*.regions.tsv")] == ["BGC0003066.regions.tsv"]
