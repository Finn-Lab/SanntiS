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

import numpy as np
import pandas as pd

from sanntis.config import KeyDomainsConfig
from sanntis.io.gff import feature_lines
from sanntis.model.key_domains import KeyDomains
from sanntis.model.npc import NpcModel
from sanntis.predict import REGION_COLUMNS
from sanntis.train import key_domains as kd

# Wilson bounds on _loci(): K in A 3/0 → 0.44, X+Y 2/0 → 0.34; X 3/2, Y 2/2, Z 2/1 → < 0.25
CFG = KeyDomainsConfig(tau=0.3, min_support=2, min_coverage=0.5, entry_types=(), background_max=1.0)


def _loci():
    """Pathway A: token K (A only) in 3 loci, or the pair X+Y (each also in B) in 2; pathway B: token X or Y alone."""
    cds = [[["K"], ["X"]], [["K"]], [["K", "Z"]], [["X"], ["Y"]], [["X", "Y", "Z"]],
           [["X"]], [["Y"]], [["X", "Z"]], [["Y"]]]
    Y = np.zeros((len(cds), 2))
    Y[:5, 0] = 1
    Y[5:, 1] = 1
    return cds, Y


def _rules(df, cat):
    d = df[df["category"] == cat]
    return set(zip(d["t1"], d["t2"], strict=True))


def test_admission_pairs_and_short_list():
    cds, Y = _loci()
    bags = [kd.bag(c) for c in cds]
    adm = kd.admitted(bags, Y, ["A", "B"], CFG)
    assert _rules(adm, "A") == {("K", ""), ("X", "Y")}  # X, Y alone are not specific to A
    assert (adm["a"] >= 2).all() and (adm["score"] >= 0.3).all() and not (adm["category"] == "B").any()
    short = kd.short_list(adm, bags, Y, ["A", "B"])
    assert _rules(short, "A") == {("K", ""), ("X", "Y")}  # K covers 3 loci, X+Y the other 2


def test_entry_types_and_background():
    cds, Y = _loci()
    # K is a superfamily: A keeps only X+Y; with X+Y in 1 of 2 background entities, nothing is left for A
    allowed = kd.allowed_tokens(CFG.model_copy(update={"entry_types": ("Domain",)}),
                                {"K": "Homologous_superfamily", "X": "Domain", "Y": "Domain", "Z": "Domain"})
    assert allowed == {"X", "Y", "Z"}
    r = kd.fit_rules(cds, Y, ["A", "B"], CFG, allowed=allowed, background=[])
    assert _rules(r, "A") == {("X", "Y")}
    bg = [[["X"], ["Y"]], [["Z"]]]
    r = kd.fit_rules(cds, Y, ["A", "B"], CFG.model_copy(update={"background_max": 0.5}), allowed=allowed,
                     background=bg)
    assert _rules(r, "A") == {("X", "Y")} and (r["bg_rate"] <= 0.5).all()
    r = kd.fit_rules(cds, Y, ["A", "B"], CFG.model_copy(update={"background_max": 0.4}), allowed=allowed,
                     background=bg)
    assert not len(r[r["category"] == "A"])
    assert kd.allowed_tokens(CFG, None) is None  # no entry_types: every token


def test_scores():
    a, b = np.array([10, 3]), np.array([0, 0])
    w = kd.wilson_lower_bound(a, b)
    assert w[0] > w[1] and (w < 1).all()  # Wilson bound shrinks small support
    assert np.allclose(kd.precision(a, np.array([10, 3])), 0.5)


def _kd():
    cats = {"A": {"status": "active", "oof_coverage": 0.9,
                  "rules": [{"tokens": ["K"]}, {"tokens": ["X", "Y"]}]},
            "B": {"status": "gated", "oof_coverage": 0.2, "rules": [{"tokens": ["X"]}]},
            "C": {"status": "no_rules", "oof_coverage": None, "rules": []}}
    return KeyDomains(kd.setting(CFG), cats)


def test_support_semantics():
    k = _kd()
    assert k.support([["K"], ["X", "Y"]], ["A"]) == ("A:K|A:X+Y", False)
    assert k.support([["X"]], ["A"]) == ("A:none", True)
    assert k.support([["X"]], ["B"]) == ("B:na", None)  # gated
    assert k.support([["X"]], ["C", "B"]) == ("C:na|B:na", None)
    assert k.support([["X"]], ["A", "B"]) == ("A:none|B:na", True)  # any judged pathway unsupported
    assert k.support([["K"]], ["A", "C"]) == ("A:K|C:na", False)
    assert KeyDomains.from_json(k.to_json()).support([["X", "Y"]], ["A"]) == ("A:X+Y", False)


def test_bundle_roundtrip_and_outputs(tmp_path, dummy_model):
    npc = dummy_model.npc
    p = npc.pathways
    npc.key_domains = KeyDomains(kd.setting(CFG), {p[0]: {"status": "active", "oof_coverage": 1.0,
                                                    "rules": [{"tokens": ["IPR000001"]}]}})
    out = dummy_model.save_pretrained(tmp_path / "b")
    loaded = NpcModel.load(out)
    assert loaded.key_domains.to_json() == npc.key_domains.to_json()
    regions = [[["IPR000001"]], [["PF00002"]]]
    df = loaded.predict(regions)
    assert df["npc_pathway_key_domains_support"].notna().all()
    for (_, r), cds in zip(df.iterrows(), regions, strict=True):
        assert (r["npc_pathway_key_domains_support"], r["npc_pathway_key_domains_unsupported"]) == \
            loaded.key_domains.support(cds, r["npc_pathway"].split("|"))
    # without rules: columns present, None
    loaded.key_domains = None
    assert loaded.predict(regions)["npc_pathway_key_domains_unsupported"].isna().all()
    assert {"npc_pathway_key_domains_support", "npc_pathway_key_domains_unsupported"} <= set(REGION_COLUMNS)


def test_gff_attributes():
    base = {c: None for c in REGION_COLUMNS} | {
        "record_id": "r", "region": 1, "start": 1, "end": 10, "score": 0.9, "p_non_bgc": 0.1, "n_orfs": 3,
        "npc_pathway": "Polyketides|Terpenoids", "npc_pathway_score": 0.8, "npc_pathway_confident": True,
        "npc_superclass": "unassigned", "npc_superclass_top": "x", "npc_superclass_score": 0.2}
    rows = [base | {"npc_pathway_key_domains_support": "Polyketides:IPR1+IPR2|Terpenoids:none",
                    "npc_pathway_key_domains_unsupported": True},
            base | {"npc_pathway_key_domains_support": "Polyketides:na", "npc_pathway_key_domains_unsupported": None},
            base]
    lines = list(feature_lines(pd.DataFrame(rows, columns=REGION_COLUMNS)))
    assert "npc_pathway_key_domains_support=Polyketides:IPR1+IPR2|Terpenoids:none;" \
           "npc_pathway_key_domains_unsupported=true" in lines[0]
    assert lines[1].endswith("npc_pathway_key_domains_support=Polyketides:na;npc_pathway_key_domains_unsupported=na")
    assert "key_domains" not in lines[2]


def test_oof_evaluation_gate():
    from sanntis.model.npc import NpcThresholds
    from sanntis.train.npc_train import CvPreds

    # 4 regions, 2 categories; calls = argmax; A: covered except region 1 (a wrong call); B: never covered
    pp = np.array([[0.9, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.1]])
    yp = np.array([[1, 0], [0, 1], [0, 1], [1, 0]], float)
    has = np.array([[True, False], [False, False], [False, False], [True, False]])
    p = CvPreds(pp, np.zeros((4, 0)), yp, np.zeros((4, 0)), np.zeros(4), np.array(["proka"] * 4), [])
    th = NpcThresholds(np.full(2, 0.5), {"pw": {0.9: 0.0}, "sc": {}})
    parts = [{"preds": p, "has": has, "with_rules": np.array([True, True])}]
    calls = kd.oof_calls(parts, lambda k: th, 0.9, ["A", "B"])
    cov = kd.coverage(parts, ["A", "B"])
    assert list(cov["coverage"]) == [1.0, 0.0]
    ev = kd.evaluate(calls, cov, 0.5)
    assert ev["gated"] == ["B"]
    assert ev["all"]["flag_wrong"] == 1 and ev["all"]["flag_correct"] == 0  # region 2's B call gated, not flagged
    assert (ev["all"]["wrong"], ev["all"]["catch"], ev["all"]["false_alarm"]) == (1, 1.0, 0.0)
    assert ev["all"]["na_rate"] == 0.25  # the B call
    assert not calls.loc[calls["category"] == "B", "na"].any()  # na only after the gate
