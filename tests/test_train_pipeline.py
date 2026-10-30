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

"""Training core end to end on a tiny synthetic pool: cv → tune → finalize → bundle → Predictor."""

import json

import numpy as np
import pandas as pd

from sanntis.config import load_recipe
from sanntis.model.bundle import SanntisModel
from sanntis.predict import Predictor
from sanntis.train.cv import run_cv_unit
from sanntis.train.finalize import build_bundle, run_seed_unit
from sanntis.train.tune import run_tune

CLS = ["NRPS", "PKS", "terpene"]
PW = ["Amino acids and Peptides", "Polyketides", "Terpenoids"]


def synthetic_table(rng):
    rows = []
    for i in range(36):
        c = CLS[i % 3]
        cds = [[f"IPR{c}{j % 4}", f"PF{c}{(j + i) % 3}"] for j in range(int(rng.integers(4, 9)))]
        rows.append({"entity_id": f"BGC{i:04d}", "kind": "positive", "label": 1, "class": c, "classes": (c,),
                     "kingdom": "proka" if i % 6 else "euk", "genome": None, "n_cds": len(cds), "cds_tokens": cds})
    for i in range(90):
        cds = [[f"IPRbg{int(x)}"] if x % 5 else [] for x in rng.integers(0, 40, int(rng.integers(3, 10)))]
        kind = "chunk" if i < 60 else "background"
        rows.append({"entity_id": f"n{i}", "kind": kind, "label": 0, "class": "non-BGC", "classes": (),
                     "kingdom": "proka" if i % 6 else "euk", "genome": f"g{i % 9}", "n_cds": len(cds),
                     "cds_tokens": cds})
    return pd.DataFrame(rows)


def test_train_core(tmp_path):
    rng = np.random.default_rng(0)
    r = load_recipe()
    npc = r.npc.model_copy(update={"min_labelled": 5, "min_support": 2,
                                   "rf": r.npc.rf.model_copy(update={"n_estimators": 20})})
    recipe = r.model_copy(update={"npc": npc})
    table = synthetic_table(rng)
    run = tmp_path / "run"
    for k in range(5):
        m = run_cv_unit(table, k, 0, run / "cv" / f"split{k}_seed0", recipe=recipe, device="cpu", max_epochs=2)
        assert m["epochs_run"] == 2 and m["vocab_size"] > 3
    labels = pd.DataFrame({"accession": table.entity_id[:36], "npc_status": "classified",
                           "pathway": [[PW[i % 3]] for i in range(36)],
                           "superclass": [[f"sc{i % 3}"] for i in range(36)]}).set_index("accession")
    parents = {f"sc{i}": [PW[i]] for i in range(3)}
    # IPRNRPS0 typed as a superfamily: never a rule
    types = {t: "Homologous_superfamily" if t == "IPRNRPS0" else "Domain"
             for c in table["cds_tokens"] for g in c for t in g if t.startswith("IPR")}
    tune = run_tune(run, table, recipe=recipe, labels=labels, parents=parents, n_boot=20, entry_types=types)
    assert 0.1 <= tune["t_star"] <= 0.95 and tune["epochs"] >= 1
    assert len(tune["npc"]["pathway_thresholds"]) == 7
    kd = tune["npc"]["key_domains"]
    assert len(kd["per_category"]) == 7 and kd["all"]["calls"] > 0
    assert set(kd["gated"]) <= {c["category"] for c in kd["per_category"]}
    assert kd["setting"]["background_max"] == recipe.npc.key_domains.background_max
    sweep = pd.read_csv(run / "tune" / "sweep.tsv", sep="\t")
    assert set(sweep["stratum"]) == {"all", "proka", "euk"}
    for s in recipe.train.seeds:
        run_seed_unit(table, s, 1, run / "finalize", recipe=recipe, device="cpu")
    sm = pd.DataFrame({"signature": ["PF1"], "app": ["pfam"], "entry": ["IPR1"]})
    oof = pd.read_parquet(run / "tune" / "oof_regions.parquet")
    oof["cds_tokens"] = oof["cds_tokens"].map(lambda c: [list(g) for g in c])
    build_bundle(table, tune, run / "finalize", run / "bundle", signature_map=sm, recipe=recipe, labels=labels,
                 parents=parents, oof_regions=oof, entry_types=types)
    model = SanntisModel.from_pretrained(run / "bundle")
    assert model.config.decode.threshold == tune["t_star"] and len(model.detector_states) == 3
    mf = json.loads((run / "bundle" / "manifest.json").read_text())
    assert mf["tuned"]["epochs"] == tune["epochs"]
    assert "npc_key_domains.json" in mf["files"] and model.npc.key_domains is not None
    rules = [r for c in model.npc.key_domains.categories.values() for r in c["rules"]]
    assert rules and all(t.startswith("IPR") and t != "IPRNRPS0" for r in rules for t in r["tokens"])
    assert all(r["background"] <= 0.02 for r in rules)
    pos = table.iloc[0]
    cds = pd.DataFrame({"record_id": "r", "cds_idx": range(pos.n_cds), "locus_tag": [f"l{i}" for i in range(pos.n_cds)],
                        "start": np.arange(pos.n_cds) * 1000, "end": np.arange(pos.n_cds) * 1000 + 900,
                        "tokens": pos.cds_tokens})
    pred = Predictor(model).predict(cds)
    assert len(pred.orf_scores) == pos.n_cds
    assert {"npc_pathway_key_domains_support", "npc_pathway_key_domains_unsupported"} <= set(pred.regions.columns)

    from sanntis.train.report import write_report

    table["lineage"] = ["prokaryotes" if k == "proka" else "fungi" for k in table["kingdom"]]
    rep = write_report(run, table, metadata={"Run ID": "test"}, recipe=recipe)
    md = (rep / "report.md").read_text()
    assert "Precision_T" in md and (rep / "figures" / "operating_point.png").stat().st_size > 1000
    assert (rep / "npc_validation.tsv").exists()
    assert "key-domain support" in md and (rep / "npc_key_domains_oof.tsv").exists()
