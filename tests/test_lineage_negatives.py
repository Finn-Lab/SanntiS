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

"""Lineage-matched negative pools: plan, portions, portion entities, fungi-only euk track."""

import pandas as pd

from sanntis.negpool import strip
from sanntis.negpool.plan import PlanOptions, plan_pool
from sanntis.train.entities import negative_entities, negative_units, positive_entities


def _cands():
    rows = []
    for fam, order, phylum, n in [("Strep", "Kita", "Actino", 6), ("Micro", "Micro_o", "Actino", 3),
                                  ("Pseudo", "Pseudo_o", "Proteo", 3), ("Clost", "Eubact", "Bacillota", 2),
                                  ("Odd", "Odd_o", "Candidatus X", 2)]:
        for i in range(n):
            rows.append({"genome": f"{fam}{i}", "track": "proka", "phylum": phylum, "order": order, "family": fam,
                         "species": f"{fam} sp{i}", "clean_cds": 4000, "priority": 1})
    rows.append({"genome": "Strep_gold", "track": "proka", "phylum": "Actino", "order": "Kita", "family": "Strep",
                 "species": "Strep gold", "clean_cds": 4000, "priority": 0})
    rows.append({"genome": "Strep_dup", "track": "proka", "phylum": "Actino", "order": "Kita", "family": "Strep",
                 "species": "Strep sp0", "clean_cds": 4000, "priority": 1})  # same species as Strep0
    return pd.DataFrame(rows)


def _positives():
    rows = [{"track": "proka", "phylum": "Actino", "order": "Kita", "family": "Strep", "n_cds": 30}] * 6
    rows += [{"track": "proka", "phylum": "Actino", "order": "Micro_o", "family": "Unseen", "n_cds": 20}]  # → order
    rows += [{"track": "proka", "phylum": "Proteo", "order": "Pseudo_o", "family": "Pseudo", "n_cds": 20}]
    return pd.DataFrame(rows)


def test_plan_matches_lineage_mix():
    o = PlanOptions(budget={"proka": 6000}, cap=1500, floor_cds=500, min_quota=100, seed=1,
                    exclude_species=frozenset({"Pseudo sp0"}))
    plan, st = plan_pool(_positives(), _cands(), o)
    s = st["proka"]
    # floor: one genome for each phylum without positives (Bacillota, Candidatus X)
    floor = plan[plan["plan_role"] == "floor"]
    assert sorted(floor["phylum"]) == ["Bacillota", "Candidatus X"] and (floor["quota_cds"] == 500).all()
    assert s["floor_cds"] == 1000 and s["matched_budget"] == 5000
    m = plan[plan["plan_role"] == "matched"].groupby("plan_group")["quota_cds"].sum()
    # 180 : 20 : 20 positive CDS → 4091 : 455 : 455 (rounded)
    assert abs(m["family:Strep"] - 4091) <= 1 and abs(m["order:Micro_o"] - 455) <= 1
    assert abs(m["family:Pseudo"] - 455) <= 1
    assert plan["quota_cds"].max() <= 1500
    strep = plan[plan["plan_group"] == "family:Strep"]
    assert strep.iloc[0]["genome"] == "Strep_gold"  # priority first
    assert plan["species"].is_unique and "Pseudo sp0" not in set(plan["species"])


def test_portions_are_disjoint_windows_of_the_concatenation():
    runs = pd.DataFrame({"genome": "g", "contig": ["c1", "c1", "c2"], "run_id": ["c1_run10", "c1_run2", "c2_run0"],
                         "cds_idx": [list(range(50, 60)), list(range(0, 40)), list(range(0, 50))],
                         "n_cds": [10, 40, 50]})
    p = strip.portions(runs, 30, k=3, seed=0)
    assert [x["n_cds"] for x in p] == [10, 10, 10]
    keys = [(c, i) for x in p for c, i in zip(x["contigs"], x["cds_idx"], strict=True)]
    assert len(keys) == len(set(keys)) == 30
    # concatenation order is contig, then genomic position (c1 0..39, c1 50..59, c2 0..49)
    order = [("c1", i) for i in range(40)] + [("c1", i) for i in range(50, 60)] + [("c2", i) for i in range(50)]
    pos = [order.index(k) for k in keys]
    assert pos == sorted(pos)
    assert strip.portions(runs, 30, k=3, seed=0) == p  # deterministic
    small = strip.portions(runs, 1000)
    assert len(small) == 1 and small[0]["n_cds"] == 100


def _portion_pool():
    genomes = pd.DataFrame({"genome": ["g1", "f1", "p1"], "kingdom": ["proka", "euk", "euk"],
                            "lineage": ["d__Bacteria", "superkingdom__Eukaryota;kingdom__fungi",
                                        "superkingdom__Eukaryota;kingdom__protozoa"]})
    rows = []
    for g in ("g1", "f1", "p1"):
        for c in ("c1", "c2"):
            for i in range(40):
                rows.append((g, c, i, [f"IPR{i % 11}"] if i < 30 else None))  # CDS 30.. not annotated
    cds = pd.DataFrame(rows, columns=["genome", "contig", "cds_idx", "tokens"])
    portions = pd.DataFrame({"genome": ["g1", "f1", "p1"], "run_id": ["g1:portion0", "f1:portion0", "p1:portion0"],
                             "contigs": [["c1"] * 30 + ["c2"] * 30] * 3,
                             "cds_idx": [list(range(30)) * 2] * 3, "n_cds": [60] * 3})
    empty = pd.DataFrame(columns=["genome", "contig", "run_id", "cds_idx", "n_cds"])
    return {"portions": portions, "genomes": genomes, "cds": cds, "runs": empty,
            "backbones": pd.DataFrame(columns=["genome", "contig", "run_id", "kingdom"])}


def test_entities_from_portions_cross_contigs_and_filter_euk():
    loci = pd.DataFrame({"entity_id": ["BGC1", "BGC2"], "kingdom": ["proka", "euk"], "class_raw": ["NRPS", "PKS"],
                         "length": [1000, 1000], "cds_tokens": [[["IPR1"], ["IPR2"]] * 5, [["IPR3"]] * 6]})
    pos, _ = positive_entities(loci)
    neg, st = negative_entities(_portion_pool(), pos, euk_lineages=("fungi",))
    assert st["negative_units"] == "portions" and st["genomes_euk"] == 1
    assert set(neg.loc[neg["kingdom"] == "euk", "genome"]) == {"f1"}  # protozoa genome left out
    bg = neg[(neg["kind"] == "background") & (neg["genome"] == "g1")]
    # one portion spanning two contigs: chunks are cut out, the rest forms background pieces
    total = neg.loc[neg["genome"] == "g1", "n_cds"].sum()
    assert total <= 60 and bg["n_cds"].min() >= 5
    neg_all, _ = negative_entities(_portion_pool(), pos)
    assert set(neg_all.loc[neg_all["kingdom"] == "euk", "genome"]) == {"f1", "p1"}


def test_v1_pool_units_keep_backbones_for_chunks_only():
    from tests.test_entities import _pool

    u = negative_units(_pool())
    assert list(u["backbone"]) == [False, False, True]
    assert [len(m) for m in u["members"]] == [60, 40, 20]
