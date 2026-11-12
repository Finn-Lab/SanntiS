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
"""sanntis plan-negatives: positives of a train run, candidates by NCBI taxid (one taxdump), exclusion, breadth."""

import json
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

from sanntis.cli import main
from sanntis.negpool import sample as S
from sanntis.negpool.plan import gtdb_candidates
from sanntis.negpool.taxdump import Taxdump

NODES = [  # taxid, parent, rank, name
    (1, 1, "no rank", "root"), (2, 1, "domain", "Bacteria"),
    (10, 2, "phylum", "Actinomycetota"), (11, 10, "order", "Kitasatosporales"), (12, 11, "family", "Streptomycetaceae"),
    (13, 12, "genus", "Streptomyces"), (14, 13, "species", "Streptomyces alpha"),
    (15, 13, "species", "Streptomyces beta"), (16, 13, "species", "Streptomyces gamma"),
    (20, 2, "phylum", "Pseudomonadota"), (21, 20, "order", "Pseudomonadales"), (22, 21, "family", "Pseudomonadaceae"),
    (23, 22, "genus", "Pseudomonas"), (24, 23, "species", "Pseudomonas a"), (25, 23, "species", "Pseudomonas b"),
    (30, 2, "phylum", "Bacillota"), (31, 30, "order", "Eubacteriales"), (32, 31, "family", "Clostridiaceae"),
    (33, 32, "genus", "Clostridium"), (34, 33, "species", "Clostridium a"),
    (2759, 1, "domain", "Eukaryota"), (4751, 2759, "kingdom", "Fungi"), (40, 4751, "phylum", "Ascomycota"),
    (41, 40, "order", "Eurotiales"), (42, 41, "family", "Aspergillaceae"), (43, 42, "genus", "Aspergillus"),
    (44, 43, "species", "Aspergillus niger"), (45, 43, "species", "Aspergillus nidulans"),
]


@pytest.fixture
def taxdump(tmp_path):
    d = tmp_path / "taxdump"
    d.mkdir()
    (d / "nodes.dmp").write_text("".join(f"{t}\t|\t{p}\t|\t{r}\t|\tXX\t|\n" for t, p, r, _ in NODES))
    (d / "names.dmp").write_text("".join(f"{t}\t|\t{n}\t|\t\t|\tscientific name\t|\n" for t, _, _, n in NODES))
    (d / "merged.dmp").write_text("99\t|\t14\t|\n")
    return d


def test_taxdump_lineage(taxdump):
    tx = Taxdump(taxdump)
    lin = tx.lineage(99)  # merged → 14
    assert lin["family"] == "Streptomycetaceae" and lin["species"] == "Streptomyces alpha"
    assert lin["superkingdom"] == lin["domain"] == "Bacteria"
    assert tx.lineage(44)["kingdom"] == "Fungi" and tx.lineage("nan") == {} and tx.lineage(123456) == {}


def _run(tmp_path):
    """A train run stopped after annotate: 5 prokaryotic + 2 fungal positives, distinct tokens (no dedup)."""
    run = tmp_path / "run"
    (run / "annotate").mkdir(parents=True)
    (run / "collect").mkdir()
    pos = [("BGC1", "proka", 14, 30), ("BGC2", "proka", 14, 30), ("BGC3", "proka", 15, 30), ("BGC4", "proka", 14, 30),
           ("BGC5", "proka", 24, 20), ("BGC6", "euk", 44, 10), ("BGC7", "euk", 45, 10)]
    pd.DataFrame({
        "entity_id": [p[0] for p in pos], "source": "mibig", "kingdom": [p[1] for p in pos], "class_raw": "NRPS",
        "status": "active", "length": 30_000,
        "lineage": ["prokaryotes" if p[1] == "proka" else "fungi" for p in pos],
        "cds_tokens": [[[f"{p[0]}_{j}"] for j in range(p[3])] for p in pos], "pfam": [[] for _ in pos],
    }).to_parquet(run / "annotate" / "positives.parquet", index=False)
    pd.DataFrame({"accession": [p[0] for p in pos], "ncbi_tax_id": [p[2] for p in pos],
                  "superkingdom": "", "kingdom": [p[1] for p in pos]}).to_csv(run / "collect" / "kingdom_map.tsv",
                                                                                sep="\t", index=False)
    return run


def _tsv(path: Path, rows: list[dict]) -> Path:
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    return path


def _cand(genome, kingdom, acc, taxid, prio, cds, source="test"):
    return {"genome": genome, "kingdom": kingdom, "accession": acc, "taxid": taxid, "source": source,
            "priority": prio, "clean_cds": cds}


def test_plan_negatives_cli(tmp_path, taxdump):
    run = _run(tmp_path)
    cands = _tsv(tmp_path / "cands.tsv", [
        _cand("s1", "proka", "GCF_000000001.1", 14, 0, 3000),
        _cand("s2", "proka", "GCA_000000002.1", 15, 1, 3000),
        _cand("s2_refseq", "proka", "GCF_000000002.1", 15, 0, 3000),  # same assembly, lower priority: kept
        _cand("s3", "proka", "GCF_000000003.1", 16, 0, 3000),  # excluded species
        _cand("p1", "proka", "GCF_000000004.1", 24, 1, 3000),  # excluded assembly
        _cand("p2", "proka", "GCF_000000005.1", 25, 1, 3000),
        _cand("c1", "proka", "GCF_000000006.1", 34, 1, 3000),  # phylum without positives: floor
        _cand("tiny", "proka", "GCF_000000010.1", 25, 0, 50),  # below min quota
        _cand("f1", "euk", "GCF_000000007.1", 44, 1, 8000),
        _cand("f2", "euk", "GCF_000000008.1", 45, 1, 8000),
        {**_cand("u1", "proka", "GCA_000000011.1", 999999, 1, 3000), "phylum": "Candidatus Z"},  # no NCBI lineage
    ])
    exclude = _tsv(tmp_path / "exclude.tsv", [{"accession": "GCA_000000004.9", "species": ""},
                                              {"accession": "", "species": "Streptomyces gamma"}])
    breadth = _tsv(tmp_path / "breadth.tsv", [_cand("b1", "proka", "GCF_000000009.1", 25, 0, 2000, "pool_v1"),
                                              _cand("b2", "proka", "GCA_000000001.1", 14, 0, 2000, "pool_v1"),
                                              _cand("b3", "proka", "GCF_000000004.1", 24, 0, 2000, "pool_v1")])
    out = tmp_path / "plan" / "genomes.tsv"
    r = CliRunner().invoke(main, ["plan-negatives", "--out", str(out), "--positives", str(run), "--taxdump",
                                  str(taxdump), "--candidates", str(cands), "--exclude", str(exclude), "--budget",
                                  "proka=4000", "--budget", "euk=3000", "--breadth", str(breadth)])
    assert r.exit_code == 0, r.output
    t = pd.read_csv(out, sep="\t", dtype=str, keep_default_na=False).set_index("genome")
    assert {"s2_refseq", "c1", "f1", "f2", "b1"} <= set(t.index)
    assert not {"s2", "s3", "p1", "tiny", "b2", "b3"} & set(t.index)
    assert t.at["c1", "plan_role"] == "floor" and t.at["c1", "quota_cds"] == "250"
    assert t.at["u1", "plan_role"] == "floor" and t.at["u1", "phylum"] == "Candidatus Z"
    assert t.at["b1", "plan_role"] == "breadth" and t.at["b1", "quota_cds"] == "500"
    assert "kingdom__Fungi" in t.at["f1", "lineage"] and t.at["f1", "taxon"] == "fungi"
    assert t.at["s1", "species"] == "Streptomyces alpha" and t.at["s1", "group"] == "family:Streptomycetaceae"
    st = json.loads(out.with_suffix(".stats.json").read_text())
    assert st["candidates_excluded"] == 2 and st["breadth_genomes"] == 1
    assert st["proka"]["positives"] == 5 and st["euk"]["positives"] == 2
    g = S.read_genomes(out)  # the table build-negatives reads
    assert set(g["quota_cds"]) - {""} and g["species"].ne("").all()


def test_gtdb_candidates_filter_and_phylum_fallback(tmp_path, taxdump):
    md = _tsv(tmp_path / "bac120_metadata.tsv", [
        {"accession": "RS_GCF_000000011.1", "gtdb_representative": "t", "gtdb_taxonomy": "d__B;p__Actinomycetota",
         "ncbi_taxid": "14", "ncbi_organism_name": "Streptomyces alpha DSM 1", "protein_count": "8000",
         "checkm2_completeness": "99", "checkm2_contamination": "1"},
        {"accession": "GB_GCA_000000012.1", "gtdb_representative": "t", "gtdb_taxonomy": "d__B;p__Candidatus X",
         "ncbi_taxid": "999999", "ncbi_organism_name": "uncultured bacterium", "protein_count": "1000",
         "checkm2_completeness": "95", "checkm2_contamination": "2"},
        {"accession": "RS_GCF_000000013.1", "gtdb_representative": "t", "gtdb_taxonomy": "d__B;p__X",
         "ncbi_taxid": "24", "ncbi_organism_name": "Pseudomonas a", "protein_count": "5000",
         "checkm2_completeness": "80", "checkm2_contamination": "1"},  # incomplete
        {"accession": "RS_GCF_000000014.1", "gtdb_representative": "f", "gtdb_taxonomy": "d__B;p__X",
         "ncbi_taxid": "24", "ncbi_organism_name": "Pseudomonas a", "protein_count": "5000",
         "checkm2_completeness": "99", "checkm2_contamination": "1"},  # not a representative
    ])
    c = gtdb_candidates([md], Taxdump(taxdump), release="226").set_index("genome")
    assert list(c.index) == ["GCF_000000011.1", "GCA_000000012.1"]
    assert c.at["GCF_000000011.1", "clean_cds"] == 7200 and c.at["GCF_000000011.1", "family"] == "Streptomycetaceae"
    assert c.at["GCA_000000012.1", "phylum"] == "Candidatus X" and c.at["GCF_000000011.1", "source"] == "gtdb_r226"


def test_plan_ignores_unselected_candidates():
    """Removing candidates the plan did not select (later exclusions) leaves the plan unchanged."""
    from sanntis.negpool.plan import PlanOptions, plan_pool

    rows = [{"genome": f"g{i}", "track": "proka", "phylum": "A", "order": "O", "family": "F", "species": f"s{i}",
             "clean_cds": 1000, "priority": 1} for i in range(30)]
    pos = pd.DataFrame([{"track": "proka", "phylum": "A", "order": "O", "family": "F", "n_cds": 10}])
    o = PlanOptions(budget={"proka": 3000}, floor_cds=250)
    a, _ = plan_pool(pos, pd.DataFrame(rows), o)
    left = [r for r in rows if r["genome"] in set(a["genome"]) or int(r["genome"][1:]) % 2]
    b, _ = plan_pool(pos, pd.DataFrame(left), o)
    assert list(a["genome"]) == list(b["genome"]) and len(a) == 3
