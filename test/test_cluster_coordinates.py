import pathlib

import pytest

# CLUSTER rows (all GFF columns but source, which holds the version) expected
# from BGC0001472, whatever the order of the input proteins
EXPECTED_CLUSTERS = [
    ["BGC0001472", "CLUSTER", "312", "19485", ".", ".", ".",
     "ID=BGC0001472_sanntis_1;nearest_MiBIG=BGC0001472;nearest_MiBIG_class=RiPP;"
     "nearest_MiBIG_diceDistance=0.037;score=0.900;partial=11"],
]


def assert_cluster_coordinates(gff):
    clusters = [
        l.split("\t") for l in gff.read_text().splitlines()
        if not l.startswith("#") and l.split("\t")[2:3] == ["CLUSTER"]
    ]
    assert clusters, f"no CLUSTER in {gff}"
    for spl in clusters:
        assert int(spl[3]) <= int(spl[4]), f"CLUSTER start > end: {spl[:5]}"
    assert [spl[:1] + spl[2:] for spl in clusters] == EXPECTED_CLUSTERS


@pytest.mark.workflow("sanntis_with_preprocessed_files")
def test_preprocessed_cluster_coordinates(workflow_dir):
    assert_cluster_coordinates(pathlib.Path(
        workflow_dir,
        "BGC0001472.fna.prodigal.faa.sanntis/BGC0001472.fna.prodigal.faa.sanntis.full.gff",
    ))


@pytest.mark.workflow("sanntis_with_gzipped_preprocessed_files")
def test_gzipped_preprocessed_cluster_coordinates(workflow_dir):
    assert_cluster_coordinates(pathlib.Path(
        workflow_dir,
        "BGC0001472.fna.prodigal.faa.gz.sanntis/BGC0001472.fna.prodigal.faa.gz.sanntis.full.gff",
    ))


@pytest.mark.workflow("sanntis_with_shuffled_preprocessed_files")
def test_shuffled_cluster_coordinates(workflow_dir):
    assert_cluster_coordinates(pathlib.Path(
        workflow_dir,
        "BGC0001472.fna.prodigal.shuffled.faa.sanntis/BGC0001472.fna.prodigal.shuffled.faa.sanntis.full.gff",
    ))


@pytest.mark.workflow("sanntis_with_gzipped_shuffled_preprocessed_files")
def test_gzipped_shuffled_cluster_coordinates(workflow_dir):
    assert_cluster_coordinates(pathlib.Path(
        workflow_dir,
        "BGC0001472.fna.prodigal.shuffled.faa.gz.sanntis/BGC0001472.fna.prodigal.shuffled.faa.gz.sanntis.full.gff",
    ))
