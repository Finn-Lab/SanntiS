import os
import sys
import filecmp
import pytest
import sanntis
import numpy as np

sanntis_dir = os.path.dirname(os.path.abspath(sanntis.__file__))
test_files_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "files")
modules_dir = os.path.join(sanntis_dir, "modules")
sys.path.append(modules_dir)

from BGCdetection import AnnotationFilesToEmerald

def test_bgc_prediction():
    
    prodigal_file = os.path.join(test_files_dir, "BGC0001472.fna.prodigal.faa")
    ips_file = os.path.join(test_files_dir, "BGC0001472.fna.prodigal.faa.ip.tsv")
    hmm_file = os.path.join(test_files_dir, "BGC0001472.fna.prodigal.faa.sanntis.tsv")

    ann = AnnotationFilesToEmerald()
    ann.transformIPS(ips_file)
    ann.transformEmeraldHmm(hmm_file)
    fmt = "fna"
    ann.transformCDSpredToCDScontigs(
            prodigal_file,
            fmt)

    ann.buildMatrices()
    assert ann.annDct['BGC0001472'].shape == (200, 15264)
    assert np.sum(ann.annDct['BGC0001472']) == 46.
    ann.predictAnn()
    np.sum(ann.annResults['BGC0001472']) == 15.492591619491577
    score,g = None, 1
    ann.defineLooseClusters(score=score, g=g)

    ann.predictType()
    assert np.array_equal(np.where(ann.typesClst['BGC0001472']!=None)[0], np.array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15]))

