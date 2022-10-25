import os
import sys
import filecmp
import pytest
import sanntis
import tempfile
import subprocess

emrld_dir = os.path.dirname(os.path.abspath(sanntis.__file__))
test_files_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "files")
modules_dir = os.path.join(emrld_dir, "modules")
sys.path.append(modules_dir)

from Preproc import Preprocess

def test_prodigal():
    with tempfile.TemporaryDirectory() as tmpdir:
        pp = Preprocess(
            os.path.join(test_files_dir, "BGC0001472.fna"),
            None,
            False,
            1,
            tmpdir
        )
        test_prodigal_file =  pp.runProdigal()
        assert filecmp.cmp(test_prodigal_file, os.path.join(test_files_dir, "BGC0001472.fna.prodigal.faa")) 
    
def test_gbk_transform():
    with tempfile.TemporaryDirectory() as tmpdir:
        pp = Preprocess(
            os.path.join(test_files_dir, "BGC0001472.fna.prodigal.faa.gb"),
            None,
            False,
            1,
            tmpdir
        )
        test_prodigal_file =  pp.gbkToProdigal()
        assert filecmp.cmp(test_prodigal_file, os.path.join(test_files_dir, "BGC0001472.fna.prodigal.faa.gb.prodigal.faa"))
     
def test_hmmscan_ih():
    assert subprocess.check_output('hmmscan -h',shell=True)
     
def test_ipscan_ih():
    if 'linux' not in sys.platform:
        assert True
    else: 
        assert subprocess.check_output("interproscan.sh --version",shell=True)