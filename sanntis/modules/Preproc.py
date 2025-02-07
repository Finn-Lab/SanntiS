# Copyright 2021 EMBL - European Bioinformatics Institute
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

import logging
import os
import sys
import subprocess
from Bio import SeqIO

from sanntis import _params
log = logging.getLogger(f"SanntiS.{__name__}")

from distutils.spawn import find_executable

class Preprocess:
    """External tools needed for sanntis bgc detection"""

    def __init__(self, seq_file, seqfile_is_proteins, ip_file, meta, cpus, outdir):

        self.seq_file = seq_file
        self.seqfile_is_proteins = seqfile_is_proteins
        self.ip_file = ip_file
        self.meta = meta
        self.cpus = int(cpus)
        self.outdir = outdir if outdir else "temp"

    def runProdigal(self):
        """Predict genes using prodigal"""
        log.info("Progial gene prediction...")

        if not find_executable("prodigal"):
            log.exception("Parodigal is not installed or in PATH")

        if not os.path.isfile(self.seq_file):
            log.exception(f"{self.seq_file} file not found")

        outFaa = os.path.join(
            self.outdir, "{}.prodigal.faa".format(os.path.basename(self.seq_file))
        )
        cmd = ["prodigal", "-i", self.seq_file, "-a", outFaa] + (
            ["-p", "meta"] if self.meta == "True" else []
        )

        outs, errs = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        ).communicate()

        if "Error:  Sequence must be 20000 characters" in errs.decode("utf8"):

            log.info(
                "Sequence must be 20000 characters when running Prodigal in normal mode. Trying -p meta"
            )

            cmd = ["prodigal", "-i", self.seq_file, "-a", outFaa, "-m"] + (["-p", "meta"])

            outs, errs = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            ).communicate()

        log.info(errs.decode("utf8"))

        """ Remove asterixs from faa
            """
        log.info("Removing asterix from prodigal faa")
        with open(outFaa, "r") as h:
            noAstFaa = [x.replace("*", "") for x in h]
        with open(outFaa, "w") as h:
            for l in noAstFaa:
                h.write(f"{l}")

        return os.path.abspath(outFaa)

    def gbkToProdigal(self):
        """Transform gbk to faa. This enables preprocessing with sequences"""
        log.info("write gbk as faa")
        from Bio import SeqIO

        recs = list(SeqIO.parse(open(self.seq_file, "r"), "gb"))

        base = os.path.basename(self.seq_file)

        outFaa = os.path.join(
            self.outdir, "{}.prodigal.faa".format(os.path.basename(self.seq_file))
        )

        with open(outFaa, "w") as h:

            for rec in recs:
                ct = 0
                for f in rec.features:

                    
                    if f.type == "CDS" and "translation" in f.qualifiers:
                        ct += 1
                        pid = (
                            f.qualifiers["protein_id"][0].replace(" ","")
                            if "protein_id" in f.qualifiers
                            else f.qualifiers["locus_tag"][0]
                        )
                        seq = f.qualifiers["translation"][0]
                        h.write(f">{pid}\n{seq}\n")
                        
                if ct == 0:
                    log.info("{} CDS found with translation in {}".format(ct, rec.name) )

        return os.path.abspath(outFaa)

    def check_fmt(self):
        """ Evaluate if input format is FNA or GBK"""
        for fmt in ["fasta","genbank"]:
            seqFile = SeqIO.parse(open(self.seq_file),fmt)
            if any(seqFile):
                self.fmt = fmt
                log.info(f"{fmt} sequence file detected")
                return
        log.exception(f"sequence file {self.seq_file} not in fasta or genbank format")
        raise SystemExit(f"sequence file {self.seq_file} not in fasta or genbank format")


    def process_sequence(self):
        """ CDS prediction on sequence file"""

        self.check_fmt()
        
        if self.fmt == "fasta":
            self.outFaa = self.seq_file if self.seqfile_is_proteins else self.runProdigal()
        elif self.fmt == "genbank":
            self.outFaa = self.gbkToProdigal()
        else:
            log.info("missing sequence file format")

        ip_f = self.runInterproscan() if self.ip_file == None else self.ip_file
        ih_f = self.runHmmScan()
        return self.outFaa, ip_f, ih_f

    def runHmmScan(self):
        """annotate functionally with sanntis hmm library and hmmScan"""
        log.info("sanntis functional annotation...")
        hmmLib = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "models",
            "hmm_lib",
            "sanntis.hmm",
        )

        if not find_executable("hmmscan"):
            log.exception("hmmscan is not installed or in PATH")

        if not os.path.isfile(self.outFaa):
            log.exception(f"{self.outFaa} file not found")

        if not os.path.isfile(hmmLib):
            log.exception(f"{hmmLib} file not found")

        outTsv = os.path.join(
            self.outdir, "{}.sanntis.tsv".format(os.path.basename(self.outFaa))
        )

        cmd = (
            ["hmmscan", "--domtblout", outTsv, "--cut_ga"]
            + (["--cpu", str(self.cpus)] if self.cpus else [])
            + ([hmmLib, self.outFaa])
        )
        log.info(" ".join(cmd))
        try:
            outs, errs = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            ).communicate()
            log.info(errs.decode("utf8"))
        except subprocess.CalledProcessError as err:
            log.exception(err.output)

        return os.path.abspath(outTsv)

    def runInterproscan(self):
        """annotate functionally with InterproScan"""
        log.info("InterProScan")

        if not find_executable('interproscan.sh'):
            print("\nInterProScan (IPS) executable interproscan.sh could not be found\n")
            print("If IPS is not installed. Make sure to run sanntis with --ip-file option\n")
            print("Alternatively:")
            print("    Activate sanntis environment")
            print("    get a full copy of IPS https://interproscan-docs.readthedocs.io/en/latest/InstallationRequirements.html and be sure to have interproscan.sh in PATH\n")
            print("    Or use the docker script found in the sanntis repository\t")
            log.exception(f"interproscan.sh not found, only available for Linux OS")
            raise SystemExit('interproscan.sh not found')

        if not os.path.isfile(self.outFaa):
            log.exception(f"self.outFaa file not found")

        outGff = os.path.join(
            self.outdir, "{}.ip.tsv".format(os.path.basename(self.outFaa))
        )
        os.environ['LD_LIBRARY_PATH'] = os.path.join(os.environ["CONDA_PREFIX"], "lib")
        os.environ["PERL5LIB"] = ""
        cmd = (
            ["interproscan.sh", "-i", self.outFaa, "-o", outGff, "-f", "TSV"]
            + (["-appl", ",".join(_params["ip_an"])] if _params["ip_an"] else [])
            + (["-cpu", str(self.cpus)] if self.cpus else [])
        )
        log.info(" ".join(cmd))

        outs, errs = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        ).communicate()
        log.info(outs.decode("utf8"))
        log.info(errs.decode("utf8"))

        return os.path.abspath(outGff)
