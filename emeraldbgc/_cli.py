#!/usr/bin/env python3

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

import argparse
import glob
import logging
import os
import sys

from emeraldbgc import __version__

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.BGCdetection import AnnotationFilesToEmerald
from modules.Preproc import Preprocess
from modules.WriteOutput import Outputs


def main(args=None):

    parser = argparse.ArgumentParser(description="EMERALD. SMBGC detection tool")
    parser.add_argument(
        "seq_file",
        type=str,
        help="nucleotide sequence file. FASTA or GBK. mandatory",
        metavar="SEQUENCE_FILE",
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        help="Show the version number and exit.",
        version=f"EMERALD {__version__}",
    )
    parser.add_argument(
        "--greed",
        dest="greed",
        default=2,
        type=int,
        help="Level of greediness. 0,1,2,3 [default 2]",
        metavar="INT",
    )
    parser.add_argument(
        "--score",
        dest="score",
        default=None,
        type=float,
        help="validation filter threshold. overrides --greed",
        metavar="FLOAT",
    )
    parser.add_argument(
        "--meta",
        dest="meta",
        default="True",
        type=str,
        help="prodigal option meta [default True]",
        metavar="True|False",
    )
    parser.add_argument(
        "--outdir",
        default=os.getcwd(),
        dest="outdir",
        type=str,
        help="output directory [default $PWD/SEQUENCE_FILE.emerald]",
        metavar="DIRECTORY",
    )
    parser.add_argument(
        "--outfile",
        dest="outfile",
        type=str,
        help="output file [default outdir/SEQUENCE_FILE.emerald.gff]",
        metavar="FILE",
    )
    parser.add_argument(
        "--minimal",
        dest="minimal_out",
        default="True",
        type=str,
        help="minimal output in a gff3 file [default True]",
        metavar="True|False",
    )
    parser.add_argument(
        "--refined",
        dest="ref_b",
        default="True",
        type=str,
        help="annotate high probability borders [default True]",
        metavar="True|False",
    )
    parser.add_argument(
        "--cpu",
        dest="cpu",
        default=1,
        type=int,
        help="cpus for INTERPROSCAN and HMMSCAN",
        metavar="INT",
    )

    args = parser.parse_args(args)

    basef = args.seq_file
    base = os.path.basename(basef)

    outdir = os.path.join(args.outdir, f"{base}.emerald")

    os.makedirs(outdir, exist_ok=True)

    logging.basicConfig(
        filename=f"{outdir}/emerald.log",
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    print(f"LOG_FILE: {outdir}/emerald.log")
    log = logging.getLogger("EMERALD")
    log.info(
        f"""
    ******
    EMERALD v{__version__}
    ******"""
    )
    log.info(f"outdir: {outdir}")

    log.info("preprocessing files")
    preprocess = Preprocess(os.path.abspath(args.seq_file), args.meta, args.cpu, outdir)
    prodigal_file, ips_file, hmm_file = preprocess.process_sequence()

    log.info("EMERALD process")
    annotate = AnnotationFilesToEmerald()

    log.info("transform interpro file")
    annotate.transformIPS(ips_file)

    log.info("transform inhouse hmm file")
    annotate.transformEmeraldHmm(hmm_file)

    log.info("transform cds file")
    annotate.transformCDSpredToCDScontigs(prodigal_file, preprocess.fmt)

    log.info("transform dicts to np matrices")
    annotate.buildMatrices()

    log.info("predict bgc regions")
    annotate.predictAnn()

    log.info("define clusters")
    annotate.defineLooseClusters()

    log.info("post-processing filters and type classification")
    annotate.predictType(score=args.score, g=args.greed)

    log.info("write output file file")
    outp = Outputs(
        annotate,
        args.minimal_out,
        args.ref_b,
        args.outfile if args.outfile else f"{outdir}/{base}.emerald.full.gff",
    )

    log.info("EMERALD succesful")
    print("EMERALD succesful")


if __name__ == "__main__":
    main(sys.argv[1:])
