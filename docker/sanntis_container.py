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
#

###
### run sanntis in a docker container

import os
import sys
import argparse
import subprocess
from distutils.spawn import find_executable

def main(args=None):

    parser = argparse.ArgumentParser(description="SanntiS. SMBGC detection tool")
    parser.add_argument(
        "seq_file",
        type=str,
        help="input nucleotide sequence file. FASTA or GBK. mandatory",
        metavar="SEQUENCE_FILE",
    )
    parser.add_argument(
        "--ip-file",
        dest="ip_file",
        default=None,
        type=str,
        help="Optional, preprocessed InterProScan GFF3 output file. Requires a GBK file as SEQUENCE_FILE. The GBK must have CDS as features, and \"protein_id\" matching the ids in the InterProScan file",
        metavar="FILE",
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
        help="output directory [default $PWD/SEQUENCE_FILE.sanntis]",
        metavar="DIRECTORY",
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
        default="False",
        type=str,
        help="annotate high probability borders [default False]",
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

    args_ = list(args)
    args = parser.parse_args(args)

    sf_ix = args_.index(args.seq_file)
    del args_[sf_ix]
    

    if '--outdir' in args:
        od_ix = args_.index(args.outdir)
        del args_[od_ix:od_ix+2]

    if args.ip_file != None:
        ip_file_cmd = [
                "-v",
                 "{}:/home/inp/ips_file.f:ro".format( os.path.abspath( args.ip_file ) )
                ]
        od_ix = args_.index(args.ip_file)
        args_[od_ix] = "/home/inp/ips_file.f"
    else:
        ip_file_cmd = []


    os.makedirs( os.path.abspath( args.outdir ), exist_ok=True )

    cmd = ["docker",
            "run",
            "--rm",
            "-v",
            "{}/data:/opt/interproscan/:ro".format( os.path.abspath( os.path.dirname(sys.argv[0]) ) ),
            "-v",
            "{}:/home/in/:ro".format( os.path.abspath( os.path.dirname(args.seq_file) ) ),
            "-v",
            "{}:/home/out/:rw".format( os.path.abspath( args.outdir ) ),
    ] + ip_file_cmd + [
            "santiagosanchezf/sanntis:ips_nodata",
    ] + args_ + [ "--outdir", "/home/out/" ,"/home/in/{}".format( os.path.basename(args.seq_file) ) ]  

    if not find_executable("docker"):
        raise ValueError("docker is not installed or in PATH")

    try:
        proc = subprocess.Popen(
            cmd,
            shell=False,
        )
        proc.communicate()

    except subprocess.CalledProcessError as err:
        print(err.output)

if __name__ == "__main__":
    main(sys.argv[1:])
