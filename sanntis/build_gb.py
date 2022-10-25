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
import sys
import glob
import os
from Bio import SeqIO

def main(args=None):

    parser = argparse.ArgumentParser(description="build_gb. Tool to build genbank format files ")
    parser.add_argument(
        "-n",
        dest="nuc_f",
        default=None,
        type=str,
        help="FASTA file with nucleotide sequence(s)",
        metavar="FILE",
        required=True,
    )
    parser.add_argument(
        "-a",
        dest="pro_f",
        default=None,
        type=str,
        help="prodigal output FASTA file with aminoacids sequences",
        metavar="FILE",
        required=True,
    )
    parser.add_argument(
        "-o",
        dest="out",
        default=None,
        type=str,
        help="output genebank format file",
        metavar="FILE",
        required=True,
    )
    
    args = parser.parse_args(args)

    fna = {rec.id:rec.seq for rec in SeqIO.parse(open(args.nuc_f),'fasta')}
    faa = list(SeqIO.parse(open(args.pro_f),'fasta'))

    feats = {}

    for ff in faa:
        spl = ff.description.split()
        s,e,st = int(spl[2])-1,int(spl[4]),int(spl[6])
        
        from Bio import SeqFeature
        start_pos = SeqFeature.ExactPosition(s)
        end_pos = SeqFeature.ExactPosition(e)

        from Bio.SeqFeature import FeatureLocation
        feature_location = FeatureLocation(start_pos,end_pos)

        feature_type = "CDS"

        from Bio.SeqFeature import SeqFeature
        qual = {}
        qual['translation'] = str(ff.seq).replace("*","")
        qual['protein_id'] = str(ff.id)
        feature = SeqFeature(feature_location,type=feature_type,qualifiers=qual)

        cont = "_".join(ff.id.split('_')[:-1])
        feats.setdefault(cont,[]).append(feature)

    recs = []
    for cont,v in feats.items():

        sequence = fna.get(cont)

        from Bio.SeqRecord import SeqRecord
        sequence_record = SeqRecord(sequence)    
        sequence_record.id = cont    
        sequence_record.name = cont    
        sequence_record.description = cont    
        sequence_record.annotations={"molecule_type": "DNA"}

        sequence_record.features = v

        recs.append(sequence_record)



    with open(args.out, 'w') as h:
        SeqIO.write(recs, h, 'genbank')
    print(f"Done!.. outfile :{args.out}")

if __name__ == "__main__":
    main(sys.argv[1:])