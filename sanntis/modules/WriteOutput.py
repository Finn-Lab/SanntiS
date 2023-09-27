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

import json
import logging
import os
import pickle
from itertools import groupby

import numpy as np
import tensorflow as tf
from joblib import load

from sanntis import __version__
from sanntis import _info
log = logging.getLogger(f"SanntiS.{__name__}")


class Outputs:

    """ write results in file """

    def __init__(self, obj, minimal_output, cli_args, refined_borders, outfile):
        self.annotation = obj
        self.minimal = minimal_output
        self.cli_args = cli_args
        self.ref_b = refined_borders
        self.outfile = outfile
        self.gff3 = []
        self.clusts = []
        self.writeGff3()
        if self.cli_args.antismash_out != "False":
            self.writeAntismashJson()

    def writeAntismashJson(self):
        
        ### tool section
        self.asj = {"tool":{"name":"SanntiS"},"records":[]}
        self.asj["tool"]["version"] = __version__
        self.asj["tool"]["description"] = "{}: {}".format(_info["description"], _info["url"])
        self.asj["tool"]["configuration"] = {
            "greediness":str(self.cli_args.greed),
            "meta":self.cli_args.meta,
        }
        
        ### records section
        # f"{contig}\tSanntiSv{__version__}\tCLUSTER_border\t{start}\t{end}\t.\t.\t.\tID={ID}_{ct2};{typs};partial={edge}"
        records = {}
        for l in self.gff3:
            if "CLUSTER_border" in l:
                continue
            spl = l.split("\t")
            contig, start, end, annot = [spl[x] for x in [0,3,4,8]]
            records.setdefault(spl[0],[]).append({
                "start":int(start),
                "end":int(end),
                "label":annot.split(";")[0].split("=")[-1],
                "details":{
                    "score":annot.split(";")[4].split("=")[-1],
                    "class":annot.split(";")[2].split("=")[-1]
                }
            })
        for contig,bgcs in records.items():
            dct = {}
            dct["name"] = contig
            for bgc in bgcs:
                dct.setdefault("subregions",[]).append({
                    "start":bgc["start"],
                    "end":bgc["end"],
                    "label":bgc["label"],
                    "details":bgc["details"],
                    
                })
            self.asj["records"].append(dct)
            
        with open("{}.antismash.json".format(self.outfile), "w") as h:
            json.dump(self.asj,h)
            
        
    def writeGff3(self):

        log.info("build GFF3")
        log.info(self.outfile)
        type_code = {
            0: "Alkaloid",
            1: "NRP",
            2: "Polyketide",
            3: "RiPP",
            4: "Saccharide",
            5: "Terpene",
            6: "Other",
        }

        for contig in self.annotation.looseClst:

            if self.minimal != "True":
                for ix, f in enumerate(self.annotation.contigsDct[contig]):

                    ID, (start, end) = f
                    sanntisProb = "{:.3f}".format(self.annotation.annResults[contig][ix])
                    self.gff3.append(
                        f"{contig}\tSanntiSv{__version__}\tCDS\t{start}\t{end}\t.\t.\t.\tID={ID};sanntis_probability={sanntisProb}"
                    )

            ct = 1
            for k, g in groupby(
                enumerate(
                    self.annotation.looseClst[contig][
                        : len(self.annotation.contigsDct[contig])
                    ]
                ),
                key=lambda x: x[1],
            ):

                if k == 0:
                    continue

                ID = f"{contig}_sanntis_{ct}"
                ct += 1

                gg = list(list(zip(*g))[0])

                start, end = (
                    self.annotation.contigsDct[contig][gg[0]][1][0],
                    self.annotation.contigsDct[contig][gg[-1]][1][1],
                )

                edge = "{}{}".format(
                    1 if gg[0] == 0 else 0,
                    1 if gg[-1] == len(self.annotation.contigsDct[contig]) - 1 else 0,
                )

                typs = self.annotation.typesClst[contig][gg[0]]
                # typs = "nearest_MiBIG={};nearest_MiBIG_class={};nearest_MiBIG_jacardDistance={}".format(typs[0],typs[1],typs[2])
                self.gff3.append(
                    f"{contig}\tSanntiSv{__version__}\tCLUSTER\t{start}\t{end}\t.\t.\t.\tID={ID};{typs};partial={edge}"
                )

                if self.ref_b == "True":

                    ct2 = 1
                    for k2, g2 in groupby(
                        zip(gg, self.annotation.borderClst[contig][gg]),
                        key=lambda x: x[1],
                    ):
                        if k2 == 0:
                            continue
                        gg2 = list(list(zip(*g2))[0])
                        start, end = (
                            self.annotation.contigsDct[contig][gg2[0]][1][0],
                            self.annotation.contigsDct[contig][gg2[-1]][1][1],
                        )
                        edge = "{}{}".format(
                            1 if gg[0] == 0 else 0,
                            1
                            if gg[-1] == len(self.annotation.contigsDct[contig]) - 1
                            else 0,
                        )

                        self.gff3.append(
                            f"{contig}\tSanntiSv{__version__}\tCLUSTER_border\t{start}\t{end}\t.\t.\t.\tID={ID}_{ct2};{typs};partial={edge}"
                        )
                        ct2 += 1

        log.info(f"Writing output to file {self.outfile}")
        with open(self.outfile, "w") as h:

            h.write("##gff-version 3\n")

            key = lambda x: (
                x.split("\t")[0],
                int(x.split("\t")[3]),
                "Z" if x.split("\t")[2] == "CDS" else x.split("\t")[2],
            )
            
            self.gff3 = sorted(self.gff3, key=key)
            for l in self.gff3:

                h.write(f"{l}\n")
