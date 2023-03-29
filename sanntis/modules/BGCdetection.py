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
import re
import pickle
from itertools import groupby

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from joblib import load

from sanntis import __version__, _params

log = logging.getLogger(f"SanntiS.{__name__}")


class AnnotationFilesToEmerald:
    """Transform external tool's files into SanntiS STR"""

    def __init__(self):

        self.entriesDct = {}
        self.contigsDct = {}
        self.annDct = {}
        self.typeDct = {}
        self.annResults = {}
        post_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "models",
            "post_filters.pickle",
        )
        with open(post_file, "rb") as h:
            self.vocab, self.typeModel, self.mbdoms = pickle.load(h)

    def transformIPS(self, ipsFile):

        if not os.path.isfile(ipsFile):
            log.exception(f"{ipsFile} file not found")

        with open(ipsFile, "r") as h:

            lines = h.readlines()
            fmt = "gff" if lines[0][:5] == "##gff" else "tsv"

            for l in lines:
        
                if fmt == "tsv":

                    spl = l.split("\t")

                    try:
                        self.entriesDct.setdefault(spl[0], []).append(
                            spl[11] if spl[11] != '-' else spl[4]
                        )
                    except IndexError:
                        log.warning(f"WARNING. Incorrect number of fields in some lines of InterProScan TSV. check IPS version. FILE: {ipsFile} line: {line}.")
                if fmt == "gff":
                    spl = l.split("\t")
                    if len(spl) > 3 and spl[2]=='protein_match':
                        self.entriesDct.setdefault(spl[0],[]).append(
                                re.split("InterPro:|\"",spl[-1])[-2] if 'InterPro' in spl[-1]     else re.split("Name=|;",spl[-1])[-2]
                        )

    def transformEmeraldHmm(self, hmmFile):

        log.info(f"processing {hmmFile}")
        if not os.path.isfile(hmmFile):
            log.exception(f"{hmmFile} file not found")

        with open(hmmFile, "r") as h:

            for l in h:

                if l[0] == "#":
                    continue

                spl = l.split()
                self.entriesDct.setdefault(spl[3], []).append(spl[0])

    def transformCDSpredToCDScontigs(self, cdsPredFile, f):

        if not os.path.isfile(cdsPredFile):
            log.exception(f"{cdsPredFile} file not found")

        with open(cdsPredFile, "r") as h:

            if f == "fasta":

                for l in h:

                    if l[0] != ">":
                        continue

                    spl = l.split()
                    start, end = int(spl[2]), int(spl[4])

                    self.contigsDct.setdefault(
                        "_".join(spl[0].split("_")[:-1])[1:], []
                    ).append((spl[0][1:].strip(), (start, end)))

            elif f == "genbank":

                from Bio import SeqIO

                recs = list(SeqIO.parse(open(cdsPredFile, "r"), "gb"))

                for rec in recs:
                    for f in rec.features:
                        if f.type == "CDS":


                            start, end = int(f.location.start) + 1, int(f.location.end)

                            self.contigsDct.setdefault(rec.id, []).append(
                                (
                                    f.qualifiers["protein_id"][0].strip()
                                    if "protein_id" in f.qualifiers
                                    else f.qualifiers["locus_tag"][0].strip(),
                                    (start, end),
                                )
                            )

    def buildMatrices(self):

        for contig in self.contigsDct:

            cdss = self.contigsDct[contig]

            samps = (
                len(cdss)
                if not _params["shape"]
                else ((len(cdss) // _params["shape"]) + 1) * _params["shape"]
            )

            self.annDct[contig] = np.zeros((samps, len(self.vocab)))

            for ix, cds in enumerate(cdss):

                name = cds[0]
                if name not in self.entriesDct:
                    continue

                modiAnn = [
                    self.vocab[x] for x in self.entriesDct[name] if x in self.vocab
                ]
                self.annDct[contig][ix][modiAnn] = 1


    def predictAnn(self, colapseFunc=max):

        log.info("Predict BGC probability w/ TensorFlow")
        model_BGC_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "models", "sanntis.h5"
        )

        model = tf.keras.models.load_model(
            model_BGC_file, custom_objects={"robustLoss": {}}
        )

        for contig in self.annDct:
            mat = self.annDct[contig]
            xva_, vaIx = self.transformMat(mat)
            vaS = (xva_.shape[0] // _params["shape"]) * _params["shape"]
            xva1 = xva_[:vaS].reshape(vaS // _params["shape"], _params["shape"])
            xva = np.append(
                xva1,
                [list(xva_[vaS:]) + [0] * (_params["shape"] - (vaIx.shape[0] - vaS))],
                axis=0,
            )
            self.predict_ = model.predict(xva)
            predict = self.predict_.reshape(vaS + _params["shape"])
            predict_str_, nix = self.partialReStrMat(vaIx, predict, func=colapseFunc)
            self.annResults[contig] = self.projectRes(predict_str_, nix, mat.shape[0])

    def transformMat(self, mat):

        scuash = np.array(list(map(lambda x: np.where(x == 1)[0], mat)))
        nli, nix = [], []
        for ix, x in enumerate(scuash):
            if x.shape[0] == 0:
                continue
            nli.extend(x)
            nix.extend([ix] * x.shape[0])
        sups = [
            (k, list(zip(*g))[1][0])
            for k, g in groupby(zip(nli, nix), key=lambda x: x[0])
        ]
        if len(nli) == 0:
            return np.array([0]), np.array([0])
        a, b = list(zip(*sups))
        a, b = np.array(a), np.array(b)
        return a, b

    def projectRes(self, res, ixes, lenOri):

        nres_ = np.zeros(lenOri)
        nres_[ixes] = res
        prev = 0
        rprev = 0
        for ix, r in zip(ixes, res):
            if ix - 1 == prev:
                prev = ix
                rprev = r
                continue
            for ixx, s in list(enumerate(np.linspace(rprev, r, ix - prev + 1)))[1:-1]:
                nres_[prev + ixx] = s
            prev = ix

        return nres_

    def partialReStrMat(self, Ix, res, func):

        nres, nix = [], []
        for k, g in groupby(zip(res, Ix), key=lambda x: x[1]):

            gg = list(list(zip(*g))[0])
            nix.append(k)
            nres.append(func(gg))
        return np.array(nres), np.array(nix)

    def defineLooseClusters(self, score, g):
        
        log.info("Define Clusters")
        self.bridged, self.looseClst, self.borderClst, self.typesClst = {}, {}, {}, {}
        self.score = _params["greed"][str(g)] if score == None else score

        for contig in self.annResults:
            self.looseClst[contig] = np.array(
                self.rmLessThan(
                    self.fillGap(
                        np.where(self.annResults[contig] < self.score, 0, 1),
                        _params["fill"],
                    ),
                    _params["rmless"],
                )
            )
            self.borderClst[contig] = np.where(
                self.annResults[contig] < _params["thBorder"], 0, 1
            )
            self.typesClst[contig] = np.empty(
                len(self.annResults[contig]), dtype=object
            )

    def rmLessThan(self, contig, n):
        newContig = []
        for k, g in groupby(contig):
            if k == 0:
                newContig.extend(list(g))
            else:
                p = list(g)
                if len(p) <= n:
                    newContig.extend([0 for x in p])
                else:
                    newContig.extend(p)
        return newContig

    def fillGap(self, contig, gap):
        newContig = []
        for k, g in groupby(contig):
            gg = list(g)
            if k == 0 and len(gg) <= gap:
                newContig.extend([1 for x in gg])
            else:
                newContig.extend(gg)
        return newContig

    def scoreFunc(self, x, b, m):
        y = b+(m*x)
        return 0 if y<0 else 1 if y>1 else y
    
    def predictType(self):

        log.info("Predict BGC classes")

        type_score = self.scoreFunc(self.score, _params["score_b"], _params["score_m"])
        log.info(f"Positive class model threshold: {type_score}")
        log.info(f"type:{type_score} score {self.score}")
        locations, matrix, typeLi = [], [], []
        claa = [
            "Alkaloid",
            "NRP",
            "Polyketide",
            "RiPP",
            "Saccharide",
            "Terpene",
            "Other",
        ]

        for contig in self.contigsDct:
            for k, g in groupby(enumerate(self.looseClst[contig]), key=lambda x: x[1]):
                if k == 0:
                    continue
                gg = list(list(zip(*g))[0])
                tmat_ = np.sum(self.annDct[contig][gg], axis=0)
                tmat = np.where(tmat_ > 0, 1, 0)
                locations.append((contig, gg))
                matrix.append(tmat)
        if len(matrix) > 0:
            pred = np.empty((len(matrix), len(claa)))
            for nc, cla in enumerate(claa):
                pred[:, nc] = self.typeModel[cla].predict_proba(matrix)[:, 1]
        else:
            pred = []

        for ix, p in enumerate(pred):
            
            contig, gg = locations[ix]

            if not (np.max(p[[0, 1, 2, 3, 4, 5, 6]]) >= type_score):
                self.looseClst[contig][gg] = 0
                self.borderClst[contig][gg] = 0

            else:
                nearest_ = self.near_classifer(set(np.where(matrix[ix] == 1)[0]))
                nearest = "nearest_MiBIG={};nearest_MiBIG_class={};nearest_MiBIG_diceDistance={:.3f};score={:.3f}".format(
                    nearest_[0], nearest_[1], nearest_[2], np.max(p[[0, 1, 2, 3, 4, 5, 6]])
                )
                self.typesClst[contig][gg] = nearest

    def near_classifer(self, doms):
        mb_ord = [(b, c, self.diceDistance(doms, mbd)) for b, c, mbd in self.mbdoms]
        nearest = sorted(mb_ord, key=lambda x: x[2])[0]
        return nearest

    def jacardDistance(self, a, b):
        try:
            return 1 - ((len(set(a) & set(b))) / len(set(a) | set(b)))
        except:
            return 1

    def diceDistance(self, a, b):
        try:
            return 1 - ( (2*len(set(a)&set(b))) / (len(set(a))+len(set(b))) )
        except:
            return 1
