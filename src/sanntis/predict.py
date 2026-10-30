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

"""Predictor: CDS table with tokens → detector → decode → RF post-filter → NPC → regions."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from sanntis.features.tokens import linearize
from sanntis.model.bundle import SanntisModel
from sanntis.model.decode import decode, orf_scores
from sanntis.model.detector import predict_tokens

log = logging.getLogger(__name__)

CDS_COLUMNS = ["record_id", "cds_idx", "locus_tag", "start", "end", "tokens"]
REGION_COLUMNS = ["record_id", "region", "start_orf", "end_orf", "start", "end", "score", "bgc_class", "p_non_bgc",
                  "n_orfs", "npc_pathway", "npc_pathway_score", "npc_pathway_confident", "npc_superclass",
                  "npc_superclass_top", "npc_superclass_score", "npc_pathway_key_domains_support",
                  "npc_pathway_key_domains_unsupported"]


@dataclass
class Prediction:
    orf_scores: pd.DataFrame  # record_id, cds_idx, locus_tag, score
    regions: pd.DataFrame  # REGION_COLUMNS; start 1-based, end inclusive


class Predictor:
    """In-process inference with a loaded bundle. ``predict`` takes a CDS table (``CDS_COLUMNS``; ``start`` /
    ``end`` 0-based half-open, ``tokens`` the per-CDS domain token list) and returns pandas frames."""

    def __init__(self, model: SanntisModel, *, kingdom: str = "proka", npc_precision: float = 0.9,
                 device: str = "cpu", threads: int | None = None):
        if threads:
            torch.set_num_threads(threads)
        self.model = model
        self.kingdom = kingdom
        self.npc_precision = npc_precision
        self.device = torch.device(device)
        self.detectors = model.detectors(self.device)
        f = model.config.features
        self.lin = {"unk_policy": f.unk_policy, "gene_sep": f.gene_sep, "dedup_contiguous": f.dedup_contiguous}

    def token_probs(self, cds_tokens: list[list[str]]):
        ts = linearize(cds_tokens, self.model.vocab, **self.lin)
        w = self.model.config.detector.window
        p = np.mean([predict_tokens(m, ts.tokens, w, self.device) for m in self.detectors], axis=0)
        return p, ts

    def predict_record(self, cds: pd.DataFrame) -> tuple[np.ndarray, list[tuple]]:
        """One record (sorted by ``cds_idx``) → ORF scores and kept regions ``(a, b, score, class, p_non)``."""
        cfg = self.model.config
        toks = [list(x) for x in cds["tokens"]]
        p, ts = self.token_probs(toks)
        s = orf_scores(p, ts.orf_index, len(toks), agg=cfg.decode.agg, fill=cfg.decode.fill)
        regions = decode(s, cfg.decode)
        out = []
        if regions:
            keep, cls, pnon = self.model.postfilter.classify([toks[a:b] for a, b in regions],
                                                              cfg.postfilter_drop_threshold)
            for (a, b), k, c, pn in zip(regions, keep, cls, pnon, strict=True):
                if k:
                    out.append((a, b, float(s[a:b].mean()), c, float(pn)))
        return s, out

    def predict(self, cds_table: pd.DataFrame) -> Prediction:
        missing = set(CDS_COLUMNS) - set(cds_table.columns)
        if missing:
            raise ValueError(f"CDS table lacks columns {sorted(missing)}")
        orf_rows, reg_rows = [], []
        for rid, g in cds_table.groupby("record_id", sort=False):
            g = g.sort_values("cds_idx").reset_index(drop=True)
            s, regions = self.predict_record(g)
            orf_rows.append(pd.DataFrame({"record_id": rid, "cds_idx": np.arange(len(g)),
                                          "locus_tag": g["locus_tag"].to_numpy(), "score": s}))
            npc = None
            if regions and self.model.npc is not None and self.model.config.npc_enabled:
                toks = [list(x) for x in g["tokens"]]
                npc = self.model.npc.predict([toks[a:b] for a, b, *_ in regions], self.kingdom, self.npc_precision)
            for i, (a, b, score, c, pn) in enumerate(regions):
                row = {"record_id": rid, "region": i + 1, "start_orf": a, "end_orf": b,
                       "start": int(g["start"].iloc[a]) + 1, "end": int(g["end"].iloc[b - 1]), "score": score,
                       "bgc_class": c if self.model.config.class_enabled else None, "p_non_bgc": pn,
                       "n_orfs": b - a}
                if npc is not None:
                    row.update(npc.iloc[i].to_dict())
                reg_rows.append(row)
        orfs = (pd.concat(orf_rows, ignore_index=True) if orf_rows
                else pd.DataFrame(columns=["record_id", "cds_idx", "locus_tag", "score"]))
        regions = pd.DataFrame(reg_rows, columns=REGION_COLUMNS)
        return Prediction(orf_scores=orfs, regions=regions)

