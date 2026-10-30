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

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from sanntis.config import load_recipe
from sanntis.features.vocab import Vocab
from sanntis.model.bundle import BundleConfig, SanntisModel
from sanntis.model.detector import Detector
from sanntis.model.npc import NpcModel, NpcThresholds
from sanntis.model.postfilter import NON_BGC, PostFilter
from sanntis.train.npc_train import PATHWAYS

CLASSES = ["NRPS", "PKS", "ribosomal", "terpene", "saccharide", "other"]


def random_regions(rng: np.random.Generator, tokens: list[str], n: int) -> list[list[list[str]]]:
    return [[list(rng.choice(tokens, size=rng.integers(0, 4))) for _ in range(rng.integers(1, 8))] for _ in range(n)]


def make_dummy_model(seed: int = 0, n_seeds: int = 2, npc: bool = True) -> SanntisModel:
    """A small, fully functional bundle with random weights (unit tests, round-trips)."""
    rng = np.random.default_rng(seed)
    r = load_recipe()
    tokens = [f"IPR{i:06d}" for i in range(40)] + [f"PF{i:05d}" for i in range(20)]
    vocab = Vocab(tokens)
    torch.manual_seed(seed)
    states = [{k: v.clone() for k, v in Detector(len(vocab), r.detector).state_dict().items()}
              for _ in range(n_seeds)]
    regions = random_regions(rng, tokens + ["UNSEEN1", "UNSEEN2"], 60)
    labels = [CLASSES[i % 6] if i % 3 else NON_BGC for i in range(len(regions))]
    pf = PostFilter.fit(regions, labels, vocab, n_estimators=10, n_jobs=1)
    npc_model = None
    if npc:
        pathways = list(PATHWAYS)
        sc = [f"sc{i}" for i in range(5)]
        yp = (rng.random((len(regions), len(pathways))) < 0.3).astype(float)
        yp[:, 0] = np.arange(len(regions)) % 2  # every label has both classes
        ys = (rng.random((len(regions), len(sc))) < 0.3).astype(float)
        ys[:, 0] = np.arange(len(regions)) % 2
        th = NpcThresholds(np.full(len(pathways), 0.4), {"pw": {0.8: 0.3, 0.9: 0.55}, "sc": {0.8: 0.3, 0.9: 0.6}})
        npc_model = NpcModel.fit(regions, np.where(np.arange(len(regions)) % 4 == 0, "euk", "proka"), yp, ys,
                                 np.ones(len(regions)), pathways, sc, th, rf_trees=10, n_jobs=1)
    sigmap = pd.DataFrame({"signature": [f"PF{i:05d}" for i in range(5)], "app": "pfam",
                           "entry": [f"IPR{i:06d}" for i in range(5)]})
    cfg = BundleConfig(detector=r.detector, features=r.features, decode=r.decode,
                       postfilter_drop_threshold=r.postfilter.drop_threshold, bgc_classes=CLASSES,
                       npc_enabled=npc)
    return SanntisModel(config=cfg, vocab=vocab, signature_map=sigmap, detector_states=states, postfilter=pf,
                        npc=npc_model)


@pytest.fixture
def dummy_model() -> SanntisModel:
    return make_dummy_model()


@pytest.fixture
def dummy_bundle(tmp_path, dummy_model):
    return dummy_model.save_pretrained(tmp_path / "bundle", provenance={"mibig": {"release": "test"}})
