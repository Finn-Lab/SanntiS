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

"""``finalize``: the recipe fitted on the whole pool → model bundle.

- vocabulary on all positives + chunks; one composed corpus (positives ×3, chunks, all background, seed 0);
- detector seeds (0, 1, 2) trained for the tuned epoch count at constant lr (one unit per seed);
- RF post-filter on all positives (primary class) + chunks;
- NPC members on all labelled loci + the OOF regions at t* (label space fitted on the loci), CV-tuned thresholds;
- NPC pathway key-domain rules on all labelled loci, gated pathways from ``tune``;
- bundle: decode threshold = t*.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from safetensors.torch import load_file, save_file

from sanntis.config import Recipe, load_recipe
from sanntis.features.vocab import Vocab
from sanntis.model.bundle import BundleConfig, SanntisModel
from sanntis.model.npc import NpcThresholds
from sanntis.train import key_domains as kd
from sanntis.train.compose import compose
from sanntis.train.cv import fit_postfilter, fit_vocab, linearized
from sanntis.train.npc_train import Examples, LabelSpace, fit_members, labelled, locus_examples, region_examples
from sanntis.train.trainer import debug_epoch_cap, fit_detector

log = logging.getLogger(__name__)


def corpus(table: pd.DataFrame, recipe: Recipe):
    select = table[table["kind"] != "background"]
    vocab = fit_vocab(select, recipe)
    seqs = compose(table, duplicate_positives=recipe.compose.duplicate_positives, seed=recipe.compose.seed)
    return select, vocab, linearized(seqs, vocab, recipe)


def run_seed_unit(table: pd.DataFrame, seed: int, epochs: int, out: Path, *, recipe: Recipe | None = None,
                  device: str = "auto") -> dict:
    recipe = recipe or load_recipe()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    _, vocab, seqs = corpus(table, recipe)
    fit = fit_detector(seqs, len(vocab), recipe, seed=seed, epochs=epochs, device=device)
    save_file({k: v.contiguous() for k, v in sorted(fit.state_dict.items())}, out / f"detector-seed{seed}.safetensors")
    vocab.save(out / f"vocab-seed{seed}.json")
    n_tok = sum(len(s.tokens) for s in seqs)
    m = {"seed": seed, "epochs": epochs, "seconds": fit.seconds, "corpus_tokens": n_tok,
         "bgc_token_share": float(np.concatenate([s.labels for s in seqs]).mean()), "vocab_size": len(vocab)}
    (out / f"seed{seed}.json").write_text(json.dumps(m, indent=2))
    pd.DataFrame(fit.history).to_csv(out / f"history-seed{seed}.tsv", sep="\t", index=False)
    return m


def build_bundle(table: pd.DataFrame, tune: dict, seed_dir: Path, out: Path, *, signature_map: pd.DataFrame,
                 recipe: Recipe | None = None, labels: pd.DataFrame | None = None, parents: dict | None = None,
                 oof_regions: pd.DataFrame | None = None, provenance: dict | None = None,
                 report_dir: Path | None = None, coverage_reference: dict | None = None,
                 entry_types: dict[str, str] | None = None) -> SanntisModel:
    recipe = recipe or load_recipe()
    seed_dir = Path(seed_dir)
    seeds = recipe.train.seeds
    vocab = Vocab.load(seed_dir / f"vocab-seed{seeds[0]}.json")
    for s in seeds[1:]:  # same corpus → same vocabulary for every seed
        assert Vocab.load(seed_dir / f"vocab-seed{s}.json").itos == vocab.itos
    states = [load_file(seed_dir / f"detector-seed{s}.safetensors") for s in seeds]
    select = table[table["kind"] != "background"]
    pf = fit_postfilter(select, vocab, recipe)
    decode = recipe.decode.model_copy(update={"threshold": float(tune["t_star"])})
    classes = list(recipe.postfilter.classes)
    n_lab = int(select.loc[select["kind"] == "positive", "class"].notna().sum())
    class_on = n_lab >= recipe.npc.min_labelled
    npc = None
    if labels is not None and parents is not None and "npc" in tune:
        loci = table[table["kind"] == "positive"].set_index("entity_id")
        lab_ids = labelled(labels, list(loci.index))
        if len(lab_ids) >= recipe.npc.min_labelled:
            ls = LabelSpace.fit(labels, lab_ids, parents, recipe.npc.min_support)
            train = Examples.concat([locus_examples(loci, labels, lab_ids),
                                     region_examples(oof_regions, labels, lab_ids) if oof_regions is not None
                                     else Examples([], [], np.zeros(0, dtype=object), [])])
            th = NpcThresholds(np.asarray(tune["npc"]["pathway_thresholds"]),
                               {lv: {float(k): float(v) for k, v in d.items()}
                                for lv, d in tune["npc"]["abstention"].items()})
            npc = fit_members(train, labels, ls, th, recipe.npc, seed=0)
            kd_cfg = recipe.npc.key_domains
            if "key_domains" in tune["npc"]:  # tune.json of this run has the CV gate
                fit = locus_examples(loci, labels, lab_ids)
                rules = kd.fit_rules(fit.cds, ls.encode(labels, fit.ids)[0], ls.pathways, kd_cfg,
                                     allowed=kd.allowed_tokens(kd_cfg, entry_types),
                                     background=table.loc[table["kind"] == "background", "cds_tokens"].tolist())
                npc.key_domains = kd.key_domains(rules, ls.pathways, kd_cfg, tune["npc"]["key_domains"])
            provenance = {**(provenance or {}), "npc_training": {
                "loci": len(lab_ids), "regions": len(train) - len(lab_ids), "superclasses": len(ls.superclasses)}}
    cfg = BundleConfig(detector=recipe.detector, features=recipe.features, decode=decode,
                       postfilter_drop_threshold=recipe.postfilter.drop_threshold, bgc_classes=classes,
                       npc_enabled=npc is not None, class_enabled=class_on,
                       coverage_reference=coverage_reference or {})
    model = SanntisModel(config=cfg, vocab=vocab, signature_map=signature_map, detector_states=states,
                         postfilter=pf, npc=npc, report_dir=report_dir)
    counts = {f"{k}_{g}": int(n) for (k, g), n in table.groupby(["kind", "kingdom"]).size().items()}

    if debug_epoch_cap():
        provenance = {**(provenance or {}), "debug_max_epochs": debug_epoch_cap()}
    model.save_pretrained(out, provenance={**(provenance or {}), "counts": counts,
                                           "tuned": {"epochs": tune["epochs"], "t_star": tune["t_star"],
                                                     "threshold_low": tune["threshold_low"],
                                                     "floor_metric": tune["floor_metric"], "floor": tune["floor"],
                                                     "min_pred_ratio": tune["min_pred_ratio"],
                                                     "floor_met": tune["floor_met"],
                                                     "npc": tune.get("npc")}})
    return model
