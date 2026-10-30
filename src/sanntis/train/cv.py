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

"""CV unit: one ``mc`` split × one seed.

Per split: vocabulary fitted on the training positives + chunks; detector trained on the composed training
sequences (positives ×3, chunks, training-genome background) with early stopping on the validation ORF AUCPR;
out-of-fold (OOF) ORF scores on the validation compositions (validation positives + chunks + validation-genome
background); an RF post-filter fitted on the split's training positives + chunks; and the decoded + RF-filtered
OOF regions at every threshold of the PR sweep (so ``tune`` pools splits without re-decoding).

Outputs in ``<out>/``: metrics.json, history.tsv, vocab.json, detector.safetensors, oof.npz, regions.parquet.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from safetensors.torch import save_file

from sanntis.config import Recipe, load_recipe
from sanntis.features.tokens import linearize
from sanntis.features.vocab import Vocab, select_tokens
from sanntis.model.decode import decode
from sanntis.model.detector import Detector
from sanntis.model.postfilter import PostFilter
from sanntis.train.compose import OrfSequence, compose
from sanntis.train.metrics import threshold_grid
from sanntis.train.splits import mc_split
from sanntis.train.trainer import fit_detector, val_orf_scores

log = logging.getLogger(__name__)


@dataclass
class SplitData:
    select: pd.DataFrame  # vocabulary + RF entities (training positives + chunks)
    train: pd.DataFrame  # + training-genome background
    val: pd.DataFrame  # validation positives + chunks + validation-genome background


def split_data(table: pd.DataFrame, split: int, recipe: Recipe) -> SplitData:
    role = mc_split(table, split, n_splits=recipe.splits.n_splits, val_fraction=recipe.splits.val_fraction)
    is_bg = table["kind"] == "background"
    select = table[(role == "train") & ~is_bg]
    train = table[role == "train"]
    val = table[role == "val"]
    return SplitData(select=select, train=train, val=val)


def fit_vocab(select: pd.DataFrame, recipe: Recipe) -> Vocab:
    f = recipe.features
    labelled = select[select["class"].notna()]
    return Vocab(select_tokens(list(labelled["cds_tokens"]), labelled["label"].to_numpy(),
                               labelled["class"].to_numpy(), min_df=f.vocab_min_df, top_k=f.vocab_top_k,
                               top_k_per_class=f.vocab_top_k_per_class))


def fit_postfilter(select: pd.DataFrame, vocab: Vocab, recipe: Recipe, n_jobs: int = -1) -> PostFilter:
    lab = select[select["class"].notna()]
    pf = recipe.postfilter
    return PostFilter.fit(list(lab["cds_tokens"]), list(lab["class"]), vocab, n_estimators=pf.n_estimators,
                          random_state=pf.random_state, n_jobs=n_jobs)


def linearized(seqs: list[OrfSequence], vocab: Vocab, recipe: Recipe):
    f = recipe.features
    return [linearize(s.cds_tokens, vocab, orf_label=s.orf_label, unk_policy=f.unk_policy, gene_sep=f.gene_sep,
                      dedup_contiguous=f.dedup_contiguous) for s in seqs]


def decode_sweep(scores: list[np.ndarray], seqs: list[OrfSequence], pf: PostFilter, recipe: Recipe) -> pd.DataFrame:
    """Decoded + RF-filtered regions at every sweep threshold → one row per kept region."""
    rows = []
    drop = recipe.postfilter.drop_threshold
    for t in threshold_grid(recipe.operating_point.grid):
        for si, (s, seq) in enumerate(zip(scores, seqs, strict=True)):
            regions = decode(s, recipe.decode, threshold=t)
            if not regions:
                continue
            keep, cls, pnon = pf.classify([seq.cds_tokens[a:b] for a, b in regions], drop)
            for (a, b), k, c, pn in zip(regions, keep, cls, pnon, strict=True):
                rows.append({"threshold": t, "seq": si, "kingdom": seq.kingdom, "start": a, "end": b,
                             "kept": bool(k), "bgc_class": c, "p_non_bgc": float(pn)})
    return pd.DataFrame(rows, columns=["threshold", "seq", "kingdom", "start", "end", "kept", "bgc_class",
                                       "p_non_bgc"])


def run_cv_unit(table: pd.DataFrame, split: int, seed: int, out: Path, *, recipe: Recipe | None = None,
                device: str = "auto", max_epochs: int | None = None) -> dict:
    recipe = recipe or load_recipe()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    sd = split_data(table, split, recipe)
    vocab = fit_vocab(sd.select, recipe)
    c = recipe.compose
    train_orf = compose(sd.train, duplicate_positives=c.duplicate_positives, seed=c.seed)
    val_orf = compose(sd.val, duplicate_positives=1, seed=c.seed + 1)
    train_seqs, val_seqs = linearized(train_orf, vocab, recipe), linearized(val_orf, vocab, recipe)
    n_tok = sum(len(s.tokens) for s in train_seqs)
    log.info("split=%d seed=%d vocab=%d train_tokens=%d val_tokens=%d", split, seed, len(vocab), n_tok,
             sum(len(s.tokens) for s in val_seqs))
    fit = fit_detector(train_seqs, len(vocab), recipe, seed=seed, val_seqs=val_seqs,
                       val_orf_labels=[s.orf_label for s in val_orf], device=device, epochs=max_epochs)

    model = Detector(len(vocab), recipe.detector)
    model.load_state_dict(fit.state_dict)
    scores = val_orf_scores(model, val_seqs, recipe.detector.window, "cpu", recipe.decode.agg, recipe.decode.fill)
    pf = fit_postfilter(sd.select, vocab, recipe)
    regions = decode_sweep(scores, val_orf, pf, recipe)

    vocab.save(out / "vocab.json")
    save_file({k: v.contiguous() for k, v in sorted(fit.state_dict.items())}, out / "detector.safetensors")
    pd.DataFrame(fit.history).to_csv(out / "history.tsv", sep="\t", index=False)
    np.savez_compressed(out / "oof.npz", **{f"{s.kingdom}_scores": x for s, x in zip(val_orf, scores, strict=True)},
                        **{f"{s.kingdom}_label": s.orf_label for s in val_orf},
                        **{f"{s.kingdom}_entity": s.orf_entity for s in val_orf})
    regions.to_parquet(out / "regions.parquet", index=False)
    metrics = {"split": split, "seed": seed, "vocab_size": len(vocab), "train_tokens": n_tok,
               "bgc_token_share": float(np.concatenate([s.labels for s in train_seqs]).mean()),
               "best_epoch": fit.best_epoch, "epochs_run": fit.epochs_run, "val_orf_aucpr": fit.val_orf_aucpr,
               "seconds": fit.seconds, "n_val_entities": len(sd.val), "n_train_entities": len(sd.train),
               "kingdoms": [s.kingdom for s in val_orf]}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics
