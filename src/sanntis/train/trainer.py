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

"""Detector training: 200-token windows with a random phase per epoch, token dropout, focal loss,
Adam + gradient clipping; CV fits early-stop on validation ORF-level AUCPR with ReduceLROnPlateau, final fits
run a fixed epoch count at constant lr."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from sanntis.config import Recipe
from sanntis.features.tokens import TokenSequence
from sanntis.features.vocab import UNK
from sanntis.model.decode import orf_scores
from sanntis.model.detector import Detector, predict_tokens
from sanntis.train.losses import sigmoid_focal_loss

log = logging.getLogger(__name__)


class Windows(torch.utils.data.Dataset):
    """Fixed-length windows over token sequences; ``reshuffle`` redraws the phase offset per sequence."""

    def __init__(self, seqs: list[TokenSequence], window: int, stride: int, token_dropout: float = 0.0):
        self.seqs, self.window, self.stride, self.token_dropout = seqs, window, stride, token_dropout
        self.reshuffle(np.random.default_rng(0), phase=False)

    def reshuffle(self, rng: np.random.Generator, phase: bool = True) -> None:
        self.index = []
        for si, s in enumerate(self.seqs):
            off = int(rng.integers(0, self.stride)) if phase else 0
            starts = list(range(-off if off else 0, max(len(s.tokens) - 1, 1), self.stride))
            self.index.extend((si, max(st, 0)) for st in starts)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        si, st = self.index[i]
        s = self.seqs[si]
        x = np.zeros(self.window, dtype=np.int64)
        y = np.zeros(self.window, dtype=np.float32)
        m = np.zeros(self.window, dtype=bool)
        seg = s.tokens[st : st + self.window]
        n = len(seg)
        x[:n], y[:n], m[:n] = seg, s.labels[st : st + self.window], True
        if self.token_dropout > 0:
            drop = (np.random.random(self.window) < self.token_dropout) & m
            x[drop] = UNK
        return {"x": torch.from_numpy(x), "y": torch.from_numpy(y), "mask": torch.from_numpy(m)}


@dataclass
class FitResult:
    state_dict: dict[str, torch.Tensor]
    best_epoch: int
    epochs_run: int
    history: list[dict] = field(default_factory=list)
    val_orf_aucpr: float | None = None
    seconds: float = 0.0


def debug_epoch_cap() -> int | None:
    """``SANNTIS_DEBUG_MAX_EPOCHS``: cap on epochs for tests and smoke runs (recorded in the bundle manifest)."""
    v = os.environ.get("SANNTIS_DEBUG_MAX_EPOCHS")
    return int(v) if v else None


def device_of(pref: str = "auto") -> torch.device:
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


def aucpr_orf(orf_s: np.ndarray, orf_label: np.ndarray) -> float:
    if orf_label.sum() == 0:
        return float("nan")
    return float(average_precision_score(orf_label, orf_s))


def val_orf_scores(model: Detector, seqs: list[TokenSequence], window: int, device, agg: str = "mean",
                   fill: str = "zero") -> list[np.ndarray]:
    return [orf_scores(predict_tokens(model, ts.tokens, window, device), ts.orf_index, ts.n_orfs, agg=agg, fill=fill)
            for ts in seqs]


def fit_detector(train_seqs: list[TokenSequence], vocab_size: int, recipe: Recipe, *, seed: int,
                 val_seqs: list[TokenSequence] | None = None, val_orf_labels: list[np.ndarray] | None = None,
                 epochs: int | None = None, device: str = "auto") -> FitResult:
    """CV fit (``val_seqs`` given: early stopping, best state restored) or final fit (``epochs`` fixed)."""
    t0 = time.time()
    tr, dcfg = recipe.train, recipe.detector
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = device_of(device)
    window = dcfg.window
    stride = max(window // 2, 1)
    ds = Windows(train_seqs, window, stride, tr.token_dropout)
    model = Detector(vocab_size, dcfg).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=tr.optimizer.lr, weight_decay=tr.optimizer.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=tr.scheduler.factor,
                                                       patience=tr.scheduler.patience, min_lr=1e-7)
    amp = tr.amp and dev.type == "cuda"
    rng = np.random.default_rng(seed)
    max_epochs = epochs or tr.early_stopping.max_epochs
    cap = debug_epoch_cap()
    if cap:
        log.warning("SANNTIS_DEBUG_MAX_EPOCHS=%d: epochs capped (tests / smoke runs only, not a release fit)", cap)
        max_epochs = min(max_epochs, cap)
    patience = tr.early_stopping.patience
    gamma, alpha = tr.loss.gamma, tr.loss.alpha
    labels = np.concatenate(val_orf_labels) if val_seqs else None

    best, best_epoch, best_state, bad = -1.0, 0, None, 0
    history: list[dict] = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        ds.reshuffle(rng)
        loader = torch.utils.data.DataLoader(ds, batch_size=tr.batch_size, shuffle=True, num_workers=0)
        tot, nb = 0.0, 0
        for batch in loader:
            x, y, m = (batch[k].to(dev, non_blocking=True) for k in ("x", "y", "mask"))
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=amp):
                logits = model(x)
            loss = sigmoid_focal_loss(logits.float(), y.float(), gamma=gamma, alpha=alpha, mask=m)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tr.optimizer.grad_clip)
            opt.step()
            tot += float(loss.item())
            nb += 1
        rec = {"epoch": epoch, "train_loss": tot / max(nb, 1), "lr": opt.param_groups[0]["lr"]}
        if val_seqs:
            es = tr.early_stopping
            scores = val_orf_scores(model, val_seqs, window, dev, es.orf_agg, es.orf_fill)
            metric = aucpr_orf(np.concatenate(scores), labels)
            rec["val_orf_aucpr"] = metric
            sched.step(metric)
            if metric > best + 1e-4:
                best, best_epoch, bad = metric, epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
        history.append(rec)
        log.info("epoch %s", json.dumps({k: round(v, 5) if isinstance(v, float) else v for k, v in rec.items()}))
        if val_seqs and bad >= patience:
            break
    if best_state is None:
        best_epoch = len(history)
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return FitResult(state_dict=best_state, best_epoch=best_epoch, epochs_run=len(history), history=history,
                     val_orf_aucpr=best if val_seqs else None, seconds=round(time.time() - t0, 1))
