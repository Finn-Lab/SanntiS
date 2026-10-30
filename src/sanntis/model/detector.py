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

"""BGC detector: Embedding → Dropout1d → Conv1D (same) → 2-layer BiLSTM → Dropout → per-token logit."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from sanntis.config import DetectorConfig


class _SameConv(nn.Module):
    """Conv1d over (B, T, C) with 'same' padding, also for even kernels (k=4: 1 left, 2 right)."""

    def __init__(self, cin: int, cout: int, k: int):
        super().__init__()
        total = k - 1
        self.pad = (total // 2, total - total // 2)
        self.conv = nn.Conv1d(cin, cout, k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x.transpose(1, 2), self.pad)
        return self.conv(x).transpose(1, 2)


class Detector(nn.Module):
    """Per-position BGC logit over a token window ``x: (B, T)`` → ``(B, T)``."""

    def __init__(self, vocab_size: int, cfg: DetectorConfig):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, cfg.embed_dim, padding_idx=0)
        self.embed_drop = nn.Dropout1d(cfg.embed_dropout)  # drops whole embedding channels
        self.conv = _SameConv(cfg.embed_dim, cfg.conv_filters, cfg.conv_kernel)
        self.rnn = nn.LSTM(cfg.conv_filters, cfg.rnn_units, num_layers=cfg.n_layers, batch_first=True,
                           bidirectional=True, dropout=cfg.dropout if cfg.n_layers > 1 else 0.0)
        self.drop = nn.Dropout(cfg.dropout)
        self.head = nn.Linear(2 * cfg.rnn_units, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed_drop(self.embed(x).transpose(1, 2)).transpose(1, 2)
        h, _ = self.rnn(F.relu(self.conv(h)))
        return self.head(self.drop(h)).squeeze(-1)


def window_starts(n: int, window: int) -> list[int]:
    """Half-overlapping window starts covering ``n`` tokens (the last window is right-aligned)."""
    stride = max(window // 2, 1)
    starts = list(range(0, max(n - window, 0) + 1, stride))
    if starts[-1] + window < n:
        starts.append(n - window)
    return starts


@torch.no_grad()
def predict_tokens(model: Detector, tokens: np.ndarray, window: int, device: torch.device | str = "cpu",
                   batch: int = 256) -> np.ndarray:
    """Per-token probabilities: half-overlapping windows, averaged per token."""
    model.eval()
    n = len(tokens)
    if n == 0:
        return np.zeros(0)
    starts = window_starts(n, window)
    acc = np.zeros(n)
    cnt = np.zeros(n)
    for b in range(0, len(starts), batch):
        sts = starts[b : b + batch]
        x = np.zeros((len(sts), window), dtype=np.int64)
        for i, st in enumerate(sts):
            seg = tokens[st : st + window]
            x[i, : len(seg)] = seg
        p = torch.sigmoid(model(torch.from_numpy(x).to(device)).float()).cpu().numpy()
        for i, st in enumerate(sts):
            k = min(window, n - st)
            acc[st : st + k] += p[i, :k]
            cnt[st : st + k] += 1
    return acc / np.maximum(cnt, 1)
