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

"""Token vocabulary: 3 specials + the selected domain tokens."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

PAD, UNK, SEP = 0, 1, 2
SPECIAL = ["<pad>", "<unk>", "<sep>"]


class Vocab:
    def __init__(self, tokens: Iterable[str]):
        self.itos = SPECIAL + list(tokens)
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        if len(self.stoi) != len(self.itos):
            raise ValueError("duplicate tokens in vocabulary")

    def __len__(self) -> int:
        return len(self.itos)

    @property
    def tokens(self) -> list[str]:
        """Domain tokens without the specials."""
        return self.itos[len(SPECIAL) :]

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps({"specials": SPECIAL, "tokens": self.tokens}, indent=0))

    @classmethod
    def load(cls, path: Path) -> Vocab:
        d = json.loads(Path(path).read_text())
        if d["specials"] != SPECIAL:
            raise ValueError(f"unexpected special tokens {d['specials']}")
        return cls(d["tokens"])


# ── Vocabulary fit ────────────────────────────────────────────────


def frequent_tokens(regions: Sequence[Sequence[Sequence[str]]], min_df: int) -> list[str]:
    """Sorted tokens present in >= ``min_df`` regions (per-CDS token lists)."""
    df = Counter(t for cds in regions for t in {t for g in cds for t in g})
    return sorted(t for t, n in df.items() if n >= min_df)


def _mi_scores(sets: list[set], y, vocab: list[str]):
    """Mutual information between token presence (per entity) and a binary label."""
    n = len(sets)
    idx = {t: i for i, t in enumerate(vocab)}
    n11 = np.zeros(len(vocab))
    n1_ = np.zeros(len(vocab))
    for s, lab in zip(sets, y, strict=True):
        for t in s:
            i = idx.get(t)
            if i is None:
                continue
            n1_[i] += 1
            if lab:
                n11[i] += 1
    ny = float(np.sum(y))
    n10 = n1_ - n11
    n01 = ny - n11
    n00 = n - n1_ - n01
    mi = np.zeros(len(vocab))
    for nij, ni, nj in ((n11, n1_, ny), (n10, n1_, n - ny), (n01, n - n1_, ny), (n00, n - n1_, n - ny)):
        with np.errstate(divide="ignore", invalid="ignore"):
            term = (nij / n) * np.log((n * nij) / (ni * nj))
        mi += np.nan_to_num(term)
    return mi


def select_tokens(entity_tokens: list[list[list[str]]], labels, classes, *, min_df: int = 2, top_k: int | None = 5000,
                  top_k_per_class: int | None = None, non_bgc: str = "non-BGC") -> list[str]:
    """Tokens in >= ``min_df`` entities, then the union of the top-k by MI (BGC vs non-BGC) and the top-k per class
    (one-vs-rest over the BGC classes). Fitted on positives + chunks (no background)."""
    sets = [{t for cds in c for t in cds} for c in entity_tokens]
    vocab = frequent_tokens(entity_tokens, min_df)
    if top_k is None or top_k >= len(vocab):
        return vocab
    k_cls = top_k_per_class if top_k_per_class is not None else max(top_k // 4, 50)
    keep: set = set()
    mi = _mi_scores(sets, np.asarray(labels), vocab)
    keep.update(vocab[i] for i in np.argsort(-mi)[:top_k])
    classes = np.asarray(classes)
    for c in sorted(set(classes) - {non_bgc}):
        mi_c = _mi_scores(sets, (classes == c).astype(int), vocab)
        keep.update(vocab[i] for i in np.argsort(-mi_c)[:k_cls])
    return sorted(keep)
