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

"""Per-CDS domain token lists → detector token arrays."""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from sanntis.features.vocab import SEP, UNK, Vocab


def bag_matrix(bags: Sequence[Iterable[Hashable]], index: dict, binary: bool = False) -> sp.csr_matrix:
    """Bag × item count (or presence) matrix over a fixed index; items outside ``index`` are ignored."""
    indptr, indices, values = [0], [], []
    for bag in bags:
        c = Counter(j for j in map(index.get, bag) if j is not None)
        for j in sorted(c):
            indices.append(j)
            values.append(1 if binary else c[j])
        indptr.append(len(indices))
    return sp.csr_matrix((np.asarray(values, dtype=np.int64), np.asarray(indices, dtype=np.int32),
                          np.asarray(indptr, dtype=np.int32)), shape=(len(bags), len(index)))


def token_counts(regions: Sequence[Sequence[Sequence[str]]], index: dict[str, int],
                 binary: bool = False) -> sp.csr_matrix:
    """Region × token count matrix over a fixed vocabulary (regions = per-CDS token lists); unknown tokens ignored."""
    return bag_matrix([[t for g in cds for t in g] for cds in regions], index, binary)


@dataclass
class TokenSequence:
    tokens: np.ndarray  # int64 vocab ids
    labels: np.ndarray  # float32 0/1 per token
    orf_index: np.ndarray  # int64 ORF index per token
    n_orfs: int


def linearize(cds_tokens: Sequence[Sequence[str]], vocab: Vocab, *, orf_label: np.ndarray | None = None,
              unk_policy: str = "unk", gene_sep: bool = True, dedup_contiguous: bool = True) -> TokenSequence:
    """``<sep>`` before every CDS with >= 1 token; contiguous duplicates removed within a gene (on the raw
    tokens, before the vocabulary is applied); out-of-vocabulary tokens → ``<unk>``; CDS without hits emit nothing."""
    toks: list[int] = []
    orfs: list[int] = []
    prev = None
    for i, doms in enumerate(cds_tokens):
        if gene_sep and len(doms):
            toks.append(SEP)
            orfs.append(i)
            prev = None
        for d in doms:
            if dedup_contiguous and d == prev:
                continue
            prev = d
            t = vocab.stoi.get(d)
            if t is None:
                if unk_policy == "drop":
                    continue
                t = UNK
            toks.append(t)
            orfs.append(i)
    orf_idx = np.asarray(orfs, dtype=np.int64)
    labels = (orf_label[orf_idx].astype(np.float32) if orf_label is not None
              else np.zeros(len(toks), dtype=np.float32))
    return TokenSequence(tokens=np.asarray(toks, dtype=np.int64), labels=labels, orf_index=orf_idx,
                         n_orfs=len(cds_tokens))
