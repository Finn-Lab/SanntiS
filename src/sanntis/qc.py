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

"""Coverage QC of one input: share of CDS with >= 1 token, share of ``<unk>`` tokens, hits per
app, compared with the bundle's training reference (``config.coverage_reference``) when it has one.

Reference keys (all optional): ``frac_cds_with_tokens``, ``frac_unk_tokens``, ``hits_per_cds`` ({app: rate}); either
flat or per kingdom (``{"proka": {...}, "euk": {...}}``, genome-wide rates differ: disorder is far commoner in
eukaryotes).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from sanntis.features.vocab import Vocab

# Flag thresholds (relative to the reference).
LOW_COVERAGE_RATIO = 0.5  # frac_cds_with_tokens < 0.5 × reference
HIGH_UNK_RATIO, HIGH_UNK_SLACK = 2.0, 0.05  # frac_unk_tokens > 2 × reference + 0.05
LOW_APP_RATIO, APP_MIN_RATE = 0.25, 0.05  # hits/CDS < 0.25 × reference, for apps with reference rate >= 0.05
APP_MIN_CDS = 100  # per-app rates of smaller inputs (a single BGC) are noise: no per-app flags


def coverage(cds_tokens: Sequence[Sequence[str]], app_hits: Counter, vocab: Vocab) -> dict:
    n = len(cds_tokens)
    with_tok = sum(1 for t in cds_tokens if len(t))
    toks = [t for ts in cds_tokens for t in ts]
    unk = sum(1 for t in toks if t not in vocab.stoi)
    return {"n_cds": n, "n_cds_with_tokens": with_tok, "frac_cds_with_tokens": round(with_tok / n, 4) if n else 0.0,
            "n_tokens": len(toks), "n_unk_tokens": unk,
            "frac_unk_tokens": round(unk / len(toks), 4) if toks else 0.0,
            "hits_per_app": dict(sorted(app_hits.items())),
            "hits_per_cds": {a: round(c / n, 4) for a, c in sorted(app_hits.items())} if n else {}}


def flags(qc: dict, reference: dict | None, apps: Sequence[str], kingdom: str = "proka") -> list[str]:
    out = []
    if qc["n_cds"] == 0:
        return ["no_cds"]
    if qc["n_cds_with_tokens"] == 0:
        out.append("no_annotations")
    ref = reference or {}
    if kingdom in ref or "proka" in ref or "euk" in ref:
        ref = ref.get(kingdom) or {}
    r = ref.get("frac_cds_with_tokens")
    if r and qc["frac_cds_with_tokens"] < LOW_COVERAGE_RATIO * r:
        out.append("low_cds_coverage")
    r = ref.get("frac_unk_tokens")
    if r is not None and qc["frac_unk_tokens"] > HIGH_UNK_RATIO * r + HIGH_UNK_SLACK:
        out.append("high_unk_share")
    for a in apps if qc["n_cds"] >= APP_MIN_CDS else ():
        r = (ref.get("hits_per_cds") or {}).get(a)
        if r and r >= APP_MIN_RATE and qc["hits_per_cds"].get(a, 0.0) < LOW_APP_RATIO * r:
            out.append(f"low_{a}_hits")
    return out
