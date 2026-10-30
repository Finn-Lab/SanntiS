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

"""Pathway key-domain rules (recipe ``npc.key_domains``; inference in :mod:`sanntis.model.key_domains`).

- Candidates, per pathway c, one-vs-rest on labelled BGC loci: tokens (InterPro entry, else signature) and
  unordered token pairs present anywhere in the same locus. A rule seen in ``a`` loci of c and ``b`` loci of other
  pathways is scored by the 95 % Wilson lower bound of its precision a / (a + b). A pair is a candidate only when it
  scores higher than both its tokens.
- Tokens: only InterPro entries of the ``entry_types`` (Family, Domain, sites, ...) are candidates,
  so homologous superfamilies, unintegrated member-database signatures and MobiDB never make a rule (types from the
  run's InterPro ``entries.json``).
- Admission: ``a >= min_support`` and score >= ``tau``; the rule must also be in at most ``background_max`` of the
  non-BGC background entities (``kind == background``: the split's training ones in CV, all of them for the final
  rules).
- Short list: greedy set cover of the pathway's loci by its admitted rules (largest gain, ties by score).
- CV (``tune``): rules fitted on each ``mc`` split's training loci, evaluated on its OOF regions with the NPC
  calls at cross-split thresholds. A pathway is gated when its pooled OOF coverage (share of its true regions
  with a rule present) is below ``min_coverage``; reported: coverage, false-alarm rate (flagged correct calls /
  correct calls) and catch rate (flagged wrong calls / wrong calls), gated and ``na`` calls counted as not
  flagged.
- Final rules (``finalize``): every labelled locus.

The setting was chosen in a nested-selection CV experiment during development.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from sanntis.config import KeyDomainsConfig
from sanntis.features.tokens import bag_matrix
from sanntis.model.key_domains import ACTIVE, GATED, NO_RULES, KeyDomains
from sanntis.model.npc import NpcThresholds, pathway_calls, remap

Z = 1.959964


def bag(cds) -> frozenset[str]:
    return frozenset(t for g in cds for t in g)


def precision(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    return a / np.maximum(a + np.asarray(b, dtype=float), 1)


def wilson_lower_bound(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """95 % Wilson lower bound of the precision a / (a + b)."""
    n = np.maximum(np.asarray(a, dtype=float) + np.asarray(b, dtype=float), 1)
    p = precision(a, b)
    c = p + Z * Z / (2 * n)
    r = Z * np.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n))
    return (c - r) / (1 + Z * Z / n)


def admitted(bags: Sequence[frozenset[str]], Y: np.ndarray, categories: Sequence[str],
             cfg: KeyDomainsConfig) -> pd.DataFrame:
    """→ category, t1, t2 ('' for a token), a, b, P, N, precision, score: the admitted rules of every category."""
    tau, s_min = cfg.tau, cfg.min_support
    Y = np.asarray(Y) > 0.5
    vocab = sorted({t for b in bags for t in b})
    X = bag_matrix(bags, {t: i for i, t in enumerate(vocab)}).tocsc()
    df_all = np.asarray(X.sum(0)).ravel()
    rows = []
    for j, c in enumerate(categories):
        y = Y[:, j]
        P = int(y.sum())
        N = len(bags) - P
        if s_min > P:
            continue
        a1 = np.asarray(X[y].sum(0)).ravel()
        cand = np.nonzero(a1 >= s_min)[0]
        if not len(cand):
            continue
        b1 = df_all[cand] - a1[cand]
        s1 = wilson_lower_bound(a1[cand], b1)
        parts = [pd.DataFrame({"t1": [vocab[i] for i in cand], "t2": "", "a": a1[cand], "b": b1, "score": s1})]
        if len(cand) > 1:
            Xc = X[:, cand]
            A = (Xc[y].T @ Xc[y]).toarray()
            T = (Xc.T @ Xc).toarray()
            iu, ju = np.triu_indices(len(cand), 1)
            a2 = A[iu, ju]
            keep = a2 >= s_min
            iu, ju, a2 = iu[keep], ju[keep], a2[keep]
            b2 = T[iu, ju] - a2
            s2 = wilson_lower_bound(a2, b2)
            ok = s2 > np.maximum(s1[iu], s1[ju])
            parts.append(pd.DataFrame({"t1": [vocab[cand[i]] for i in iu[ok]], "t2": [vocab[cand[i]] for i in ju[ok]],
                                       "a": a2[ok], "b": b2[ok], "score": s2[ok]}))
        t = pd.concat(parts, ignore_index=True)
        t = t[t["score"] >= tau]
        rows.append(t.assign(category=c, P=P, N=N, precision=precision(t["a"], t["b"])))
    cols = ["category", "t1", "t2", "a", "b", "P", "N", "precision", "score"]
    return pd.concat(rows, ignore_index=True)[cols] if rows else pd.DataFrame(columns=cols)


def present(rules: pd.DataFrame, bags: Sequence[frozenset[str]]) -> np.ndarray:
    """(n_bags x n_rules) bool: the rule's token(s) present in the bag."""
    toks = sorted((set(rules["t1"]) | set(rules["t2"])) - {""})
    index = {t: i for i, t in enumerate(toks)}
    Xb = bag_matrix(bags, index).toarray() > 0
    h = Xb[:, rules["t1"].map(index).to_numpy(dtype=int)] if len(rules) else np.zeros((len(bags), 0), bool)
    two = (rules["t2"] != "").to_numpy()
    if two.any():
        h[:, two] &= Xb[:, rules.loc[two, "t2"].map(index).to_numpy(dtype=int)]
    return h


def short_list(rules: pd.DataFrame, bags: Sequence[frozenset[str]], Y: np.ndarray,
               categories: Sequence[str]) -> pd.DataFrame:
    """Greedy set cover of each category's examples by its admitted rules (ties: higher score, then order)."""
    out = []
    for j, c in enumerate(categories):
        r = rules[rules["category"] == c]
        if not len(r):
            continue
        H = present(r, [b for b, k in zip(bags, np.asarray(Y)[:, j] > 0.5, strict=True) if k])
        order = np.argsort(-r["score"].to_numpy(), kind="stable")
        covered = np.zeros(H.shape[0], bool)
        chosen = []
        while True:
            gain = (H[:, order] & ~covered[:, None]).sum(0)
            g = int(np.argmax(gain)) if len(gain) else 0
            if not len(gain) or gain[g] == 0:
                break
            chosen.append(order[g])
            covered |= H[:, order[g]]
        out.append(r.iloc[chosen])
    return pd.concat(out, ignore_index=True) if out else rules.iloc[:0]


def allowed_tokens(cfg: KeyDomainsConfig, entry_types: dict[str, str] | None) -> set[str] | None:
    """Candidate tokens under ``cfg.entry_types`` (None when empty: every token). ``entry_types``: InterPro entry →
    type."""
    if not cfg.entry_types:
        return None
    if entry_types is None:
        raise ValueError("npc.key_domains.entry_types needs the InterPro entry types (entries.json)")
    keep = set(cfg.entry_types)
    return {acc for acc, t in entry_types.items() if t in keep}


def background_rate(rules: pd.DataFrame, bags: Sequence[frozenset[str]], chunk: int = 2000) -> np.ndarray:
    """Share of the background bags carrying each rule (both tokens for a pair)."""
    n = np.zeros(len(rules))
    for i in range(0, len(bags), chunk):
        n += present(rules, bags[i:i + chunk]).sum(0)
    return n / max(len(bags), 1)


def fit_rules(cds: Sequence, Y: np.ndarray, categories: Sequence[str], cfg: KeyDomainsConfig, *,
              background: Sequence, allowed: set[str] | None = None) -> pd.DataFrame:
    """Short-list rules from labelled loci (per-CDS token lists, (n x C) pathway labels). ``background``: per-CDS
    token lists of the non-BGC background entities; ``allowed``: candidate tokens (:func:`allowed_tokens`)."""
    bags = [bag(c) for c in cds]
    if allowed is not None:
        bags = [b & allowed for b in bags]
    rules = admitted(bags, Y, categories, cfg)
    rules = rules.assign(bg_rate=background_rate(rules, [bag(c) for c in background]))
    rules = rules[rules["bg_rate"] <= cfg.background_max]
    return short_list(rules, bags, Y, categories)


def has_rule(rules: pd.DataFrame, cds: Sequence, categories: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """→ (n x C) bool: some rule of the category present; (C,) bool: the category has rules."""
    ci = {c: j for j, c in enumerate(categories)}
    A = np.zeros((len(rules), len(categories)), bool)
    A[np.arange(len(rules)), rules["category"].map(ci).to_numpy(dtype=int)] = True
    H = present(rules, [bag(c) for c in cds])
    return (H.astype(np.int32) @ A.astype(np.int32)) > 0, A.any(0)


# ── CV evaluation (tune) ─────────────────────────────────────────────────────


def _rates(fc: int, c: int, fw: int, w: int) -> dict:
    ef = fw / (fc + fw) if fc + fw else None
    eu = (w - fw) / ((c - fc) + (w - fw)) if (c - fc) + (w - fw) else None
    return {"calls": c + w, "correct": c, "wrong": w, "flag_correct": fc, "flag_wrong": fw,
            "false_alarm": fc / c if c else None, "catch": fw / w if w else None, "err_flagged": ef,
            "err_unflagged": eu, "err_ratio": ef / eu if ef is not None and eu else None}


def oof_calls(parts: Sequence[dict], thresholds_of, tier: float, categories: Sequence[str]) -> pd.DataFrame:
    """``parts``: per split {preds: CvPreds, has, with_rules}; ``thresholds_of(k)`` → the split's cross-split
    :class:`NpcThresholds`. → one row per pathway call: split, category, na, flag, correct, confident."""
    rows = []
    for k, d in enumerate(parts):
        th: NpcThresholds = thresholds_of(k)
        p = d["preds"]
        Pm = remap(p.pp, th.pathway)
        called = pathway_calls(Pm)
        conf = Pm.max(1) >= th.abstention["pw"].get(tier, np.inf)
        r, j = np.nonzero(called)
        na = ~d["with_rules"][j]
        rows.append(pd.DataFrame({"split": k, "row": r, "category": np.asarray(categories, dtype=object)[j],
                                  "na": na, "flag": ~na & ~d["has"][r, j], "correct": p.yp[r, j] > 0.5,
                                  "confident": conf[r]}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["split", "row", "category", "na", "flag", "correct", "confident"])


def coverage(parts: Sequence[dict], categories: Sequence[str]) -> pd.DataFrame:
    """Pooled OOF coverage per category: true regions, true regions with a rule of the category present."""
    n_true = sum(((d["preds"].yp > 0.5).sum(0) for d in parts), np.zeros(len(categories), int))
    n_cov = sum((((d["preds"].yp > 0.5) & d["has"]).sum(0) for d in parts), np.zeros(len(categories), int))
    return pd.DataFrame({"category": list(categories), "n_true": n_true, "n_covered": n_cov,
                         "coverage": np.where(n_true > 0, n_cov / np.maximum(n_true, 1), np.nan)})


def evaluate(calls: pd.DataFrame, cov: pd.DataFrame, min_coverage: float) -> dict:
    """Gate + OOF rates (all calls, confident tier, per category)."""
    gated = sorted(cov.loc[~(cov["coverage"] >= min_coverage), "category"])
    d = calls.assign(flag=calls["flag"] & ~calls["category"].isin(gated),
                     na=calls["na"] | calls["category"].isin(gated))

    def rates(x: pd.DataFrame) -> dict:
        ok, f = x["correct"].astype(bool), x["flag"].astype(bool)
        return {**_rates(int((f & ok).sum()), int(ok.sum()), int((f & ~ok).sum()), int((~ok).sum())),
                "na_rate": float(x["na"].mean()) if len(x) else None}

    per = []
    for _, c in cov.iterrows():
        x = d[d["category"] == c["category"]]
        per.append({"category": c["category"], "n_true_regions": int(c["n_true"]),
                    "oof_coverage": None if np.isnan(c["coverage"]) else float(c["coverage"]),
                    "gated": c["category"] in gated, **rates(x)})
    return {"gated": gated, "all": rates(d), "confident": rates(d[d["confident"].astype(bool)]),
            "per_category": per}


def key_domains(rules: pd.DataFrame, categories: Sequence[str], cfg: KeyDomainsConfig, cv: dict | None) -> KeyDomains:
    """Bundle object from the final rules and the CV result (``cv`` = ``tune.json`` ``npc.key_domains``)."""
    per = {c["category"]: c for c in (cv or {}).get("per_category", [])}
    gated = set((cv or {}).get("gated", []))
    cats = {}
    for c in categories:
        r = rules[rules["category"] == c]
        status = GATED if c in gated else ACTIVE if len(r) else NO_RULES
        cats[c] = {"status": status, "oof_coverage": per.get(c, {}).get("oof_coverage"),
                   "rules": [{"tokens": [x.t1] + ([x.t2] if x.t2 else []), "a": int(x.a), "b": int(x.b),
                              "P": int(x.P), "N": int(x.N), "precision": round(float(x.precision), 4),
                              "score": round(float(x.score), 4), "background": round(float(x.bg_rate), 5)}
                             for x in r.itertuples(index=False)]}
    return KeyDomains(setting(cfg), cats)


def setting(cfg: KeyDomainsConfig) -> dict:
    return cfg.model_dump(mode="json")
