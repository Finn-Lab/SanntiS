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

"""NPC pathway key-domain support: does a predicted region carry the domains characteristic of its called pathways?

Per pathway, a short list of rules: an InterPro family / domain, or a pair of them present anywhere in the same
region (fitted by :mod:`sanntis.train.key_domains`). For each called pathway of a region:

- *supported*: at least one of the pathway's rules is present;
- *unsupported*: none is;
- ``na``: the pathway has no rules, or is gated (its out-of-fold rule coverage in CV was below ``min_coverage``,
  so a missing rule says little).

Region outputs:

- ``npc_pathway_key_domains_support``: ``<pathway>:<rule>`` for every rule present (pairs as ``t1+t2``),
  ``<pathway>:none`` for an unsupported and ``<pathway>:na`` for an ``na`` called pathway, joined by ``|``;
- ``npc_pathway_key_domains_unsupported``: True when some called pathway is unsupported, False when every
  judged called pathway is supported, None (``na``) when every called pathway is ``na``.

Bundle file ``npc_key_domains.json`` (optional: bundles without it give None in both columns).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

FILE = "npc_key_domains.json"
ACTIVE, GATED, NO_RULES = "active", "gated", "no_rules"


@dataclass
class KeyDomains:
    setting: dict  # the recipe npc.key_domains block the rules were fitted with
    categories: dict[str, dict]  # pathway → {status, oof_coverage, rules: [{tokens, a, b, P, N, precision, score}]}

    def __post_init__(self):
        self._rules = {c: [tuple(r["tokens"]) for r in d["rules"]] for c, d in self.categories.items()
                       if d["status"] == ACTIVE}

    def support(self, cds: Sequence[Sequence[str]], called: Sequence[str]) -> tuple[str, bool | None]:
        """One region (per-CDS tokens) and its called pathways → (support string, unsupported or None)."""
        bag = {t for g in cds for t in g}
        parts, flags = [], []
        for p in called:
            if p not in self._rules:
                parts.append(f"{p}:na")
                continue
            hit = ["+".join(r) for r in self._rules[p] if all(t in bag for t in r)]
            parts += [f"{p}:{h}" for h in hit] or [f"{p}:none"]
            flags.append(not hit)
        return "|".join(parts), (any(flags) if flags else None)

    def to_json(self) -> dict:
        return {"setting": self.setting, "categories": self.categories}

    @classmethod
    def from_json(cls, d: dict) -> KeyDomains:
        return cls(dict(d["setting"]), dict(d["categories"]))

    def save(self, bundle: Path) -> str:
        (Path(bundle) / FILE).write_text(json.dumps(self.to_json(), indent=2))
        return FILE

    @classmethod
    def load(cls, bundle: Path) -> KeyDomains | None:
        f = Path(bundle) / FILE
        return cls.from_json(json.loads(f.read_text())) if f.is_file() else None
