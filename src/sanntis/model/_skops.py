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

"""skops (de)serialisation with an explicit trusted-types allowlist."""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

import skops.io as sio

# Types skops does not trust by default but that the bundle legitimately contains: RandomForest trees and the
# constant estimator one-vs-rest uses for a label that is constant in the training data.
TRUSTED_TYPES = ["sklearn.tree._tree.Tree", "sklearn.multiclass._ConstantPredictor"]


class UntrustedTypesError(RuntimeError):
    pass


def dump(obj: Any, path: Path) -> None:
    sio.dump(obj, path, compression=zipfile.ZIP_DEFLATED, compresslevel=6)


def load(path: Path) -> Any:
    untrusted = sio.get_untrusted_types(file=path)
    bad = sorted(set(untrusted) - set(TRUSTED_TYPES))
    if bad:
        raise UntrustedTypesError(f"{path}: refusing to load untrusted types {bad}")
    return sio.load(path, trusted=untrusted)
