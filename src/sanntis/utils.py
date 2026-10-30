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

"""Small shared helpers: checksums, atomic writes, pinned downloads, cache location."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import urllib.parse
import urllib.request
from pathlib import Path

log = logging.getLogger(__name__)


class ChecksumError(RuntimeError):
    pass


def file_digest(path: Path, algo: str = "sha256") -> str:
    with Path(path).open("rb") as fh:
        return hashlib.file_digest(fh, algo).hexdigest()


def sha256_file(path: Path) -> str:
    return file_digest(path, "sha256")


def write_json(path: Path, obj, **kw) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str, **kw) + "\n")
    tmp.replace(path)
    return path


def cache_dir(sub: str = "") -> Path:
    """``$SANNTIS_CACHE`` (default ``~/.cache/sanntis``)."""
    root = Path(os.environ.get("SANNTIS_CACHE") or Path.home() / ".cache" / "sanntis").expanduser()
    d = root / sub if sub else root
    d.mkdir(parents=True, exist_ok=True)
    return d


def http_request(url: str) -> urllib.request.Request:
    """A GET request for ``url``; only ``http(s)`` URLs (never ``file://`` or ``ftp://`` from a user-supplied table)."""
    if urllib.parse.urlsplit(url).scheme not in ("http", "https"):
        raise ValueError(f"{url}: only http(s) URLs are fetched")
    return urllib.request.Request(url, headers={"User-Agent": "sanntis"})


def fetch(url: str, dest: Path, sha256: str | None = None, *, timeout: int = 600) -> Path:
    """Download ``url`` to ``dest`` unless a file with the expected sha256 is already there; verify the sha256."""
    dest = Path(dest)
    if dest.exists() and (sha256 is None or sha256_file(dest) == sha256):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    log.info("downloading %s", url)
    with urllib.request.urlopen(http_request(url), timeout=timeout) as r, tmp.open("wb") as fh:
        shutil.copyfileobj(r, fh, 1 << 20)
    got = sha256_file(tmp)
    if sha256 is not None and got != sha256:
        tmp.unlink()
        raise ChecksumError(f"{url}: sha256 {got} != pinned {sha256}")
    tmp.replace(dest)
    return dest


def available_cpus() -> int:
    """CPUs this process may use (scheduler affinity / cgroup), not the node's core count."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:  # macOS
        return max(1, os.cpu_count() or 1)


def configure_threads(n: int | None = None) -> int:
    """Size the torch / BLAS thread pools to ``n`` (default: ``$SANNTIS_THREADS`` or the available CPUs).

    torch otherwise sizes its pool from the node's physical cores, which oversubscribes a SLURM allocation or a
    process-pool worker by an order of magnitude."""
    n = n or int(os.environ.get("SANNTIS_THREADS", 0) or 0) or available_cpus()
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, str(n))
    try:
        import torch

        torch.set_num_threads(n)
    except ImportError:
        pass
    return n
