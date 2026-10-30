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

"""Configuration models: the frozen recipe (packaged ``recipe.yaml``) and the run-level options."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class IpsApp(_Frozen):
    canonical: str
    ips6: str
    ips5: str


class IpsConfig(_Frozen):
    pipeline: str
    version: str
    interpro: str
    apps: tuple[IpsApp, ...]
    required_apps: tuple[str, ...]
    required_apps_min_proteins: int

    @property
    def app_names(self) -> list[str]:
        return [a.canonical for a in self.apps]

    @property
    def apps_hash(self) -> str:
        """Short hash of the sorted canonical app list (annotation cache key)."""
        return hashlib.sha256(",".join(sorted(self.app_names)).encode()).hexdigest()[:12]


class FeaturesConfig(_Frozen):
    gene_sep: bool
    dedup_contiguous: bool
    unk_policy: Literal["unk", "drop"]
    vocab_min_df: int
    vocab_top_k: int
    vocab_top_k_per_class: int


class DetectorConfig(_Frozen):
    embed_dim: int
    embed_dropout: float
    conv_filters: int
    conv_kernel: int
    rnn_units: int
    n_layers: int
    dropout: float
    window: int


class DecodeConfig(_Frozen):
    agg: Literal["mean", "max"] = "mean"
    fill: Literal["zero", "neighbors"] = "zero"
    threshold: float = 0.5
    hysteresis_gap: float = 0.15
    merge_gap: int = 3
    min_orfs: int = 1

    @property
    def threshold_low(self) -> float:
        return round(self.threshold - self.hysteresis_gap, 10)


class PostfilterConfig(_Frozen):
    n_estimators: int
    drop_threshold: float
    random_state: int
    classes: tuple[str, ...]


# ── training-only sections ──────────────────────────────────────────────────


class LossConfig(_Frozen):
    gamma: float
    alpha: float


class OptimizerConfig(_Frozen):
    lr: float
    weight_decay: float
    grad_clip: float


class SchedulerConfig(_Frozen):
    factor: float
    patience: int


class EarlyStoppingConfig(_Frozen):
    patience: int
    max_epochs: int
    orf_agg: Literal["mean", "max"]  # validation ORF scores for early stopping / LR plateau
    orf_fill: Literal["zero", "neighbors"]


class TrainSettings(_Frozen):
    loss: LossConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    early_stopping: EarlyStoppingConfig
    batch_size: int
    token_dropout: float
    seeds: tuple[int, ...]
    amp: bool


class ComposeConfig(_Frozen):
    duplicate_positives: int
    seed: int
    chunks_per_positive: int
    chunk_len_clip: tuple[int, int]
    background_min_cds: int


class DedupConfig(_Frozen):
    dice_threshold: float
    max_locus_bp: int


class SplitsConfig(_Frozen):
    n_splits: int
    val_fraction: float


class OperatingPointConfig(_Frozen):
    grid: tuple[float, float, float]  # start, stop (inclusive), step
    floor_metric: str
    floor: float
    min_pred_ratio: float


class KeyDomainsConfig(_Frozen):
    tau: float  # Wilson lower bound of the rule precision
    min_support: int
    min_coverage: float
    entry_types: tuple[str, ...]
    background_max: float


class NpcRfConfig(_Frozen):
    n_estimators: int


class NpcLrConfig(_Frozen):
    C: float


class NpcKnnConfig(_Frozen):
    k: int
    min_df: int


class NpcConfig(_Frozen):
    min_support: int
    precision_targets: tuple[float, ...]
    default_precision: float
    min_df: int
    rf: NpcRfConfig
    lr: NpcLrConfig
    knn: NpcKnnConfig
    key_domains: KeyDomainsConfig
    min_labelled: int


class Recipe(BaseModel):
    """The frozen recipe."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int
    ips: IpsConfig
    features: FeaturesConfig
    detector: DetectorConfig
    train: TrainSettings
    compose: ComposeConfig
    dedup: DedupConfig
    splits: SplitsConfig
    decode: DecodeConfig
    operating_point: OperatingPointConfig
    postfilter: PostfilterConfig
    npc: NpcConfig


@lru_cache(maxsize=1)
def load_recipe() -> Recipe:
    text = resources.files("sanntis").joinpath("recipe.yaml").read_text()
    return Recipe.model_validate(yaml.safe_load(text))


class NextflowOptions(BaseModel):
    """Infrastructure pass-through. sanntis forwards these and does not interpret them."""

    container_engine: Literal["docker", "singularity", "apptainer", "podman"] | None = None
    nf_config: list[Path] = Field(default_factory=list)
    nf_profile: list[str] = Field(default_factory=list)
    nf_args: str = ""
    nextflow: str = "nextflow"


class TrainConfig(BaseModel):
    run_dir: Path
    mibig: str | None = None  # MIBiG release; None = external data only
    external_gbk: list[Path] = Field(default_factory=list)
    external_json: list[Path] = Field(default_factory=list)  # none, or one per GBK archive (same order)
    negatives: str | None = None
    npc_cache: str | None = None  # published NPClassifier answers to seed the cache: directory or Hub version
    train_ips_tsv: list[Path] = Field(default_factory=list)  # precomputed IPS TSVs of the proteins: no IPS6 run
    ips_datadir: Path | None = None  # IPS6 data (default $SANNTIS_CACHE/ips6)
    entries_json: Path | None = None  # InterPro entries.json (default: from the IPS6 data)
    runner: Literal["nextflow", "inline"] = "nextflow"
    gpu: bool = False
    workers: int = 1
    nextflow: NextflowOptions = Field(default_factory=NextflowOptions)

    @field_validator("mibig", mode="before")
    @classmethod
    def _mibig(cls, v):
        if v is None or v is False or str(v).strip().lower() in ("", "none", "false"):
            return None
        return str(v).strip()

    @model_validator(mode="after")
    def _external(self):
        if self.external_json and len(self.external_json) != len(self.external_gbk):
            raise ValueError(f"{len(self.external_json)} external JSON archive(s) for {len(self.external_gbk)} GBK "
                             "archive(s): give none (GBK-only) or one per GBK archive, in the same order")
        return self

    @property
    def external(self) -> list[tuple[Path, Path | None]]:
        js = self.external_json or [None] * len(self.external_gbk)
        return list(zip(self.external_gbk, js, strict=True))
