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

"""Focal loss; copied from mgnify-bgcs-common-core so that the package does not depend on it."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def sigmoid_focal_loss(logits: torch.Tensor, targets: torch.Tensor, *, gamma: float = 2.0,
                       alpha: float | None = None, mask: torch.Tensor | None = None) -> torch.Tensor:
    """FL = alpha_t (1 - p_t)^gamma BCE, mean over unmasked positions."""
    if logits.shape != targets.shape:
        raise ValueError(f"logits shape {tuple(logits.shape)} != targets shape {tuple(targets.shape)}")
    ce = F.binary_cross_entropy_with_logits(logits, targets.to(dtype=logits.dtype), reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    loss = ((1.0 - p_t) ** gamma) * ce
    if alpha is not None:
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        loss = alpha_t * loss
    if mask is None:
        return loss.mean()
    mask_f = mask.to(dtype=loss.dtype)
    return (loss * mask_f).sum() / mask_f.sum().clamp_min(1.0)
