"""Baseline selectors for ablation studies.

These provide simple alternatives to the full Physarum structural selector,
isolating the contribution of each algorithmic component:

- ``sensitivity``: Ranks blocks by operator sensitivity (prescan data),
  takes top-k.  No Physarum conductance, no diversity penalty, no cross-layer
  novelty.  This is roughly what gradient-importance methods do.

- ``random``: Uniform random block selection at the target keep_frac.
  Sanity-check baseline.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


def select_blocks_sensitivity(
    block_sensitivity: torch.Tensor,
    keep_frac: float,
    block_size: int,
    block_energy: Optional[torch.Tensor] = None,
    feature_multiplier: int = 3,
) -> Tuple[List[int], torch.Tensor, Dict[str, Any]]:
    """Select blocks by operator sensitivity alone — no Physarum, no diversity.

    Blocks with the highest sensitivity scores are kept (they contribute
    most to the MLP output, so losing them hurts most).
    """
    n_feat = block_sensitivity.shape[0]
    F = feature_multiplier
    n_blocks = n_feat // F
    if n_blocks == 0:
        n_blocks = max(1, n_feat)
        F = 1
    keep_n = max(1, int(math.ceil(keep_frac * n_blocks)))

    scores = np.asarray(block_sensitivity, dtype=np.float64)[:n_blocks]

    if block_energy is not None:
        be = np.asarray(block_energy, dtype=np.float64)[:n_blocks]
        be_norm = be / (be.max() + 1e-30)
        scores = scores * (0.5 + 0.5 * be_norm)

    ranked = np.argsort(scores)[::-1]
    selected = sorted(ranked[:keep_n].tolist())

    ffn = n_blocks * block_size
    idx: List[int] = []
    for b in selected:
        lo = b * block_size
        hi = min(ffn, (b + 1) * block_size)
        idx.extend(range(lo, hi))

    artifacts = {"block_scores": torch.from_numpy(scores), "method": "sensitivity"}
    return selected, torch.tensor(idx, dtype=torch.long), artifacts


def select_blocks_random(
    n_blocks_total: int,
    keep_frac: float,
    block_size: int,
    rng: Optional[np.random.RandomState] = None,
) -> Tuple[List[int], torch.Tensor, Dict[str, Any]]:
    """Uniform random block selection — sanity-check baseline."""
    if rng is None:
        rng = np.random.RandomState()

    keep_n = max(1, int(math.ceil(keep_frac * n_blocks_total)))
    all_blocks = np.arange(n_blocks_total)
    rng.shuffle(all_blocks)
    selected = sorted(all_blocks[:keep_n].tolist())

    ffn = n_blocks_total * block_size
    idx: List[int] = []
    for b in selected:
        lo = b * block_size
        hi = min(ffn, (b + 1) * block_size)
        idx.extend(range(lo, hi))

    artifacts = {"method": "random"}
    return selected, torch.tensor(idx, dtype=torch.long), artifacts
