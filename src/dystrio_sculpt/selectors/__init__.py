"""Block selector dispatch.

Selectors determine which FFN neuron blocks to keep during compression.
The structural selector (default) uses operator-fidelity scoring with
coupling-geometry diversity.  The magnitude selector is a simpler fallback.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .structural import (
    prescan_structural_artifacts,
    select_blocks_structural,
    CrossLayerNoveltyTracker,
)
from .magnitude import select_for_layer_magnitude
from .._calibrate import (
    collect_block_geometry_swiglu,
    collect_block_operator_sensitivity_swiglu,
)

BLOCK_SIZE = 128


def select_for_layer(
    model,
    tokenizer,
    layer_idx: int,
    texts_cal: Sequence[str],
    keep_frac: float,
    max_len: int,
    device: str,
    selector: str = "structural",
    prescan_cache: Optional[Dict[int, Dict[str, Any]]] = None,
    rng: np.random.RandomState | None = None,
    cross_layer_novelty: Optional[np.ndarray] = None,
    adapter=None,
) -> Tuple[List[int], torch.Tensor, Dict[str, object]]:
    """Select blocks for a single layer using the chosen selector.

    When *prescan_cache* contains an entry for *layer_idx* and the selector
    is structural, cached tensors are used instead of live calibration.

    *cross_layer_novelty*, when provided, is a per-block multiplier from a
    CrossLayerNoveltyTracker that boosts blocks not frequently selected in
    previously compressed layers.

    When *adapter* is provided, calibration is dispatched through it instead
    of calling the SwiGLU-specific functions directly.
    """
    if selector == "structural":
        if prescan_cache is not None and layer_idx in prescan_cache:
            pre = prescan_cache[layer_idx]
            geom_D = pre["D"]
            block_energy = pre["block_energy"]
            block_sensitivity = pre["block_sensitivity"]
            feature_multiplier = pre.get("feature_multiplier", 3)
        elif adapter is not None:
            geom = adapter.collect_block_geometry(
                model, tokenizer, layer_idx, texts_cal, max_len, device,
                block_size=BLOCK_SIZE, max_tokens=30_000,
            )
            sens = adapter.collect_block_sensitivity(
                model, tokenizer, layer_idx, texts_cal, max_len, device,
                block_size=BLOCK_SIZE, max_tokens=30_000,
            )
            geom_D = geom["D"]
            block_energy = geom.get("block_energy")
            block_sensitivity = sens["block_sensitivity"]
            feature_multiplier = geom.get("feature_multiplier", 3)
        else:
            geom = collect_block_geometry_swiglu(
                model, tokenizer, layer_idx, texts_cal, max_len, device,
                block_size=BLOCK_SIZE, max_tokens=30_000,
            )
            sens = collect_block_operator_sensitivity_swiglu(
                model, tokenizer, layer_idx, texts_cal, max_len, device,
                block_size=BLOCK_SIZE, max_tokens=30_000,
            )
            geom_D = geom["D"]
            block_energy = geom.get("block_energy")
            block_sensitivity = sens["block_sensitivity"]
            feature_multiplier = geom.get("feature_multiplier", 3)

        kept_blocks, kept_idx, arts = select_blocks_structural(
            geom_D, keep_frac, BLOCK_SIZE, topk_edges=20,
            block_energy=block_energy,
            feature_multiplier=feature_multiplier,
            block_sensitivity=block_sensitivity,
            rng=rng,
            cross_layer_novelty=cross_layer_novelty,
        )
        return kept_blocks, kept_idx.to(device), arts

    return select_for_layer_magnitude(
        model, tokenizer, layer_idx, texts_cal, max_len, device, keep_frac,
    )
