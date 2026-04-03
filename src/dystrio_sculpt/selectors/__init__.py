"""Block / expert selector dispatch.

Selectors determine which FFN neuron blocks (or which MoE experts) to keep
during compression.  The structural selector (default) uses operator-fidelity
scoring with coupling-geometry diversity.  The magnitude selector is a simpler
fallback.

For MoE models, each "block" is one expert.  The selector returns expert
indices directly instead of expanding to neuron ranges.
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
from .baselines import select_blocks_sensitivity, select_blocks_random
from .._calibrate import (
    collect_block_geometry_swiglu,
    collect_block_operator_sensitivity_swiglu,
)

BLOCK_SIZE = 128


def _is_moe_adapter(adapter) -> bool:
    """Check if the adapter operates at expert-block level."""
    if adapter is None:
        return False
    from ..architectures.descriptor import OptimizationTarget
    targets = adapter.supported_targets()
    return OptimizationTarget.EXPERT_BLOCK in targets


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
    """Select blocks (or experts) for a single layer.

    For dense models: returns neuron block indices expanded to individual
    neuron positions (used by compress_mlp_layer_swiglu_inplace).

    For MoE models: returns expert indices directly (used by
    SwiGLUMoEAdapter.compress_layer).  Each "block" is one expert;
    kept_idx contains expert IDs, not neuron ranges.
    """
    moe_mode = _is_moe_adapter(adapter)

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

        block_size = 1 if moe_mode else BLOCK_SIZE
        kept_blocks, kept_idx, arts = select_blocks_structural(
            geom_D, keep_frac, block_size, topk_edges=20,
            block_energy=block_energy,
            feature_multiplier=feature_multiplier,
            block_sensitivity=block_sensitivity,
            rng=rng,
            cross_layer_novelty=cross_layer_novelty,
        )

        if moe_mode:
            return kept_blocks, torch.tensor(kept_blocks, dtype=torch.long, device=device), arts
        return kept_blocks, kept_idx.to(device), arts

    if selector == "sensitivity":
        if prescan_cache is not None and layer_idx in prescan_cache:
            pre = prescan_cache[layer_idx]
            block_sensitivity = pre["block_sensitivity"]
            block_energy = pre.get("block_energy")
            feature_multiplier = pre.get("feature_multiplier", 3)
        elif adapter is not None:
            sens = adapter.collect_block_sensitivity(
                model, tokenizer, layer_idx, texts_cal, max_len, device,
                block_size=BLOCK_SIZE, max_tokens=30_000,
            )
            geom = adapter.collect_block_geometry(
                model, tokenizer, layer_idx, texts_cal, max_len, device,
                block_size=BLOCK_SIZE, max_tokens=30_000,
            )
            block_sensitivity = sens["block_sensitivity"]
            block_energy = geom.get("block_energy")
            feature_multiplier = geom.get("feature_multiplier", 3)
        else:
            sens = collect_block_operator_sensitivity_swiglu(
                model, tokenizer, layer_idx, texts_cal, max_len, device,
                block_size=BLOCK_SIZE, max_tokens=30_000,
            )
            geom = collect_block_geometry_swiglu(
                model, tokenizer, layer_idx, texts_cal, max_len, device,
                block_size=BLOCK_SIZE, max_tokens=30_000,
            )
            block_sensitivity = sens["block_sensitivity"]
            block_energy = geom.get("block_energy")
            feature_multiplier = geom.get("feature_multiplier", 3)

        block_size = 1 if moe_mode else BLOCK_SIZE
        kept_blocks, kept_idx, arts = select_blocks_sensitivity(
            block_sensitivity, keep_frac, block_size,
            block_energy=block_energy,
            feature_multiplier=feature_multiplier,
        )
        if moe_mode:
            return kept_blocks, torch.tensor(kept_blocks, dtype=torch.long, device=device), arts
        return kept_blocks, kept_idx.to(device), arts

    if selector == "random":
        if adapter is not None:
            n_blocks = adapter.get_ffn_size(model, layer_idx) // BLOCK_SIZE
        else:
            from .._model import get_mlp
            n_blocks = get_mlp(model, layer_idx).gate_proj.out_features // BLOCK_SIZE
        block_size = 1 if moe_mode else BLOCK_SIZE
        if moe_mode:
            n_blocks = adapter.get_ffn_size(model, layer_idx)
        kept_blocks, kept_idx, arts = select_blocks_random(
            n_blocks, keep_frac, block_size, rng=rng,
        )
        if moe_mode:
            return kept_blocks, torch.tensor(kept_blocks, dtype=torch.long, device=device), arts
        return kept_blocks, kept_idx.to(device), arts

    return select_for_layer_magnitude(
        model, tokenizer, layer_idx, texts_cal, max_len, device, keep_frac,
    )
