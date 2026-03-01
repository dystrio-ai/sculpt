"""Magnitude-based block selection (simpler fallback selector)."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from .._calibrate import collect_ffn_importance_swiglu
from .._compile import select_blocks_magnitude


def select_for_layer_magnitude(
    model, tokenizer, layer_idx: int, texts_cal, max_len: int,
    device: str, keep_frac: float,
) -> Tuple[List[int], torch.Tensor, Dict[str, object]]:
    """Calibrate + select blocks using magnitude importance."""
    imp = collect_ffn_importance_swiglu(
        model, tokenizer, layer_idx, texts_cal, max_len, device,
    )
    kept_blocks, kept_idx = select_blocks_magnitude(imp, 128, keep_frac)
    return kept_blocks, kept_idx, {"importance": imp.cpu()}
