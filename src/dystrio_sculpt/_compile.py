"""Physical weight slicing for SwiGLU FFN blocks."""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch

from ._model import get_mlp


def select_blocks_magnitude(
    importance: torch.Tensor, block_size: int, keep_frac: float,
) -> Tuple[List[int], torch.Tensor]:
    """Rank blocks by summed magnitude importance, keep top fraction."""
    ffn = int(importance.numel())
    n_blocks = int(math.ceil(ffn / block_size))
    keep_n = max(1, int(math.ceil(keep_frac * n_blocks)))
    scored = []
    for b in range(n_blocks):
        lo = b * block_size
        hi = min(ffn, (b + 1) * block_size)
        scored.append((float(importance[lo:hi].sum().item()), b))
    scored.sort(reverse=True, key=lambda x: x[0])
    kept_blocks = [b for _, b in scored[:keep_n]]
    idx: List[int] = []
    for b in kept_blocks:
        lo = b * block_size
        hi = min(ffn, (b + 1) * block_size)
        idx.extend(range(lo, hi))
    return kept_blocks, torch.tensor(idx, dtype=torch.long, device=importance.device)


@torch.no_grad()
def compress_mlp_layer_swiglu_inplace(
    model, layer_idx: int, kept_idx: torch.Tensor,
    dtype: torch.dtype, device: str,
) -> Dict[str, int]:
    """Replace gate/up/down projections with physically smaller Linear layers."""
    mlp = get_mlp(model, layer_idx)
    old_gate, old_up, old_down = mlp.gate_proj, mlp.up_proj, mlp.down_proj
    hidden = old_gate.in_features
    ffn_kept = int(kept_idx.numel())
    kept = kept_idx.to(device=device)

    new_gate = torch.nn.Linear(
        hidden, ffn_kept, bias=(old_gate.bias is not None), device=device, dtype=dtype,
    )
    new_up = torch.nn.Linear(
        hidden, ffn_kept, bias=(old_up.bias is not None), device=device, dtype=dtype,
    )
    new_down = torch.nn.Linear(
        ffn_kept, hidden, bias=(old_down.bias is not None), device=device, dtype=dtype,
    )

    new_gate.weight.copy_(old_gate.weight[kept].to(dtype))
    if old_gate.bias is not None:
        new_gate.bias.copy_(old_gate.bias[kept].to(dtype))
    new_up.weight.copy_(old_up.weight[kept].to(dtype))
    if old_up.bias is not None:
        new_up.bias.copy_(old_up.bias[kept].to(dtype))
    new_down.weight.copy_(old_down.weight[:, kept].to(dtype))
    if old_down.bias is not None:
        new_down.bias.copy_(old_down.bias.to(dtype))

    mlp.gate_proj = new_gate
    mlp.up_proj = new_up
    mlp.down_proj = new_down
    return {"hidden": hidden, "ffn_kept": ffn_kept}
