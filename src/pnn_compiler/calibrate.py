"""Calibration: collect per-neuron SwiGLU activation importance via forward hooks."""

from __future__ import annotations

import math
from typing import Dict, Sequence

import torch


@torch.no_grad()
def collect_ffn_importance_swiglu(
    model, tokenizer, layer_idx: int, texts: Sequence[str], max_len: int, device: str,
) -> torch.Tensor:
    """
    imp[j] = mean_tokens | act(gate_proj(x))_j * up_proj(x)_j |
    Accumulate in fp32.
    """
    layer = model.model.layers[layer_idx]
    mlp = layer.mlp
    ffn = mlp.gate_proj.out_features

    imp = torch.zeros(ffn, device=device, dtype=torch.float32)
    count = 0

    def hook(module, inputs, output):
        nonlocal imp, count
        x = inputs[0]  # [b,s,h]
        x2 = x.reshape(-1, x.shape[-1])
        gate = mlp.gate_proj(x2).float()
        up = mlp.up_proj(x2).float()
        a = mlp.act_fn(gate) * up
        imp += a.abs().mean(dim=0)
        count += 1

    h = mlp.register_forward_hook(hook)
    model.eval()
    for t in texts:
        inp = tokenizer(t, return_tensors="pt", truncation=True, max_length=max_len)
        inp = {k: v.to(device) for k, v in inp.items()}
        _ = model(**inp, use_cache=False)
    h.remove()

    return imp / max(count, 1)


@torch.no_grad()
def collect_block_geometry_swiglu(
    model,
    tokenizer,
    layer_idx: int,
    texts: Sequence[str],
    max_len: int,
    device: str,
    block_size: int,
    max_tokens: int = 30_000,
) -> Dict[str, object]:
    """Compute block-level covariance of SwiGLU activations for structural selection.

    For each token, computes z[b] = mean(|a[lo:hi]|) where a = act(gate(x)) * up(x),
    then accumulates Cov(z) in float64.  Returns the covariance matrix D plus metadata.
    """
    layer = model.model.layers[layer_idx]
    mlp = layer.mlp
    ffn = mlp.gate_proj.out_features
    n_blocks = math.ceil(ffn / block_size)

    sum_z = torch.zeros(n_blocks, dtype=torch.float64, device=device)
    sum_zz = torch.zeros(n_blocks, n_blocks, dtype=torch.float64, device=device)
    total_tokens = 0

    def hook(module, inputs, output):
        nonlocal sum_z, sum_zz, total_tokens
        if total_tokens >= max_tokens:
            return
        x = inputs[0]
        x2 = x.reshape(-1, x.shape[-1])
        gate = mlp.gate_proj(x2).float()
        up = mlp.up_proj(x2).float()
        a = mlp.act_fn(gate) * up  # [T, ffn]

        T = a.shape[0]
        budget = max_tokens - total_tokens
        if T > budget:
            a = a[:budget]
            T = budget

        z = torch.zeros(T, n_blocks, dtype=torch.float64, device=device)
        for b in range(n_blocks):
            lo = b * block_size
            hi = min(ffn, (b + 1) * block_size)
            z[:, b] = a[:, lo:hi].abs().to(torch.float64).mean(dim=1)

        sum_z += z.sum(dim=0)
        sum_zz += z.T @ z
        total_tokens += T

    h = mlp.register_forward_hook(hook)
    model.eval()
    for t in texts:
        if total_tokens >= max_tokens:
            break
        inp = tokenizer(t, return_tensors="pt", truncation=True, max_length=max_len)
        inp = {k: v.to(device) for k, v in inp.items()}
        _ = model(**inp, use_cache=False)
    h.remove()

    N = max(total_tokens, 1)
    mean_z = sum_z / N
    D = (sum_zz / N) - mean_z.unsqueeze(1) * mean_z.unsqueeze(0)

    return {
        "D": D.cpu(),
        "n_tokens": total_tokens,
        "block_size": block_size,
        "n_blocks": n_blocks,
    }
