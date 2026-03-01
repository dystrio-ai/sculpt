"""Calibration: per-neuron importance, block geometry, and operator sensitivity."""

from __future__ import annotations

import math
from typing import Dict, Sequence

import torch


@torch.no_grad()
def collect_ffn_importance_swiglu(
    model, tokenizer, layer_idx: int, texts: Sequence[str],
    max_len: int, device: str,
) -> torch.Tensor:
    """Per-neuron magnitude importance: imp[j] = mean |act(gate)*up|_j."""
    layer = model.model.layers[layer_idx]
    mlp = layer.mlp
    ffn = mlp.gate_proj.out_features
    imp = torch.zeros(ffn, device=device, dtype=torch.float32)
    count = 0

    def hook(module, inputs, output):
        nonlocal imp, count
        x = inputs[0].reshape(-1, inputs[0].shape[-1])
        gate = mlp.gate_proj(x).float()
        up = mlp.up_proj(x).float()
        imp += (mlp.act_fn(gate) * up).abs().mean(dim=0)
        count += 1

    h = mlp.register_forward_hook(hook)
    model.eval()
    for t in texts:
        inp = tokenizer(t, return_tensors="pt", truncation=True, max_length=max_len)
        inp = {k: v.to(device) for k, v in inp.items()}
        model(**inp, use_cache=False)
    h.remove()
    return imp / max(count, 1)


@torch.no_grad()
def collect_block_geometry_swiglu(
    model, tokenizer, layer_idx: int, texts: Sequence[str],
    max_len: int, device: str, block_size: int, max_tokens: int = 30_000,
) -> Dict[str, object]:
    """Block-level covariance of SwiGLU activations (3 features per block)."""
    layer = model.model.layers[layer_idx]
    mlp = layer.mlp
    ffn = mlp.gate_proj.out_features
    n_blocks = math.ceil(ffn / block_size)
    F = 3
    dim = F * n_blocks

    sum_z = torch.zeros(dim, dtype=torch.float64, device=device)
    sum_zz = torch.zeros(dim, dim, dtype=torch.float64, device=device)
    sum_mag = torch.zeros(n_blocks, dtype=torch.float64, device=device)
    total_tokens = 0

    def hook(module, inputs, output):
        nonlocal sum_z, sum_zz, sum_mag, total_tokens
        if total_tokens >= max_tokens:
            return
        x = inputs[0].reshape(-1, inputs[0].shape[-1])
        gate = mlp.gate_proj(x).float()
        up = mlp.up_proj(x).float()
        a = mlp.act_fn(gate) * up
        T = a.shape[0]
        budget = max_tokens - total_tokens
        if T > budget:
            a = a[:budget]
            T = budget
        a64 = a.to(torch.float64)
        z = torch.zeros(T, dim, dtype=torch.float64, device=device)
        for b in range(n_blocks):
            lo = b * block_size
            hi = min(ffn, (b + 1) * block_size)
            blk = a64[:, lo:hi]
            base = F * b
            z[:, base] = blk.mean(dim=1)
            z[:, base + 1] = blk.std(dim=1, correction=0)
            z[:, base + 2] = blk.abs().mean(dim=1)
            sum_mag[b] += blk.abs().mean(dim=1).sum()
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
        model(**inp, use_cache=False)
    h.remove()

    N = max(total_tokens, 1)
    mean_z = sum_z / N
    D = (sum_zz / N) - mean_z.unsqueeze(1) * mean_z.unsqueeze(0)
    block_energy = (sum_mag / N).cpu()
    return {
        "D": D.cpu(),
        "block_energy": block_energy,
        "n_blocks": n_blocks,
        "feature_multiplier": F,
    }


@torch.no_grad()
def collect_block_operator_sensitivity_swiglu(
    model, tokenizer, layer_idx: int, texts: Sequence[str],
    max_len: int, device: str, block_size: int, max_tokens: int = 30_000,
) -> Dict[str, object]:
    """Operator fidelity: how much zeroing each block changes MLP output."""
    layer = model.model.layers[layer_idx]
    mlp = layer.mlp
    ffn = mlp.gate_proj.out_features
    n_blocks = math.ceil(ffn / block_size)
    sensitivity = torch.zeros(n_blocks, dtype=torch.float64, device=device)
    block_energy = torch.zeros(n_blocks, dtype=torch.float64, device=device)
    total_tokens = 0
    W_down = mlp.down_proj.weight.float()

    def hook(module, inputs, output):
        nonlocal sensitivity, block_energy, total_tokens
        if total_tokens >= max_tokens:
            return
        x = inputs[0].reshape(-1, inputs[0].shape[-1])
        gate = mlp.gate_proj(x).float()
        up = mlp.up_proj(x).float()
        a = mlp.act_fn(gate) * up
        T = a.shape[0]
        budget = max_tokens - total_tokens
        if T > budget:
            a = a[:budget]
            T = budget
        for b in range(n_blocks):
            lo = b * block_size
            hi = min(ffn, (b + 1) * block_size)
            a_blk = a[:, lo:hi]
            delta = a_blk @ W_down[:, lo:hi].T
            sensitivity[b] += delta.pow(2).sum(dim=1).to(torch.float64).sum()
            block_energy[b] += a_blk.abs().to(torch.float64).mean(dim=1).sum()
        total_tokens += T

    h = mlp.register_forward_hook(hook)
    model.eval()
    for t in texts:
        if total_tokens >= max_tokens:
            break
        inp = tokenizer(t, return_tensors="pt", truncation=True, max_length=max_len)
        inp = {k: v.to(device) for k, v in inp.items()}
        model(**inp, use_cache=False)
    h.remove()

    N = max(total_tokens, 1)
    return {
        "block_sensitivity": (sensitivity / N).cpu(),
        "block_energy": (block_energy / N).cpu(),
        "n_blocks": n_blocks,
    }
