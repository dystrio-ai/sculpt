"""Calibration: collect per-neuron SwiGLU activation importance via forward hooks.

Functions:
    collect_ffn_importance_swiglu       – per-neuron magnitude importance (swiglu_mag selector)
    collect_block_geometry_swiglu       – block-level 3-feature covariance (structural selector)
    collect_block_operator_sensitivity  – operator-fidelity block sensitivity (structural v3)
"""

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

    Per block b and per token, extracts three features:
        mu_b   = mean(a_block)
        sigma_b = std(a_block)
        mag_b  = mean(|a_block|)

    The token feature vector z has dimension 3*n_blocks:
        z = [mu_0, sigma_0, mag_0, mu_1, sigma_1, mag_1, ...]

    Accumulates Cov(z) in float64 across tokens and returns the covariance
    matrix D (3B x 3B), per-block mean energy, and metadata.
    """
    layer = model.model.layers[layer_idx]
    mlp = layer.mlp
    ffn = mlp.gate_proj.out_features
    n_blocks = math.ceil(ffn / block_size)
    F = 3  # features per block: mu, sigma, mag
    dim = F * n_blocks

    sum_z = torch.zeros(dim, dtype=torch.float64, device=device)
    sum_zz = torch.zeros(dim, dim, dtype=torch.float64, device=device)
    sum_mag = torch.zeros(n_blocks, dtype=torch.float64, device=device)
    total_tokens = 0

    def hook(module, inputs, output):
        nonlocal sum_z, sum_zz, sum_mag, total_tokens
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

        a64 = a.to(torch.float64)
        z = torch.zeros(T, dim, dtype=torch.float64, device=device)
        for b in range(n_blocks):
            lo = b * block_size
            hi = min(ffn, (b + 1) * block_size)
            blk = a64[:, lo:hi]                      # [T, block_len]
            mu = blk.mean(dim=1)                      # [T]
            sigma = blk.std(dim=1, correction=0)      # [T]
            mag = blk.abs().mean(dim=1)               # [T]
            base = F * b
            z[:, base] = mu
            z[:, base + 1] = sigma
            z[:, base + 2] = mag
            sum_mag += mag.sum(dim=0).unsqueeze(0) if mag.dim() == 0 else mag.sum()

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
    block_energy = (sum_mag / N).cpu()

    return {
        "D": D.cpu(),
        "block_energy": block_energy,
        "n_tokens": total_tokens,
        "block_size": block_size,
        "n_blocks": n_blocks,
        "feature_multiplier": F,
    }


@torch.no_grad()
def collect_block_operator_sensitivity_swiglu(
    model,
    tokenizer,
    layer_idx: int,
    texts: Sequence[str],
    max_len: int,
    device: str,
    block_size: int,
    max_tokens: int = 30_000,
) -> Dict[str, object]:
    """Measure how much zeroing each block changes the MLP output (operator fidelity).

    For each token, computes:
        a = act(gate_proj(x)) * up_proj(x)       [T, ffn]
        y = down_proj(a)                          [T, hidden]

    Then for each block b (neurons lo:hi of a):
        delta_y_b = W_down[:, lo:hi] @ a[:, lo:hi].T   [hidden, T]
        score_b  += mean_T ||delta_y_b||^2

    This avoids per-block forward passes by decomposing the output difference
    via the linearity of down_proj.  Also accumulates block_energy as before.
    """
    layer = model.model.layers[layer_idx]
    mlp = layer.mlp
    ffn = mlp.gate_proj.out_features
    n_blocks = math.ceil(ffn / block_size)

    sensitivity = torch.zeros(n_blocks, dtype=torch.float64, device=device)
    block_energy = torch.zeros(n_blocks, dtype=torch.float64, device=device)
    total_tokens = 0

    W_down = mlp.down_proj.weight.float()  # [hidden, ffn]

    def hook(module, inputs, output):
        nonlocal sensitivity, block_energy, total_tokens
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

        for b in range(n_blocks):
            lo = b * block_size
            hi = min(ffn, (b + 1) * block_size)
            a_blk = a[:, lo:hi]                          # [T, blk]
            delta = a_blk @ W_down[:, lo:hi].T            # [T, hidden]
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
        _ = model(**inp, use_cache=False)
    h.remove()

    N = max(total_tokens, 1)
    return {
        "block_sensitivity": (sensitivity / N).cpu(),
        "block_energy": (block_energy / N).cpu(),
        "n_blocks": n_blocks,
    }
