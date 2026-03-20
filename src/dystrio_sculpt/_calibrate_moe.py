"""Expert-level calibration for MoE models.

Mirrors _calibrate.py but operates at expert granularity:
  - Expert operator sensitivity: how much does zeroing expert E change the MoE output?
  - Expert covariance: how correlated are experts' outputs on calibration data?
  - Expert utilization: how frequently does the router select each expert?

These feed into the same Physarum conductance → structural selection pipeline
used for dense neuron-block pruning, but with experts as the selection unit.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F


def _get_layers_module(model):
    """Locate the decoder layer list, handling multimodal wrappers.

    Tries common paths: model.model.layers, model.model.text_model.layers,
    model.language_model.model.layers, model.text_model.model.layers.
    """
    candidates = []
    if hasattr(model, "model"):
        m = model.model
        if hasattr(m, "layers"):
            candidates.append(m.layers)
        if hasattr(m, "text_model") and hasattr(m.text_model, "layers"):
            candidates.append(m.text_model.layers)
        if hasattr(m, "language_model"):
            lm = m.language_model
            if hasattr(lm, "layers"):
                candidates.append(lm.layers)
            if hasattr(lm, "model") and hasattr(lm.model, "layers"):
                candidates.append(lm.model.layers)
    if hasattr(model, "text_model"):
        tm = model.text_model
        if hasattr(tm, "model") and hasattr(tm.model, "layers"):
            candidates.append(tm.model.layers)
        if hasattr(tm, "layers"):
            candidates.append(tm.layers)
    if hasattr(model, "language_model"):
        lm = model.language_model
        if hasattr(lm, "model") and hasattr(lm.model, "layers"):
            candidates.append(lm.model.layers)

    for layers in candidates:
        if hasattr(layers, "__len__") and len(layers) > 0:
            return layers

    raise ValueError(
        "Cannot locate decoder layers. Tried model.model.layers, "
        "model.model.text_model.layers, etc."
    )


def _get_moe_module(model, layer_idx: int):
    """Locate the MoE module in a transformer layer.

    Supports Mixtral (block_sparse_moe), Qwen-MoE (mlp.experts),
    and DeepSeek (mlp.experts) naming conventions.
    Handles multimodal model wrappers via _get_layers_module.
    """
    layers = _get_layers_module(model)
    layer = layers[layer_idx]
    if hasattr(layer, "block_sparse_moe"):
        return layer.block_sparse_moe
    if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
        return layer.mlp
    raise ValueError(f"Cannot locate MoE module in layer {layer_idx}")


def _get_experts_and_gate(moe_module):
    """Extract the expert list and gating network from an MoE module."""
    experts = None
    gate = None
    if hasattr(moe_module, "experts"):
        experts = moe_module.experts
    if hasattr(moe_module, "gate"):
        gate = moe_module.gate
    elif hasattr(moe_module, "router"):
        gate = moe_module.router
    if experts is None:
        raise ValueError("Cannot find experts in MoE module")
    if gate is None:
        raise ValueError("Cannot find gate/router in MoE module")
    return experts, gate


@torch.no_grad()
def collect_expert_utilization(
    model, tokenizer, layer_idx: int, texts: Sequence[str],
    max_len: int, device: str, max_tokens: int = 30_000,
) -> Dict[str, Any]:
    """Measure per-expert utilization: selection frequency and average routing weight.

    Returns:
        expert_frequency: [num_experts] — fraction of tokens that select each expert
        expert_avg_weight: [num_experts] — average gating weight when selected
        total_tokens: int — number of tokens observed
    """
    moe = _get_moe_module(model, layer_idx)
    experts, gate = _get_experts_and_gate(moe)
    n_experts = len(experts)
    top_k = getattr(moe, "num_experts_per_tok", None) or getattr(moe, "top_k", 2)

    expert_count = torch.zeros(n_experts, device=device, dtype=torch.float64)
    expert_weight_sum = torch.zeros(n_experts, device=device, dtype=torch.float64)
    total_tokens = 0

    def hook(module, inputs, output):
        nonlocal expert_count, expert_weight_sum, total_tokens
        if total_tokens >= max_tokens:
            return
        hidden = inputs[0]
        if hidden.dim() == 3:
            hidden = hidden.reshape(-1, hidden.shape[-1])
        T = hidden.shape[0]
        budget = max_tokens - total_tokens
        if T > budget:
            hidden = hidden[:budget]
            T = budget

        logits = gate(hidden.float())
        weights = F.softmax(logits, dim=-1)
        topk_weights, topk_indices = torch.topk(weights, top_k, dim=-1)

        for k in range(top_k):
            idx = topk_indices[:, k]
            w = topk_weights[:, k].to(torch.float64)
            expert_count.scatter_add_(0, idx.long(), torch.ones_like(w))
            expert_weight_sum.scatter_add_(0, idx.long(), w)

        total_tokens += T

    h = moe.register_forward_hook(hook)
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
        "expert_frequency": (expert_count / N).cpu(),
        "expert_avg_weight": (expert_weight_sum / expert_count.clamp(min=1)).cpu(),
        "total_tokens": total_tokens,
        "n_experts": n_experts,
        "top_k": top_k,
    }


@torch.no_grad()
def collect_expert_sensitivity(
    model, tokenizer, layer_idx: int, texts: Sequence[str],
    max_len: int, device: str, max_tokens: int = 30_000,
) -> Dict[str, Any]:
    """Expert operator sensitivity: how much does zeroing each expert change the MoE output?

    For each token, we compute the full MoE output, then simulate dropping each
    selected expert and measure the L2 delta. This is the expert-level analog
    of collect_block_operator_sensitivity_swiglu.
    """
    moe = _get_moe_module(model, layer_idx)
    experts, gate = _get_experts_and_gate(moe)
    n_experts = len(experts)
    top_k = getattr(moe, "num_experts_per_tok", None) or getattr(moe, "top_k", 2)

    sensitivity = torch.zeros(n_experts, device=device, dtype=torch.float64)
    energy = torch.zeros(n_experts, device=device, dtype=torch.float64)
    expert_count = torch.zeros(n_experts, device=device, dtype=torch.float64)
    total_tokens = 0

    def hook(module, inputs, output):
        nonlocal sensitivity, energy, expert_count, total_tokens
        if total_tokens >= max_tokens:
            return
        hidden = inputs[0]
        if hidden.dim() == 3:
            hidden = hidden.reshape(-1, hidden.shape[-1])
        T = hidden.shape[0]
        budget = max_tokens - total_tokens
        if T > budget:
            hidden = hidden[:budget]
            T = budget

        logits = gate(hidden.float())
        weights = F.softmax(logits, dim=-1)
        topk_weights, topk_indices = torch.topk(weights, top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        for t_idx in range(min(T, 512)):
            tok_hidden = hidden[t_idx : t_idx + 1]
            tok_experts = topk_indices[t_idx]
            tok_weights = topk_weights[t_idx]

            expert_outputs = []
            for k in range(top_k):
                eidx = tok_experts[k].item()
                exp_out = experts[eidx](tok_hidden).float()
                expert_outputs.append((eidx, tok_weights[k].float(), exp_out))

            full_out = sum(w * o for _, w, o in expert_outputs)

            for eidx, w, exp_out in expert_outputs:
                delta = w * exp_out
                sensitivity[eidx] += delta.pow(2).sum().to(torch.float64)
                energy[eidx] += exp_out.abs().mean().to(torch.float64)
                expert_count[eidx] += 1

        total_tokens += T

    h = moe.register_forward_hook(hook)
    model.eval()
    for t in texts:
        if total_tokens >= max_tokens:
            break
        inp = tokenizer(t, return_tensors="pt", truncation=True, max_length=max_len)
        inp = {k: v.to(device) for k, v in inp.items()}
        model(**inp, use_cache=False)
    h.remove()

    count_safe = expert_count.clamp(min=1)
    return {
        "expert_sensitivity": (sensitivity / count_safe).cpu(),
        "expert_energy": (energy / count_safe).cpu(),
        "expert_count": expert_count.cpu(),
        "n_experts": n_experts,
    }


@torch.no_grad()
def collect_expert_covariance(
    model, tokenizer, layer_idx: int, texts: Sequence[str],
    max_len: int, device: str, max_tokens: int = 30_000,
    n_features: int = 3,
) -> Dict[str, Any]:
    """Expert-level covariance matrix for Physarum structural selection.

    For each token, we extract per-expert features (mean, std, abs_mean of the
    expert's output) and build a covariance matrix D across experts. This is
    the expert-level analog of collect_block_geometry_swiglu.

    The resulting D matrix feeds into the same Physarum conductance →
    structural selection pipeline, with experts taking the place of
    neuron blocks.
    """
    moe = _get_moe_module(model, layer_idx)
    experts, gate = _get_experts_and_gate(moe)
    n_experts = len(experts)
    top_k = getattr(moe, "num_experts_per_tok", None) or getattr(moe, "top_k", 2)
    F_dim = n_features
    dim = F_dim * n_experts

    sum_z = torch.zeros(dim, dtype=torch.float64, device=device)
    sum_zz = torch.zeros(dim, dim, dtype=torch.float64, device=device)
    sum_mag = torch.zeros(n_experts, dtype=torch.float64, device=device)
    total_tokens = 0

    def hook(module, inputs, output):
        nonlocal sum_z, sum_zz, sum_mag, total_tokens
        if total_tokens >= max_tokens:
            return
        hidden = inputs[0]
        if hidden.dim() == 3:
            hidden = hidden.reshape(-1, hidden.shape[-1])
        T = hidden.shape[0]
        budget = max_tokens - total_tokens
        if T > budget:
            hidden = hidden[:budget]
            T = budget

        logits = gate(hidden.float())
        weights = F.softmax(logits, dim=-1)
        topk_weights, topk_indices = torch.topk(weights, top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        batch_size = min(T, 256)
        for start in range(0, T, batch_size):
            end = min(start + batch_size, T)
            batch_hidden = hidden[start:end]
            B = batch_hidden.shape[0]
            z = torch.zeros(B, dim, dtype=torch.float64, device=device)

            for k in range(top_k):
                batch_indices = topk_indices[start:end, k]
                batch_weights = topk_weights[start:end, k].to(torch.float64)

                unique_experts = batch_indices.unique()
                for eidx in unique_experts:
                    mask = batch_indices == eidx
                    if not mask.any():
                        continue
                    eidx_int = eidx.item()
                    exp_in = batch_hidden[mask]
                    exp_out = experts[eidx_int](exp_in).float().to(torch.float64)
                    w = batch_weights[mask].unsqueeze(1)
                    weighted_out = w * exp_out

                    base = F_dim * eidx_int
                    z[mask, base] += weighted_out.mean(dim=-1)
                    z[mask, base + 1] += weighted_out.std(dim=-1, correction=0)
                    z[mask, base + 2] += weighted_out.abs().mean(dim=-1)
                    sum_mag[eidx_int] += weighted_out.abs().mean(dim=-1).sum()

            sum_z += z.sum(dim=0)
            sum_zz += z.T @ z
            total_tokens += B

    h = moe.register_forward_hook(hook)
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
        "n_blocks": n_experts,
        "feature_multiplier": F_dim,
    }


# ── Batch calibration (all layers in one pass) ────────────────────────

import logging as _logging

_log = _logging.getLogger(__name__)


@torch.no_grad()
def collect_all_layers_covariance_and_utilization(
    model, tokenizer, texts: Sequence[str],
    max_len: int = 256, device: str = "cuda",
    max_tokens: int = 20_000, n_features: int = 3,
) -> Dict[int, Dict[str, Any]]:
    """Collect expert covariance + utilization for ALL MoE layers in a single forward sweep.

    Instead of running separate forward passes per layer, this registers hooks
    on every MoE layer simultaneously. One pass through the calibration texts
    yields data for all layers.

    Returns: {layer_idx: {"covariance": {...}, "utilization": {...}}}
    """
    layers = _get_layers_module(model)
    n_layers = len(layers)

    moe_layers: Dict[int, Any] = {}
    for li in range(n_layers):
        try:
            moe = _get_moe_module(model, li)
            moe_layers[li] = moe
        except ValueError:
            continue

    if not moe_layers:
        raise ValueError("No MoE layers found in model")

    first_li = next(iter(moe_layers))
    first_moe = moe_layers[first_li]
    experts_0, gate_0 = _get_experts_and_gate(first_moe)
    n_experts = len(experts_0)
    top_k = getattr(first_moe, "num_experts_per_tok", None) or getattr(first_moe, "top_k", 2)
    F_dim = n_features
    dim = F_dim * n_experts

    _log.info(
        "batch calibration: %d MoE layers, %d experts, top-%d, max_tokens=%d",
        len(moe_layers), n_experts, top_k, max_tokens,
    )

    class LayerState:
        __slots__ = ("sum_z", "sum_zz", "sum_mag", "expert_count",
                     "expert_weight_sum", "tokens")
        def __init__(self, dev):
            self.sum_z = torch.zeros(dim, dtype=torch.float64, device=dev)
            self.sum_zz = torch.zeros(dim, dim, dtype=torch.float64, device=dev)
            self.sum_mag = torch.zeros(n_experts, dtype=torch.float64, device=dev)
            self.expert_count = torch.zeros(n_experts, dtype=torch.float64, device=dev)
            self.expert_weight_sum = torch.zeros(n_experts, dtype=torch.float64, device=dev)
            self.tokens = 0

    # Each layer's state lives on the device where that layer's gate resides
    states: Dict[int, LayerState] = {}
    total_tokens = [0]

    for li, moe in moe_layers.items():
        _, gate = _get_experts_and_gate(moe)
        gate_device = next(gate.parameters()).device
        states[li] = LayerState(gate_device)

    hooks = []

    def make_hook(layer_idx, moe_mod, state):
        def hook(module, inputs, output):
            if total_tokens[0] >= max_tokens:
                return
            hidden = inputs[0]
            if hidden.dim() == 3:
                hidden = hidden.reshape(-1, hidden.shape[-1])
            T = hidden.shape[0]
            budget = max_tokens - total_tokens[0]
            if T > budget:
                hidden = hidden[:budget]
                T = budget

            experts, gate = _get_experts_and_gate(moe_mod)
            gate_dev = next(gate.parameters()).device
            hidden_f = hidden.to(gate_dev).float()
            logits = gate(hidden_f)
            weights = F.softmax(logits, dim=-1)
            topk_weights, topk_indices = torch.topk(weights, top_k, dim=-1)
            topk_weights_norm = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

            # --- Utilization ---
            for k in range(top_k):
                idx = topk_indices[:, k]
                w = topk_weights[:, k].to(torch.float64)
                state.expert_count.scatter_add_(0, idx.long().to(state.expert_count.device),
                                                torch.ones_like(w).to(state.expert_count.device))
                state.expert_weight_sum.scatter_add_(0, idx.long().to(state.expert_weight_sum.device),
                                                     w.to(state.expert_weight_sum.device))

            # --- Covariance features (subsample for speed) ---
            batch_size = min(T, 128)
            z = torch.zeros(batch_size, dim, dtype=torch.float64, device=gate_dev)

            for k in range(top_k):
                batch_indices = topk_indices[:batch_size, k]
                batch_weights = topk_weights_norm[:batch_size, k].to(torch.float64)
                unique_experts = batch_indices.unique()
                for eidx in unique_experts:
                    mask = batch_indices == eidx
                    if not mask.any():
                        continue
                    eidx_int = eidx.item()
                    exp_in = hidden_f[:batch_size][mask]
                    exp_dev = next(experts[eidx_int].parameters()).device
                    exp_out = experts[eidx_int](exp_in.to(exp_dev)).float().to(torch.float64).to(gate_dev)
                    w = batch_weights[mask].unsqueeze(1)
                    weighted_out = w * exp_out

                    base = F_dim * eidx_int
                    z[mask, base] += weighted_out.mean(dim=-1)
                    z[mask, base + 1] += weighted_out.std(dim=-1, correction=0)
                    z[mask, base + 2] += weighted_out.abs().mean(dim=-1)
                    state.sum_mag[eidx_int] += weighted_out.abs().mean(dim=-1).sum().to(state.sum_mag.device)

            z_on_dev = z.to(state.sum_z.device)
            state.sum_z += z_on_dev.sum(dim=0)
            state.sum_zz += z_on_dev.T @ z_on_dev
            state.tokens += batch_size

        return hook

    for li, moe in moe_layers.items():
        h = moe.register_forward_hook(make_hook(li, moe, states[li]))
        hooks.append(h)

    model.eval()
    _log.info("running %d calibration texts through model...", len(texts))
    for i, t in enumerate(texts):
        if total_tokens[0] >= max_tokens:
            break
        inp = tokenizer(t, return_tensors="pt", truncation=True, max_length=max_len)
        # device_map="auto" handles input placement via model.forward
        first_device = next(model.parameters()).device
        inp = {k: v.to(first_device) for k, v in inp.items()}
        model(**inp, use_cache=False)
        total_tokens[0] += inp["input_ids"].shape[1]
        if (i + 1) % 50 == 0:
            _log.info("  processed %d/%d texts (%d tokens)", i + 1, len(texts), total_tokens[0])

    for h in hooks:
        h.remove()

    _log.info("batch calibration complete: %d tokens processed", total_tokens[0])

    results: Dict[int, Dict[str, Any]] = {}
    for li, state in states.items():
        N_cov = max(state.tokens, 1)
        N_util = max(total_tokens[0], 1)
        mean_z = state.sum_z / N_cov
        D = (state.sum_zz / N_cov) - mean_z.unsqueeze(1) * mean_z.unsqueeze(0)

        results[li] = {
            "covariance": {
                "D": D.cpu(),
                "block_energy": (state.sum_mag / N_cov).cpu(),
                "n_blocks": n_experts,
                "feature_multiplier": F_dim,
            },
            "utilization": {
                "expert_frequency": (state.expert_count / N_util).cpu(),
                "expert_avg_weight": (state.expert_weight_sum / state.expert_count.clamp(min=1)).cpu(),
                "total_tokens": total_tokens[0],
                "n_experts": n_experts,
                "top_k": top_k,
            },
        }

    return results
