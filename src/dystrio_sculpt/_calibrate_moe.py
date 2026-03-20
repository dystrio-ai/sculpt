"""Expert-level calibration for MoE models.

Two calibration modes:

1. **Router-based** (works with ANY expert implementation, including fused):
   Uses gate logit correlation to identify which experts compete for the same
   tokens. This is the primary mode for routing canonicalization.

2. **Expert-output-based** (requires iterable experts, e.g. ModuleList):
   Runs individual expert forwards to build output covariance. Used for
   expert-level structural compression (drop/merge).

Architecture reference (Qwen3.5-MoE / Qwen2MoE family):
  layer.mlp = Qwen3_5MoeSparseMoeBlock
    .gate   = Qwen3_5MoeTopKRouter  (has .weight nn.Parameter [num_experts, hidden])
              forward() returns (post_softmax_logits, top_k_weights, top_k_indices)
    .experts = Qwen3_5MoeExperts  (FUSED: .gate_up_proj, .down_proj as 3D Parameters)
              NOT a ModuleList — cannot len() or index
    .shared_expert = Qwen3_5MoeMLP
    .shared_expert_gate = nn.Linear(hidden, 1)

To get raw pre-softmax logits: F.linear(hidden, gate.weight)
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

_log = logging.getLogger(__name__)


# ── Model structure helpers ───────────────────────────────────────────

def _get_layers_module(model):
    """Locate the decoder layer list, handling multimodal wrappers."""
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
    """Locate the MoE module (SparseMoeBlock) in a transformer layer."""
    layers = _get_layers_module(model)
    layer = layers[layer_idx]
    if hasattr(layer, "block_sparse_moe"):
        return layer.block_sparse_moe
    if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
        return layer.mlp
    raise ValueError(f"Cannot locate MoE module in layer {layer_idx}")


def _get_gate(moe_module) -> torch.nn.Module:
    """Extract the gating/router network from an MoE module."""
    if hasattr(moe_module, "gate"):
        return moe_module.gate
    if hasattr(moe_module, "router"):
        return moe_module.router
    raise ValueError("Cannot find gate/router in MoE module")


def _get_gate_weight(moe_module) -> torch.nn.Parameter:
    """Get the raw gate weight matrix for computing pre-softmax logits.

    Works with both nn.Linear gates and TopKRouter gates that store
    weight as a plain nn.Parameter.
    """
    gate = _get_gate(moe_module)
    if hasattr(gate, "weight"):
        return gate.weight
    raise ValueError("Gate has no weight attribute")


def _compute_raw_logits(hidden: torch.Tensor, gate_weight: torch.nn.Parameter) -> torch.Tensor:
    """Compute raw pre-softmax router logits via F.linear.

    Bypasses the router's forward() to avoid:
    - Tuple return values
    - Internal softmax (which we want to control ourselves)
    - Any router-specific post-processing
    """
    return F.linear(hidden.to(gate_weight.dtype), gate_weight)


def _get_num_experts(moe_module, model=None) -> int:
    """Get the number of experts from the gate weight shape or config."""
    gate = _get_gate(moe_module)
    if hasattr(gate, "weight"):
        return gate.weight.shape[0]
    if hasattr(gate, "num_experts"):
        return gate.num_experts

    if model is not None:
        cfg = model.config
        if hasattr(cfg, "text_config") and cfg.text_config is not None:
            cfg = cfg.text_config
        for attr in ("num_experts", "num_local_experts"):
            if hasattr(cfg, attr):
                return getattr(cfg, attr)

    raise ValueError("Cannot determine number of experts")


def _get_top_k(moe_module) -> int:
    """Get the number of experts selected per token."""
    gate = _get_gate(moe_module)
    if hasattr(gate, "top_k"):
        return gate.top_k
    return getattr(moe_module, "num_experts_per_tok", None) or getattr(moe_module, "top_k", 2)


def _experts_are_iterable(moe_module) -> bool:
    """Check if experts can be individually indexed (ModuleList vs fused)."""
    experts = getattr(moe_module, "experts", None)
    if experts is None:
        return False
    try:
        n = len(experts)
        if n > 0:
            _ = experts[0]
        return True
    except (TypeError, IndexError, KeyError):
        return False


def _get_experts_and_gate(moe_module):
    """Extract expert module and gating network. For backward compat."""
    experts = getattr(moe_module, "experts", None)
    gate = _get_gate(moe_module)
    if experts is None:
        raise ValueError("Cannot find experts in MoE module")
    return experts, gate


# ── Router-based calibration (works with fused experts) ───────────────

@torch.no_grad()
def collect_router_logit_covariance(
    model, tokenizer, layer_idx: int, texts: Sequence[str],
    max_len: int, device: str, max_tokens: int = 30_000,
) -> Dict[str, Any]:
    """Build expert covariance from router logit correlation.

    Computes raw pre-softmax logits via F.linear(hidden, gate.weight),
    then builds a covariance matrix over softmax probabilities.
    Works with ANY expert implementation (fused or ModuleList).
    """
    moe = _get_moe_module(model, layer_idx)
    gate_weight = _get_gate_weight(moe)
    n_experts = gate_weight.shape[0]

    sum_z = torch.zeros(n_experts, dtype=torch.float64, device=device)
    sum_zz = torch.zeros(n_experts, n_experts, dtype=torch.float64, device=device)
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

        logits = _compute_raw_logits(hidden, gate_weight)
        probs = F.softmax(logits.float(), dim=-1).to(torch.float64)

        sum_z += probs.sum(dim=0).to(device)
        sum_zz += (probs.T @ probs).to(device)
        sum_mag += probs.abs().mean(dim=0).to(device)
        total_tokens += T

    h = moe.register_forward_hook(hook)
    model.eval()
    for t in texts:
        if total_tokens >= max_tokens:
            break
        inp = tokenizer(t, return_tensors="pt", truncation=True, max_length=max_len)
        first_device = next(model.parameters()).device
        inp = {k: v.to(first_device) for k, v in inp.items()}
        model(**inp, use_cache=False)
    h.remove()

    N = max(total_tokens, 1)
    mean_z = sum_z / N
    D = (sum_zz / N) - mean_z.unsqueeze(1) * mean_z.unsqueeze(0)

    return {
        "D": D.cpu(),
        "block_energy": (sum_mag / N).cpu(),
        "n_blocks": n_experts,
        "feature_multiplier": 1,
    }


@torch.no_grad()
def collect_expert_utilization(
    model, tokenizer, layer_idx: int, texts: Sequence[str],
    max_len: int, device: str, max_tokens: int = 30_000,
) -> Dict[str, Any]:
    """Measure per-expert utilization using raw gate logits.

    Works with ANY expert implementation (only uses gate.weight).
    """
    moe = _get_moe_module(model, layer_idx)
    gate_weight = _get_gate_weight(moe)
    n_experts = gate_weight.shape[0]
    top_k = _get_top_k(moe)

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

        logits = _compute_raw_logits(hidden, gate_weight)
        weights = F.softmax(logits.float(), dim=-1)
        topk_weights, topk_indices = torch.topk(weights, top_k, dim=-1)

        for k in range(top_k):
            idx = topk_indices[:, k].to(device)
            w = topk_weights[:, k].to(torch.float64).to(device)
            expert_count.scatter_add_(0, idx.long(), torch.ones_like(w))
            expert_weight_sum.scatter_add_(0, idx.long(), w)

        total_tokens += T

    h = moe.register_forward_hook(hook)
    model.eval()
    for t in texts:
        if total_tokens >= max_tokens:
            break
        inp = tokenizer(t, return_tensors="pt", truncation=True, max_length=max_len)
        first_device = next(model.parameters()).device
        inp = {k: v.to(first_device) for k, v in inp.items()}
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


# ── Expert-output-based calibration (requires iterable experts) ───────

@torch.no_grad()
def collect_expert_sensitivity(
    model, tokenizer, layer_idx: int, texts: Sequence[str],
    max_len: int, device: str, max_tokens: int = 30_000,
) -> Dict[str, Any]:
    """Expert operator sensitivity. REQUIRES iterable experts (ModuleList)."""
    moe = _get_moe_module(model, layer_idx)
    if not _experts_are_iterable(moe):
        raise TypeError(
            "collect_expert_sensitivity requires iterable experts (ModuleList). "
            "This model uses fused experts. Use router-based calibration instead."
        )
    experts, gate = _get_experts_and_gate(moe)
    gate_weight = _get_gate_weight(moe)
    n_experts = len(experts)
    top_k = _get_top_k(moe)

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

        logits = _compute_raw_logits(hidden, gate_weight)
        weights = F.softmax(logits.float(), dim=-1)
        topk_weights, topk_indices = torch.topk(weights, top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        for t_idx in range(min(T, 512)):
            tok_hidden = hidden[t_idx : t_idx + 1]
            tok_experts = topk_indices[t_idx]
            tok_weights = topk_weights[t_idx]

            for k in range(top_k):
                eidx = tok_experts[k].item()
                exp_out = experts[eidx](tok_hidden).float()
                w = tok_weights[k].float()
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
    """Expert-level covariance from expert outputs.

    REQUIRES iterable experts. Falls back to router logit covariance
    for fused experts.
    """
    moe = _get_moe_module(model, layer_idx)
    if not _experts_are_iterable(moe):
        _log.info(
            "layer %d: fused experts detected, falling back to router logit covariance",
            layer_idx,
        )
        return collect_router_logit_covariance(
            model, tokenizer, layer_idx, texts, max_len, device, max_tokens,
        )

    experts, gate = _get_experts_and_gate(moe)
    gate_weight = _get_gate_weight(moe)
    n_experts = len(experts)
    top_k = _get_top_k(moe)
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

        logits = _compute_raw_logits(hidden, gate_weight)
        weights = F.softmax(logits.float(), dim=-1)
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

    return {
        "D": D.cpu(),
        "block_energy": (sum_mag / N).cpu(),
        "n_blocks": n_experts,
        "feature_multiplier": F_dim,
    }


# ── Batch calibration (all layers in one pass, router-based) ─────────

@torch.no_grad()
def collect_all_layers_covariance_and_utilization(
    model, tokenizer, texts: Sequence[str],
    max_len: int = 256, device: str = "cuda",
    max_tokens: int = 20_000,
) -> Dict[int, Dict[str, Any]]:
    """Collect expert covariance + utilization for ALL MoE layers in one sweep.

    Uses raw gate logits via F.linear(hidden, gate.weight) — works with
    any expert implementation including fused experts.
    """
    layers = _get_layers_module(model)
    n_layers = len(layers)

    moe_layers: Dict[int, Any] = {}
    gate_weights: Dict[int, torch.nn.Parameter] = {}
    for li in range(n_layers):
        try:
            moe = _get_moe_module(model, li)
            gw = _get_gate_weight(moe)
            moe_layers[li] = moe
            gate_weights[li] = gw
        except ValueError:
            continue

    if not moe_layers:
        raise ValueError("No MoE layers found in model")

    first_li = next(iter(moe_layers))
    n_experts = gate_weights[first_li].shape[0]
    top_k = _get_top_k(moe_layers[first_li])

    _log.info(
        "batch calibration: %d MoE layers, %d experts, top-%d, max_tokens=%d",
        len(moe_layers), n_experts, top_k, max_tokens,
    )

    class LayerState:
        __slots__ = ("sum_z", "sum_zz", "sum_mag", "expert_count",
                     "expert_weight_sum", "tokens")
        def __init__(self, dev):
            self.sum_z = torch.zeros(n_experts, dtype=torch.float64, device=dev)
            self.sum_zz = torch.zeros(n_experts, n_experts, dtype=torch.float64, device=dev)
            self.sum_mag = torch.zeros(n_experts, dtype=torch.float64, device=dev)
            self.expert_count = torch.zeros(n_experts, dtype=torch.float64, device=dev)
            self.expert_weight_sum = torch.zeros(n_experts, dtype=torch.float64, device=dev)
            self.tokens = 0

    states: Dict[int, LayerState] = {}
    total_tokens = [0]

    for li in moe_layers:
        gw_dev = gate_weights[li].device
        states[li] = LayerState(gw_dev)

    hooks = []

    def make_hook(layer_idx, gw, state):
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

            logits = _compute_raw_logits(hidden, gw)
            probs = F.softmax(logits.float(), dim=-1)

            probs_d = probs.to(torch.float64)
            state.sum_z += probs_d.sum(dim=0)
            state.sum_zz += probs_d.T @ probs_d
            state.sum_mag += probs_d.abs().mean(dim=0)
            state.tokens += T

            topk_weights, topk_indices = torch.topk(probs, top_k, dim=-1)
            for k in range(top_k):
                idx = topk_indices[:, k]
                w = topk_weights[:, k].to(torch.float64)
                state.expert_count.scatter_add_(0, idx.long(), torch.ones_like(w))
                state.expert_weight_sum.scatter_add_(0, idx.long(), w)

        return hook

    for li, moe in moe_layers.items():
        h = moe.register_forward_hook(make_hook(li, gate_weights[li], states[li]))
        hooks.append(h)

    model.eval()
    _log.info("running %d calibration texts through model...", len(texts))
    for i, t in enumerate(texts):
        if total_tokens[0] >= max_tokens:
            break
        inp = tokenizer(t, return_tensors="pt", truncation=True, max_length=max_len)
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
        N = max(state.tokens, 1)
        N_util = max(total_tokens[0], 1)
        mean_z = state.sum_z / N
        D = (state.sum_zz / N) - mean_z.unsqueeze(1) * mean_z.unsqueeze(0)

        results[li] = {
            "covariance": {
                "D": D.cpu(),
                "block_energy": (state.sum_mag / N).cpu(),
                "n_blocks": n_experts,
                "feature_multiplier": 1,
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
