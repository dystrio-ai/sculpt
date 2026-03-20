"""SwiGLU MoE adapter for Mixtral, DeepSeek-MoE, Qwen-MoE.

MoE architectures use sparse expert selection with SwiGLU FFN blocks per
expert.  Optimization operates at TWO levels:

  1. Expert-level: drop/merge entire experts using Physarum structural selection
     on expert covariance, then rescale the router to compensate.
  2. Intra-expert: apply standard neuron-block pruning within each surviving
     expert (reuses the dense SwiGLU pipeline per expert).

Expert merging uses Physarum conductance to identify functionally redundant
expert pairs — experts with high coupling have correlated outputs, so one
can absorb the other's weights via linear interpolation.

Router rescaling: after dropping experts, the router's output logits are
rewritten to exclude dropped expert columns, preserving routing distribution
over surviving experts.
"""

from __future__ import annotations

import copy
import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
import numpy as np

from .base import ArchitectureAdapter
from .descriptor import OptimizationTarget

_log = logging.getLogger(__name__)


def _get_moe_module(model, layer_idx: int):
    """Locate the MoE module. Delegates to _calibrate_moe for multimodal support."""
    from .._calibrate_moe import _get_moe_module as _get_moe
    return _get_moe(model, layer_idx)


def _get_experts_and_gate(moe_module):
    """Extract expert list and gating network. Delegates to _calibrate_moe."""
    from .._calibrate_moe import _get_experts_and_gate as _get_eg
    return _get_eg(moe_module)


def _merge_expert_weights(
    expert_keep: torch.nn.Module,
    expert_drop: torch.nn.Module,
    alpha: float = 0.5,
) -> None:
    """Merge expert_drop's weights into expert_keep via linear interpolation.

    After merging, expert_keep ≈ alpha * keep + (1-alpha) * drop.
    This preserves the surviving expert's identity while absorbing
    knowledge from the redundant one.
    """
    with torch.no_grad():
        for (name_k, p_k), (name_d, p_d) in zip(
            expert_keep.named_parameters(),
            expert_drop.named_parameters(),
        ):
            p_k.data.lerp_(p_d.data.to(p_k.device, p_k.dtype), 1.0 - alpha)


def _build_expert_merge_plan(
    kept_indices: List[int],
    dropped_indices: List[int],
    coupling_matrix: np.ndarray,
) -> Dict[int, int]:
    """For each dropped expert, find the most coupled surviving expert to merge into.

    Returns: {dropped_idx: merge_target_idx}
    """
    merge_plan: Dict[int, int] = {}
    kept_set = set(kept_indices)
    for d in dropped_indices:
        best_target = -1
        best_coupling = -1.0
        for k in kept_indices:
            c = float(coupling_matrix[d, k])
            if c > best_coupling:
                best_coupling = c
                best_target = k
        if best_target >= 0:
            merge_plan[d] = best_target
    return merge_plan


class SwiGLUMoEAdapter(ArchitectureAdapter):
    """Adapter for Mixture-of-Experts with SwiGLU FFN blocks.

    Supports expert-level compression (drop/merge) and the standard
    calibration/selection interface for integration with the search engine.
    """

    def supported_targets(self) -> List[OptimizationTarget]:
        return [OptimizationTarget.EXPERT_BLOCK, OptimizationTarget.MLP_BLOCK]

    # ── Layer access ──────────────────────────────────────────────────────

    def get_num_layers(self, model) -> int:
        from .._calibrate_moe import _get_layers_module
        return len(_get_layers_module(model))

    def get_mlp(self, model, layer_idx: int):
        return _get_moe_module(model, layer_idx)

    def get_ffn_size(self, model, layer_idx: int) -> int:
        """Return total expert FFN width (single expert's intermediate_size).

        For expert-level selection, n_blocks = n_experts, and each "block"
        is one expert. The Physarum pipeline doesn't use ffn_size directly
        for expert selection — it uses the covariance matrix shape.
        """
        moe = _get_moe_module(model, layer_idx)
        experts, _ = _get_experts_and_gate(moe)
        if len(experts) > 0:
            expert = experts[0]
            for attr in ("w1", "gate_proj"):
                if hasattr(expert, attr):
                    return getattr(expert, attr).out_features
        raise ValueError("Cannot determine FFN size for MoE expert")

    def get_num_experts(self, model, layer_idx: int) -> int:
        moe = _get_moe_module(model, layer_idx)
        experts, _ = _get_experts_and_gate(moe)
        return len(experts)

    # ── Compression ───────────────────────────────────────────────────────

    def compress_layer(
        self, model, layer_idx: int, kept_idx: torch.Tensor,
        dtype: torch.dtype, device: str,
        merge: bool = True,
        coupling_matrix: Optional[np.ndarray] = None,
    ) -> Dict[str, int]:
        """Drop experts not in kept_idx, optionally merging their weights.

        1. Identify dropped experts
        2. If merge=True and coupling_matrix provided, merge each dropped
           expert into its most-coupled survivor
        3. Rebuild the expert ModuleList with only survivors
        4. Slice the router weight matrix to exclude dropped columns
        """
        moe = _get_moe_module(model, layer_idx)
        experts, gate = _get_experts_and_gate(moe)
        n_orig = len(experts)
        kept = sorted(kept_idx.tolist()) if isinstance(kept_idx, torch.Tensor) else sorted(kept_idx)
        dropped = [i for i in range(n_orig) if i not in set(kept)]

        _log.info(
            "layer %d: dropping %d/%d experts (keeping %s)",
            layer_idx, len(dropped), n_orig, kept,
        )

        if merge and coupling_matrix is not None and len(dropped) > 0:
            merge_plan = _build_expert_merge_plan(kept, dropped, coupling_matrix)
            for d_idx, k_idx in merge_plan.items():
                _log.debug(
                    "  merging expert %d → %d (coupling=%.3f)",
                    d_idx, k_idx, coupling_matrix[d_idx, k_idx],
                )
                _merge_expert_weights(experts[k_idx], experts[d_idx], alpha=0.7)

        new_experts = torch.nn.ModuleList([experts[i] for i in kept])

        if hasattr(moe, "experts"):
            moe.experts = new_experts
        elif hasattr(moe, "block_sparse_moe"):
            model.model.layers[layer_idx].block_sparse_moe.experts = new_experts

        # Rescale router: slice out columns for dropped experts
        if hasattr(gate, "weight"):
            old_w = gate.weight.data
            new_w = old_w[kept, :] if old_w.shape[0] == n_orig else old_w[:, kept]
            # Determine correct dimension — router maps hidden→n_experts
            if old_w.shape[0] == n_orig:
                new_gate = torch.nn.Linear(
                    old_w.shape[1], len(kept), bias=gate.bias is not None,
                    device=device, dtype=dtype,
                )
                new_gate.weight.data.copy_(old_w[kept].to(dtype))
                if gate.bias is not None:
                    new_gate.bias.data.copy_(gate.bias.data[kept].to(dtype))
            else:
                new_gate = torch.nn.Linear(
                    old_w.shape[0], len(kept), bias=gate.bias is not None,
                    device=device, dtype=dtype,
                )
                new_gate.weight.data.copy_(old_w[:, kept].T.to(dtype))
                if gate.bias is not None:
                    new_gate.bias.data.copy_(gate.bias.data[kept].to(dtype))

            if hasattr(moe, "gate"):
                moe.gate = new_gate
            elif hasattr(moe, "router"):
                moe.router = new_gate

        # Update config
        if hasattr(model.config, "num_local_experts"):
            model.config.num_local_experts = len(kept)
        if hasattr(model.config, "num_experts"):
            model.config.num_experts = len(kept)

        _log.info(
            "layer %d: %d → %d experts  (merged=%d)",
            layer_idx, n_orig, len(kept), len(dropped) if merge else 0,
        )

        return {
            "n_experts_orig": n_orig,
            "n_experts_kept": len(kept),
            "n_merged": len(dropped) if merge else 0,
            "ffn_kept": self.get_ffn_size(model, layer_idx),
            "hidden": model.config.hidden_size,
        }

    # ── Calibration ───────────────────────────────────────────────────────

    def collect_block_geometry(
        self, model, tokenizer, layer_idx: int, texts: Sequence[str],
        max_len: int, device: str, block_size: int = 128, max_tokens: int = 30_000,
    ) -> Dict[str, Any]:
        """Expert-level covariance for Physarum structural selection."""
        from .._calibrate_moe import collect_expert_covariance
        return collect_expert_covariance(
            model, tokenizer, layer_idx, texts, max_len, device,
            max_tokens=max_tokens,
        )

    def collect_block_sensitivity(
        self, model, tokenizer, layer_idx: int, texts: Sequence[str],
        max_len: int, device: str, block_size: int = 128, max_tokens: int = 30_000,
    ) -> Dict[str, Any]:
        """Expert-level operator sensitivity."""
        from .._calibrate_moe import collect_expert_sensitivity
        result = collect_expert_sensitivity(
            model, tokenizer, layer_idx, texts, max_len, device,
            max_tokens=max_tokens,
        )
        return {
            "block_sensitivity": result["expert_sensitivity"],
            "block_energy": result["expert_energy"],
            "n_blocks": result["n_experts"],
        }

    def collect_importance(
        self, model, tokenizer, layer_idx: int, texts: Sequence[str],
        max_len: int, device: str,
    ) -> torch.Tensor:
        """Expert-level importance based on utilization-weighted sensitivity."""
        from .._calibrate_moe import collect_expert_sensitivity, collect_expert_utilization
        sens = collect_expert_sensitivity(
            model, tokenizer, layer_idx, texts, max_len, device,
        )
        util = collect_expert_utilization(
            model, tokenizer, layer_idx, texts, max_len, device,
        )
        importance = sens["expert_sensitivity"] * util["expert_frequency"]
        return importance

    # ── Repair support ────────────────────────────────────────────────────

    def snapshot_trainable(
        self, model, layers: Sequence[int],
    ) -> Dict[str, torch.Tensor]:
        snap: Dict[str, torch.Tensor] = {}
        for li in layers:
            moe = _get_moe_module(model, li)
            experts, gate = _get_experts_and_gate(moe)
            for ei, expert in enumerate(experts):
                for name, p in expert.named_parameters():
                    key = f"layers.{li}.expert.{ei}.{name}"
                    snap[key] = p.data.detach().cpu().clone()
            for name, p in gate.named_parameters():
                key = f"layers.{li}.gate.{name}"
                snap[key] = p.data.detach().cpu().clone()
        return snap

    def restore_trainable(
        self, model, layers: Sequence[int],
        snap: Dict[str, torch.Tensor],
    ) -> None:
        for li in layers:
            moe = _get_moe_module(model, li)
            experts, gate = _get_experts_and_gate(moe)
            for ei, expert in enumerate(experts):
                for name, p in expert.named_parameters():
                    key = f"layers.{li}.expert.{ei}.{name}"
                    if key in snap:
                        p.data.copy_(snap[key].to(p.device))
            for name, p in gate.named_parameters():
                key = f"layers.{li}.gate.{name}"
                if key in snap:
                    p.data.copy_(snap[key].to(p.device))

    def get_trainable_params(
        self, model, layers: Sequence[int],
    ) -> List[torch.nn.Parameter]:
        params = []
        for li in layers:
            moe = _get_moe_module(model, li)
            experts, gate = _get_experts_and_gate(moe)
            for expert in experts:
                params.extend(expert.parameters())
            params.extend(gate.parameters())
        return params
