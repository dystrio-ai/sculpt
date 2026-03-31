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

    For iterable (ModuleList) experts only.
    """
    with torch.no_grad():
        for (name_k, p_k), (name_d, p_d) in zip(
            expert_keep.named_parameters(),
            expert_drop.named_parameters(),
        ):
            p_k.data.lerp_(p_d.data.to(p_k.device, p_k.dtype), 1.0 - alpha)


def _merge_fused_expert_weights(
    experts_module: torch.nn.Module,
    keep_idx: int,
    drop_idx: int,
    alpha: float = 0.5,
) -> None:
    """Merge a dropped fused expert's weights into a surviving one.

    For 3D fused weight tensors (e.g. gate_up_proj[n_experts, ...]),
    merges expert slice at drop_idx into keep_idx via lerp:
        W[keep_idx] = alpha * W[keep_idx] + (1-alpha) * W[drop_idx]

    This preserves the survivor's identity while absorbing the dropped
    expert's knowledge, using Physarum coupling to pick the best recipient.
    """
    with torch.no_grad():
        for attr in ("gate_up_proj", "down_proj"):
            param = getattr(experts_module, attr, None)
            if param is None:
                continue
            data = param.data if isinstance(param, torch.nn.Parameter) else param
            data[keep_idx].lerp_(
                data[drop_idx].to(data[keep_idx].dtype),
                1.0 - alpha,
            )


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

    Handles both iterable experts (ModuleList, e.g. Mixtral) and fused
    experts (3D Parameter tensors, e.g. Qwen3.5-MoE).
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
        """For MoE, return num_experts so the engine treats each expert as a 'block'."""
        return self.get_num_experts(model, layer_idx)

    def get_num_experts(self, model, layer_idx: int) -> int:
        from .._calibrate_moe import _get_num_experts
        moe = _get_moe_module(model, layer_idx)
        return _get_num_experts(moe, model)

    # ── Compression ───────────────────────────────────────────────────────

    def compress_layer(
        self, model, layer_idx: int, kept_idx: torch.Tensor,
        dtype: torch.dtype, device: str,
        merge: bool = True,
        coupling_matrix: Optional[np.ndarray] = None,
    ) -> Dict[str, int]:
        """Drop experts not in kept_idx, optionally merging their weights.

        Handles both iterable (ModuleList) and fused (3D tensor) experts.
        """
        from .._calibrate_moe import _experts_are_iterable, _get_num_experts

        moe = _get_moe_module(model, layer_idx)
        experts, gate = _get_experts_and_gate(moe)
        n_orig = _get_num_experts(moe, model)
        kept = sorted(kept_idx.tolist()) if isinstance(kept_idx, torch.Tensor) else sorted(kept_idx)
        dropped = [i for i in range(n_orig) if i not in set(kept)]

        _log.info(
            "layer %d: dropping %d/%d experts (keeping %d)",
            layer_idx, len(dropped), n_orig, len(kept),
        )

        fused = not _experts_are_iterable(moe)

        n_merged = 0
        if not fused:
            # --- Iterable experts (ModuleList): merge then rebuild ---
            if merge and coupling_matrix is not None and len(dropped) > 0:
                merge_plan = _build_expert_merge_plan(kept, dropped, coupling_matrix)
                for d_idx, k_idx in merge_plan.items():
                    _log.debug("  merge expert %d → %d (coupling=%.3f)",
                               d_idx, k_idx, coupling_matrix[d_idx, k_idx])
                    _merge_expert_weights(experts[k_idx], experts[d_idx], alpha=0.7)
                    n_merged += 1

            new_experts = torch.nn.ModuleList([experts[i] for i in kept])
            if hasattr(moe, "experts"):
                moe.experts = new_experts
            elif hasattr(moe, "block_sparse_moe"):
                from .._calibrate_moe import _get_layers_module
                _get_layers_module(model)[layer_idx].block_sparse_moe.experts = new_experts
        else:
            # --- Fused experts (3D tensors): merge then slice ---
            if merge and coupling_matrix is not None and len(dropped) > 0:
                merge_plan = _build_expert_merge_plan(kept, dropped, coupling_matrix)
                for d_idx, k_idx in merge_plan.items():
                    _log.debug("  merge fused expert %d → %d (coupling=%.3f)",
                               d_idx, k_idx, coupling_matrix[d_idx, k_idx])
                    _merge_fused_expert_weights(experts, k_idx, d_idx, alpha=0.7)
                    n_merged += 1

            kept_t = torch.tensor(kept, dtype=torch.long, device=device)
            for attr in ("gate_up_proj", "down_proj"):
                param = getattr(experts, attr, None)
                if param is None:
                    continue
                if isinstance(param, torch.nn.Parameter):
                    new_data = param.data[kept_t].clone().to(dtype)
                    new_param = torch.nn.Parameter(new_data)
                    setattr(experts, attr, new_param)
                elif hasattr(param, "data"):
                    param.data = param.data[kept_t].to(dtype)

        # --- Resize router gate ---
        # For custom routers (OlmoeTopKRouter, TopKRouter, etc.), modify
        # the weight parameter in-place to preserve the router's forward()
        # signature (which returns logits, weights, indices — not just a
        # raw tensor like nn.Linear).
        if hasattr(gate, "weight"):
            old_w = gate.weight
            is_plain_linear = isinstance(gate, torch.nn.Linear)

            if is_plain_linear:
                has_bias = getattr(gate, "bias", None) is not None
                if old_w.shape[0] == n_orig:
                    new_gate = torch.nn.Linear(
                        old_w.shape[1], len(kept), bias=has_bias,
                        device=device, dtype=dtype,
                    )
                    new_gate.weight.data.copy_(old_w.data[kept].to(dtype))
                    if has_bias:
                        new_gate.bias.data.copy_(gate.bias.data[kept].to(dtype))
                else:
                    new_gate = torch.nn.Linear(
                        len(kept), old_w.shape[0], bias=has_bias,
                        device=device, dtype=dtype,
                    )
                    new_gate.weight.data.copy_(old_w.data[:, kept].to(dtype))
                    if has_bias:
                        new_gate.bias.data.copy_(gate.bias.data.to(dtype))
                if hasattr(moe, "gate"):
                    moe.gate = new_gate
                elif hasattr(moe, "router"):
                    moe.router = new_gate
            else:
                # Custom router: slice weight in-place, keep the module
                if old_w.shape[0] == n_orig:
                    gate.weight = torch.nn.Parameter(
                        old_w.data[kept].to(dtype)
                    )
                else:
                    gate.weight = torch.nn.Parameter(
                        old_w.data[:, kept].to(dtype)
                    )
                has_bias = getattr(gate, "bias", None) is not None
                if has_bias and gate.bias.shape[0] == n_orig:
                    gate.bias = torch.nn.Parameter(
                        gate.bias.data[kept].to(dtype)
                    )
                # Update num_experts on the router if it tracks it
                for attr in ("num_experts", "top_k_experts", "n_experts"):
                    if hasattr(gate, attr) and getattr(gate, attr) == n_orig:
                        setattr(gate, attr, len(kept))

        # --- Update num_experts on all relevant modules ---
        # OlmoeExperts.forward() uses self.num_experts for one_hot encoding.
        # OlmoeTopKRouter stores num_experts too (handled above for custom routers).
        # OlmoeForCausalLM.num_experts is used in aux loss computation.
        n_new = len(kept)
        for obj in (experts, moe, model):
            for attr in ("num_experts", "num_local_experts"):
                if hasattr(obj, attr) and getattr(obj, attr) == n_orig:
                    setattr(obj, attr, n_new)

        # --- Update config ---
        from .._model import get_text_config
        text_cfg = get_text_config(model)
        for attr in ("num_local_experts", "num_experts"):
            if hasattr(text_cfg, attr):
                setattr(text_cfg, attr, n_new)
            if hasattr(model.config, attr):
                setattr(model.config, attr, n_new)

        _log.info(
            "layer %d: %d → %d experts  (fused=%s, merged=%d)",
            layer_idx, n_orig, len(kept), fused, n_merged,
        )

        return {
            "n_experts_orig": n_orig,
            "n_experts_kept": len(kept),
            "n_merged": n_merged,
            "ffn_kept": len(kept),
            "hidden": getattr(text_cfg, "hidden_size", 0),
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
        """Expert operator sensitivity: ||w_k * expert_k(x)||^2.

        Uses the same operator-fidelity signal as the dense SwiGLU pipeline.
        For fused experts, manually slices 3D weight tensors and runs
        per-expert SwiGLU forward passes inside the calibration hook.
        """
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
        """Expert-level importance: routing frequency * avg gate weight (REAP-style).

        Works with both fused and iterable experts since it only uses
        router logits, not expert forward passes.
        """
        from .._calibrate_moe import score_expert_importance
        scores = score_expert_importance(
            model, tokenizer, layer_idx, texts, max_len, device,
        )
        return scores["importance"]

    # ── Repair support ────────────────────────────────────────────────────

    def snapshot_trainable(
        self, model, layers: Sequence[int],
    ) -> Dict[str, torch.Tensor]:
        snap: Dict[str, torch.Tensor] = {}
        for li in layers:
            moe = _get_moe_module(model, li)
            experts, gate = _get_experts_and_gate(moe)
            for name, p in experts.named_parameters():
                key = f"layers.{li}.experts.{name}"
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
            for name, p in experts.named_parameters():
                key = f"layers.{li}.experts.{name}"
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
            params.extend(experts.parameters())
            params.extend(gate.parameters())
        return params

    # ── Model routing ─────────────────────────────────────────────────────

    def get_eval_model(self, model):
        """Return the text model for text-only inference.

        Qwen3.5 MoE multimodal wrappers (ForConditionalGeneration) have a
        nested .model or .language_model that accepts standard text inputs.
        Standalone text-only MoE models just return themselves.
        """
        if hasattr(model, "language_model"):
            return model.language_model
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            test_cfg = getattr(model.config, "text_config", None)
            if test_cfg is not None and hasattr(model.config, "vision_config"):
                return model
        return model
