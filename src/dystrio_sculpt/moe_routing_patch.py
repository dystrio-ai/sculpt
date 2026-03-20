"""MoE routing canonicalization for prefix cache compatibility.

Problem: MoE models route each token to top-k experts. With FP4/FP8
quantization, tiny numerical differences can flip which experts are
selected, invalidating the KV cache for identical prefixes.

Solution: Use Physarum conductance analysis to identify expert equivalence
classes (groups of functionally interchangeable experts). At inference,
when the router hesitates between experts in the same class, snap to the
canonical representative. This makes routing deterministic without changing
model quality — the experts in each class produce near-identical outputs.

Usage:
    from dystrio_sculpt.moe_routing_patch import calibrate_routing_patch, apply_routing_patch

    # Step 1: Calibrate (runs Physarum analysis, ~2 min on A100)
    patch = calibrate_routing_patch(model, tokenizer, texts, device="cuda")

    # Step 2: Apply (monkey-patches router forward, zero-overhead)
    apply_routing_patch(model, patch)

    # Model now has deterministic routing → prefix caching works in vLLM
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_log = logging.getLogger(__name__)


@dataclass
class ExpertEquivalenceClass:
    """A group of experts that produce near-identical outputs."""
    canonical: int
    members: List[int]
    mean_coupling: float


@dataclass
class RoutingPatch:
    """Routing canonicalization patch for one MoE model.

    Contains per-layer equivalence classes derived from Physarum conductance.
    """
    model_id: str
    n_experts_original: int
    top_k: int
    layers: Dict[int, List[ExpertEquivalenceClass]] = field(default_factory=dict)
    coupling_threshold: float = 0.7
    margin_threshold: float = 0.1

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "model_id": self.model_id,
            "n_experts_original": self.n_experts_original,
            "top_k": self.top_k,
            "coupling_threshold": self.coupling_threshold,
            "margin_threshold": self.margin_threshold,
            "layers": {
                str(li): [asdict(ec) for ec in ecs]
                for li, ecs in self.layers.items()
            },
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        _log.info("saved routing patch to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> RoutingPatch:
        with open(path) as f:
            data = json.load(f)
        patch = cls(
            model_id=data["model_id"],
            n_experts_original=data["n_experts_original"],
            top_k=data["top_k"],
            coupling_threshold=data.get("coupling_threshold", 0.7),
            margin_threshold=data.get("margin_threshold", 0.1),
        )
        for li_str, ecs_data in data["layers"].items():
            patch.layers[int(li_str)] = [
                ExpertEquivalenceClass(**ec) for ec in ecs_data
            ]
        return patch


def _build_expert_coupling_matrix(
    D: torch.Tensor,
    n_experts: int,
    feature_multiplier: int = 3,
) -> np.ndarray:
    """Convert expert covariance D into an expert-expert coupling matrix.

    Uses the same correlation + Physarum pipeline as structural selection,
    but returns the full expert coupling matrix instead of selecting blocks.
    """
    from .selectors.structural import build_graph_from_cov, physarum_conductance

    u, v, w = build_graph_from_cov(D, k=min(20, n_experts - 1))
    if len(u) == 0:
        return np.eye(n_experts, dtype=np.float64)

    FM = feature_multiplier
    k_edge = physarum_conductance(u, v, w, FM * n_experts, n_iters=200)

    coupling = np.zeros((n_experts, n_experts), dtype=np.float64)
    for e in range(len(u)):
        bu = int(u[e]) // FM
        bv = int(v[e]) // FM
        if bu != bv:
            coupling[bu, bv] += k_edge[e]
            coupling[bv, bu] += k_edge[e]

    row_max = coupling.max(axis=1, keepdims=True)
    row_max = np.where(row_max > 0, row_max, 1.0)
    coupling_norm = coupling / row_max

    return coupling_norm


def _cluster_experts(
    coupling: np.ndarray,
    utilization: np.ndarray,
    threshold: float = 0.7,
) -> List[ExpertEquivalenceClass]:
    """Cluster experts into equivalence classes using Physarum coupling.

    Two experts are in the same class if their normalized coupling exceeds
    the threshold. The canonical representative is the most-utilized expert
    in the class (highest routing frequency = most important to preserve).
    """
    n = coupling.shape[0]
    visited = set()
    classes = []

    sorted_experts = np.argsort(-utilization)

    for seed in sorted_experts:
        seed = int(seed)
        if seed in visited:
            continue

        members = [seed]
        visited.add(seed)

        for other in range(n):
            if other in visited:
                continue
            if coupling[seed, other] >= threshold:
                members.append(other)
                visited.add(other)

        canonical = members[np.argmax([utilization[m] for m in members])]
        mean_c = float(np.mean([coupling[canonical, m] for m in members if m != canonical])) if len(members) > 1 else 0.0

        classes.append(ExpertEquivalenceClass(
            canonical=canonical,
            members=sorted(members),
            mean_coupling=mean_c,
        ))

    return classes


def calibrate_routing_patch(
    model,
    tokenizer,
    texts: Sequence[str],
    device: str = "cuda",
    max_tokens: int = 30_000,
    coupling_threshold: float = 0.7,
    margin_threshold: float = 0.1,
    model_id: str = "",
) -> RoutingPatch:
    """Calibrate a routing canonicalization patch using Physarum analysis.

    Uses batch calibration to hook ALL MoE layers simultaneously in a single
    forward sweep (instead of per-layer, which would be N_layers × slower).
    Builds Physarum coupling matrices and clusters experts into
    equivalence classes per layer.
    """
    from ._calibrate_moe import (
        collect_all_layers_covariance_and_utilization,
        _get_moe_module,
        _get_experts_and_gate,
    )

    first_moe = _get_moe_module(model, 0)
    experts, _ = _get_experts_and_gate(first_moe)
    n_experts = len(experts)
    top_k = getattr(first_moe, "num_experts_per_tok", None) or getattr(first_moe, "top_k", 2)

    _log.info(
        "calibrating routing patch: %d experts, top-%d, max_tokens=%d",
        n_experts, top_k, max_tokens,
    )

    all_layer_data = collect_all_layers_covariance_and_utilization(
        model, tokenizer, texts,
        max_len=256, device=device, max_tokens=max_tokens,
    )

    patch = RoutingPatch(
        model_id=model_id,
        n_experts_original=n_experts,
        top_k=top_k,
        coupling_threshold=coupling_threshold,
        margin_threshold=margin_threshold,
    )

    for li, data in sorted(all_layer_data.items()):
        cov_result = data["covariance"]
        util_result = data["utilization"]

        coupling = _build_expert_coupling_matrix(
            cov_result["D"],
            n_experts=cov_result["n_blocks"],
            feature_multiplier=cov_result["feature_multiplier"],
        )

        utilization = util_result["expert_frequency"].numpy()
        classes = _cluster_experts(coupling, utilization, threshold=coupling_threshold)

        non_singleton = [c for c in classes if len(c.members) > 1]
        patch.layers[li] = classes

        _log.info(
            "  layer %d: %d equivalence classes (%d non-singleton, covering %d/%d experts)",
            li, len(classes), len(non_singleton),
            sum(len(c.members) for c in non_singleton), n_experts,
        )

    total_non_singleton = sum(
        1 for ecs in patch.layers.values()
        for ec in ecs if len(ec.members) > 1
    )
    _log.info(
        "routing patch calibrated: %d total non-singleton classes across %d layers",
        total_non_singleton, len(patch.layers),
    )

    return patch


class CanonicalRouter(nn.Module):
    """Drop-in replacement for MoE router that canonicalizes expert selection.

    When the top-k routing scores for two experts in the same equivalence
    class are within margin_threshold of each other, snaps to the canonical
    representative. This makes routing deterministic for prefix caching
    without changing model quality.
    """

    def __init__(
        self,
        original_gate: nn.Module,
        equivalence_classes: List[ExpertEquivalenceClass],
        margin_threshold: float = 0.1,
    ):
        super().__init__()
        self.original_gate = original_gate
        self.margin_threshold = margin_threshold

        n_experts = sum(len(ec.members) for ec in equivalence_classes)
        self.register_buffer(
            "_canonical_map",
            torch.full((n_experts,), -1, dtype=torch.long),
        )
        for ec in equivalence_classes:
            for member in ec.members:
                self._canonical_map[member] = ec.canonical

    @property
    def weight(self):
        return self.original_gate.weight

    @property
    def bias(self):
        return getattr(self.original_gate, "bias", None)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.original_gate(hidden_states)
        return self._canonicalize_logits(logits)

    def _canonicalize_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Snap routing logits so equivalent experts get identical scores.

        For each equivalence class, find the max logit among members.
        If any member is within margin_threshold of the max, boost the
        canonical representative to be the max (ensuring it wins top-k).
        """
        canonical_map = self._canonical_map.to(logits.device)
        out = logits.clone()

        unique_canonicals = canonical_map.unique()
        for canon in unique_canonicals:
            members = (canonical_map == canon).nonzero(as_tuple=True)[0]
            if len(members) <= 1:
                continue

            member_logits = out[:, members]
            max_vals, _ = member_logits.max(dim=-1, keepdim=True)
            within_margin = (max_vals - member_logits) < self.margin_threshold

            canon_idx_in_members = (members == canon.item()).nonzero(as_tuple=True)[0]
            if len(canon_idx_in_members) == 0:
                continue
            ci = canon_idx_in_members[0]

            should_canonicalize = within_margin.any(dim=-1) & (within_margin.sum(dim=-1) > 1)
            if should_canonicalize.any():
                boost = max_vals.squeeze(-1) + 1e-4
                out[should_canonicalize, members[ci]] = boost[should_canonicalize]

        return out


def apply_routing_patch(model, patch: RoutingPatch) -> int:
    """Apply a routing canonicalization patch to a live model.

    Wraps each MoE layer's router with a CanonicalRouter that snaps
    routing decisions to equivalence class representatives.

    Returns the number of layers patched.
    """
    from ._calibrate_moe import _get_moe_module, _get_experts_and_gate

    patched = 0
    for li, classes in patch.layers.items():
        non_singleton = [c for c in classes if len(c.members) > 1]
        if not non_singleton:
            continue

        try:
            moe = _get_moe_module(model, li)
            _, gate = _get_experts_and_gate(moe)
        except (ValueError, AttributeError):
            _log.warning("layer %d: cannot locate MoE module, skipping", li)
            continue

        if isinstance(gate, CanonicalRouter):
            _log.debug("layer %d: already patched, skipping", li)
            continue

        canonical_router = CanonicalRouter(
            original_gate=gate,
            equivalence_classes=classes,
            margin_threshold=patch.margin_threshold,
        )

        if hasattr(moe, "gate"):
            moe.gate = canonical_router
        elif hasattr(moe, "router"):
            moe.router = canonical_router

        patched += 1
        _log.info(
            "layer %d: patched router (%d equivalence classes, %d canonical swaps possible)",
            li, len(non_singleton),
            sum(len(c.members) - 1 for c in non_singleton),
        )

    _log.info("routing patch applied: %d/%d layers patched", patched, len(patch.layers))
    return patched


def remove_routing_patch(model) -> int:
    """Remove routing canonicalization, restoring original routers."""
    from ._calibrate_moe import _get_moe_module, _get_experts_and_gate, _get_layers_module

    removed = 0
    n_layers = len(_get_layers_module(model))
    for li in range(n_layers):
        try:
            moe = _get_moe_module(model, li)
            _, gate = _get_experts_and_gate(moe)
        except (ValueError, AttributeError):
            continue

        if isinstance(gate, CanonicalRouter):
            original = gate.original_gate
            if hasattr(moe, "gate"):
                moe.gate = original
            elif hasattr(moe, "router"):
                moe.router = original
            removed += 1

    _log.info("routing patch removed from %d layers", removed)
    return removed


@torch.no_grad()
def bake_routing_patch(model, patch: RoutingPatch) -> int:
    """Bake canonicalization directly into router weights (vLLM-compatible).

    Instead of wrapping the router at runtime, this modifies the gate weight
    matrix in-place: for each equivalence class, all member experts' rows
    in gate.weight are replaced with the canonical expert's row.

    The router physically cannot distinguish between equivalent experts,
    so it deterministically picks the canonical one.

    After baking, save the model with model.save_pretrained() and load
    it in vLLM like any normal model — no runtime patches needed.

    Returns the number of layers modified.
    """
    from ._calibrate_moe import _get_moe_module, _get_experts_and_gate

    modified = 0
    total_swaps = 0

    for li, classes in patch.layers.items():
        non_singleton = [c for c in classes if len(c.members) > 1]
        if not non_singleton:
            continue

        try:
            moe = _get_moe_module(model, li)
            _, gate = _get_experts_and_gate(moe)
        except (ValueError, AttributeError):
            _log.warning("layer %d: cannot locate router, skipping", li)
            continue

        if not hasattr(gate, "weight"):
            _log.warning("layer %d: router has no weight matrix, skipping", li)
            continue

        W = gate.weight.data  # shape: [n_experts, hidden_size]
        n_experts = W.shape[0]
        layer_swaps = 0

        for ec in non_singleton:
            if ec.canonical >= n_experts:
                continue
            canonical_row = W[ec.canonical].clone()
            for member in ec.members:
                if member != ec.canonical and member < n_experts:
                    W[member].copy_(canonical_row)
                    layer_swaps += 1

        if gate.bias is not None:
            B = gate.bias.data
            for ec in non_singleton:
                if ec.canonical >= n_experts:
                    continue
                canonical_bias = B[ec.canonical].clone()
                for member in ec.members:
                    if member != ec.canonical and member < n_experts:
                        B[member].copy_(canonical_bias)

        modified += 1
        total_swaps += layer_swaps
        _log.info(
            "layer %d: baked %d expert rows to canonical representatives",
            li, layer_swaps,
        )

    _log.info(
        "routing patch baked: %d layers modified, %d total weight rows overwritten",
        modified, total_swaps,
    )
    return modified
