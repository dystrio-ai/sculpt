"""Structural risk scoring from prescan artifacts.

Computes a scalar risk_score in [0, 1] per layer and a model-level aggregate.
Higher risk => harder to compress safely at a given keep_frac.

Signals used (all from prescan cache):
  - block_sensitivity: operator fidelity (how much zeroing a block hurts)
  - D: block covariance matrix (coupling geometry)
  - block_energy: activation magnitude per block
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _effective_rank_ratio(D: torch.Tensor) -> float:
    """Ratio of effective rank (at 95% variance) to full dimension.

    Low ratio => concentrated spectrum => low geometric redundancy => higher risk.
    """
    try:
        D_np = D.numpy().astype(np.float64)
        eigenvalues = np.linalg.eigvalsh(D_np)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        eigenvalues = np.sort(eigenvalues)[::-1]
        total = eigenvalues.sum()
        if total < 1e-30:
            return 0.5
        cumsum = np.cumsum(eigenvalues)
        rank95 = int(np.searchsorted(cumsum, total * 0.95)) + 1
        return rank95 / max(len(eigenvalues), 1)
    except Exception:
        return 0.5


def _top10_edge_mass(D: torch.Tensor) -> float:
    """Fraction of total off-diagonal correlation held by the top 10% of edges.

    High concentration => tightly coupled blocks => riskier to prune.
    """
    try:
        D_np = D.numpy().astype(np.float64)
        n = D_np.shape[0]
        std = np.sqrt(np.diag(D_np).clip(1e-30))
        corr = np.abs(D_np / np.outer(std, std))
        np.fill_diagonal(corr, 0.0)
        vals = corr[np.triu_indices(n, k=1)]
        if len(vals) == 0:
            return 0.5
        vals_sorted = np.sort(vals)[::-1]
        total = vals_sorted.sum()
        if total < 1e-30:
            return 0.0
        top_k = max(1, len(vals_sorted) // 10)
        return float(vals_sorted[:top_k].sum() / total)
    except Exception:
        return 0.5


def layer_risk_score(
    block_sensitivity: torch.Tensor,
    D: torch.Tensor,
    block_energy: Optional[torch.Tensor] = None,
) -> Tuple[float, Dict[str, float]]:
    """Compute structural risk score for a single layer.

    Returns (risk_score, detail_dict).
    """
    sens = block_sensitivity.numpy().astype(np.float64)
    mean_sens = float(sens.mean())
    max_sens = float(sens.max())

    sensitivity_term = _sigmoid((mean_sens - 0.5) / 0.3)

    top10 = _top10_edge_mass(D)
    coupling_term = _clamp01((top10 - 0.3) / 0.4)

    rank_ratio = _effective_rank_ratio(D)
    rank_term = 1.0 - _clamp01((rank_ratio - 0.1) / 0.5)

    risk = _clamp01(0.45 * sensitivity_term + 0.35 * coupling_term + 0.20 * rank_term)

    detail = {
        "mean_block_sensitivity": round(mean_sens, 6),
        "max_block_sensitivity": round(max_sens, 6),
        "sensitivity_term": round(sensitivity_term, 4),
        "top10_edge_mass": round(top10, 4),
        "coupling_term": round(coupling_term, 4),
        "eff_rank_ratio": round(rank_ratio, 4),
        "rank_term": round(rank_term, 4),
        "risk_score": round(risk, 4),
    }
    return risk, detail


def model_risk_score(
    prescan_cache: Dict[int, Dict[str, Any]],
) -> Tuple[float, Dict[str, Any]]:
    """Aggregate risk score across all layers from prescan cache.

    Returns (aggregate_risk, {layer_idx: detail, ..., "aggregate": float}).
    """
    if not prescan_cache:
        return 0.5, {"aggregate": 0.5, "source": "no_prescan"}

    layer_risks: List[float] = []
    per_layer: Dict[str, Any] = {}

    for li in sorted(prescan_cache.keys()):
        pre = prescan_cache[li]
        bs = pre.get("block_sensitivity")
        D = pre.get("D")
        be = pre.get("block_energy")

        if bs is None or D is None:
            layer_risks.append(0.5)
            per_layer[str(li)] = {"risk_score": 0.5, "source": "missing_signals"}
            continue

        risk, detail = layer_risk_score(bs, D, be)
        layer_risks.append(risk)
        per_layer[str(li)] = detail

    aggregate = float(np.mean(layer_risks))
    per_layer["aggregate"] = round(aggregate, 4)
    return aggregate, per_layer


def layer_compressibility_order(
    prescan_cache: Dict[int, Dict[str, Any]],
) -> List[int]:
    """Return layer indices sorted by increasing risk (safest to compress first)."""
    if not prescan_cache:
        return sorted(prescan_cache.keys())

    scored: List[Tuple[float, int]] = []
    for li in sorted(prescan_cache.keys()):
        pre = prescan_cache[li]
        bs = pre.get("block_sensitivity")
        D = pre.get("D")
        if bs is None or D is None:
            scored.append((0.5, li))
            continue
        risk, _ = layer_risk_score(bs, D, pre.get("block_energy"))
        scored.append((risk, li))

    scored.sort(key=lambda t: t[0])
    return [li for _, li in scored]


def risk_aware_keep_candidates(risk: float) -> List[float]:
    """Return initial keep_frac candidates adapted to structural risk."""
    if risk <= 0.35:
        return [0.85, 0.78, 0.70, 0.62, 0.55, 0.48, 0.42]
    if risk >= 0.65:
        return [0.92, 0.88, 0.84, 0.80, 0.76, 0.72, 0.66, 0.60]
    return [0.88, 0.82, 0.75, 0.70, 0.66, 0.60]


# ── Layer protection ──────────────────────────────────────────────────────────


DEFAULT_PROTECTION_THRESHOLD = 0.70


def protected_layers(
    prescan_cache: Dict[int, Dict[str, Any]],
    threshold: float = DEFAULT_PROTECTION_THRESHOLD,
) -> List[int]:
    """Return layer indices whose risk score exceeds *threshold*.

    Protected layers should be skipped during compression (keep_frac=1.0)
    because their structural risk is too high for safe pruning.  This
    corresponds to the production practice of leaving final/deep blocks
    untouched to preserve semantic integrity.
    """
    result: List[int] = []
    for li in sorted(prescan_cache.keys()):
        pre = prescan_cache[li]
        bs = pre.get("block_sensitivity")
        D = pre.get("D")
        if bs is None or D is None:
            continue
        risk, _ = layer_risk_score(bs, D, pre.get("block_energy"))
        if risk >= threshold:
            result.append(li)
    return result


# ── Risk-weighted keep_frac schedule ──────────────────────────────────────────


def risk_weighted_keep_schedule(
    prescan_cache: Dict[int, Dict[str, Any]],
    aggressiveness: float,
    floor: float = 0.30,
    ceiling: float = 1.0,
    protection_threshold: float = DEFAULT_PROTECTION_THRESHOLD,
) -> Dict[int, float]:
    """Derive per-layer keep_frac from risk scores and a single scalar.

    Ensures **total weight reduction** matches what uniform compression at
    the same keep_frac would produce.  Protected layers (risk above
    threshold) stay at 1.0, and non-protected layers compensate so the
    whole-model average equals the target.  Within non-protected layers,
    keep_frac is distributed proportionally to risk: safe layers are
    compressed harder, risky layers lighter.
    """
    target_kf = max(floor, 1.0 - aggressiveness)

    if aggressiveness <= 0:
        return {li: ceiling for li in sorted(prescan_cache.keys())}

    # Phase 1: identify protected vs compressible layers
    layers_sorted = sorted(prescan_cache.keys())
    risks: Dict[int, float] = {}
    protected: List[int] = []

    for li in layers_sorted:
        pre = prescan_cache[li]
        bs = pre.get("block_sensitivity")
        D = pre.get("D")
        if bs is None or D is None:
            protected.append(li)
            continue
        risk, _ = layer_risk_score(bs, D, pre.get("block_energy"))
        if risk >= protection_threshold:
            protected.append(li)
        else:
            risks[li] = risk

    if not risks:
        return {li: ceiling for li in layers_sorted}

    n_total = len(layers_sorted)
    n_protected = len(protected)
    n_compressible = n_total - n_protected

    # Phase 2: compute the compensated target for non-protected layers.
    # Protected layers contribute ceiling (1.0) to the whole-model average,
    # so compressible layers must absorb the full compression budget.
    compensated_target = (target_kf * n_total - ceiling * n_protected) / n_compressible
    compensated_target = max(floor, min(ceiling, compensated_target))

    # Phase 3: distribute compensated target across non-protected layers
    # proportional to risk.  Higher risk => higher keep_frac.
    risk_vals = np.array([risks[li] for li in sorted(risks.keys())])
    risk_min, risk_max = float(risk_vals.min()), float(risk_vals.max())
    risk_range = max(risk_max - risk_min, 1e-9)

    normed = {li: (risks[li] - risk_min) / risk_range for li in risks}

    span = ceiling - floor
    raw = {li: floor + normed[li] * span for li in risks}
    raw_mean = sum(raw.values()) / len(raw)
    shift = compensated_target - raw_mean

    schedule: Dict[int, float] = {}
    for li in layers_sorted:
        if li in protected or li not in risks:
            schedule[li] = ceiling
        else:
            keep = raw[li] + shift
            schedule[li] = round(max(floor, min(ceiling, keep)), 4)

    return schedule
