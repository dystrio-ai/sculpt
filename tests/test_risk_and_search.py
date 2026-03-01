"""Tests for structural risk scoring, SBS ceiling enforcement, and layer ordering."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from dystrio_sculpt.risk import (
    layer_risk_score,
    model_risk_score,
    layer_compressibility_order,
    risk_aware_keep_candidates,
)
from dystrio_sculpt.search import FrontierPoint, _is_safe, _assign_labels


# ── Risk score tests ──────────────────────────────────────────────────────────


def _make_prescan_entry(n_blocks: int = 10, sens_mean: float = 0.5) -> Dict[str, Any]:
    """Build a synthetic prescan cache entry."""
    F = 3
    dim = F * n_blocks
    D = torch.eye(dim, dtype=torch.float64) + 0.1 * torch.randn(dim, dim, dtype=torch.float64)
    D = (D + D.T) / 2  # symmetric
    bs = torch.full((n_blocks,), sens_mean, dtype=torch.float64)
    be = torch.ones(n_blocks, dtype=torch.float64)
    return {
        "D": D,
        "block_sensitivity": bs,
        "block_energy": be,
        "feature_multiplier": F,
    }


class TestRiskScoreBounds:
    def test_risk_in_unit_interval(self):
        for sens in [0.01, 0.3, 0.5, 0.8, 2.0, 10.0]:
            entry = _make_prescan_entry(sens_mean=sens)
            risk, detail = layer_risk_score(entry["block_sensitivity"], entry["D"], entry["block_energy"])
            assert 0.0 <= risk <= 1.0, f"risk={risk} for sens={sens}"

    def test_risk_stable_for_fixed_input(self):
        entry = _make_prescan_entry(n_blocks=8, sens_mean=0.4)
        r1, _ = layer_risk_score(entry["block_sensitivity"], entry["D"], entry["block_energy"])
        r2, _ = layer_risk_score(entry["block_sensitivity"], entry["D"], entry["block_energy"])
        assert r1 == r2

    def test_higher_sensitivity_higher_risk(self):
        low_entry = _make_prescan_entry(sens_mean=0.1)
        high_entry = _make_prescan_entry(sens_mean=5.0)
        r_low, _ = layer_risk_score(low_entry["block_sensitivity"], low_entry["D"])
        r_high, _ = layer_risk_score(high_entry["block_sensitivity"], high_entry["D"])
        assert r_high > r_low

    def test_model_risk_aggregates(self):
        cache = {
            0: _make_prescan_entry(sens_mean=0.2),
            1: _make_prescan_entry(sens_mean=0.5),
            2: _make_prescan_entry(sens_mean=1.0),
        }
        agg, detail = model_risk_score(cache)
        assert 0.0 <= agg <= 1.0
        assert "aggregate" in detail

    def test_empty_prescan_returns_neutral(self):
        agg, _ = model_risk_score({})
        assert agg == 0.5

    def test_missing_signals_returns_neutral(self):
        cache = {0: {"D": None, "block_sensitivity": None}}
        agg, _ = model_risk_score(cache)
        assert agg == 0.5


# ── Layer ordering tests ─────────────────────────────────────────────────────


class TestLayerOrdering:
    def test_sorted_by_increasing_risk(self):
        cache = {
            0: _make_prescan_entry(sens_mean=2.0),  # high risk
            1: _make_prescan_entry(sens_mean=0.1),  # low risk
            2: _make_prescan_entry(sens_mean=0.5),  # medium risk
        }
        order = layer_compressibility_order(cache)
        assert order[0] == 1, "safest (lowest risk) layer should be first"
        assert order[-1] == 0, "riskiest layer should be last"

    def test_all_layers_present(self):
        cache = {i: _make_prescan_entry() for i in range(5)}
        order = layer_compressibility_order(cache)
        assert set(order) == set(range(5))


# ── Risk-aware candidates tests ──────────────────────────────────────────────


class TestRiskAwareCandidates:
    def test_low_risk_includes_aggressive(self):
        cands = risk_aware_keep_candidates(0.2)
        assert min(cands) <= 0.60

    def test_high_risk_stays_conservative(self):
        cands = risk_aware_keep_candidates(0.8)
        assert min(cands) >= 0.70

    def test_candidates_sorted_possible(self):
        for risk in [0.1, 0.5, 0.9]:
            cands = risk_aware_keep_candidates(risk)
            assert len(cands) >= 4
            assert all(0.3 <= c <= 1.0 for c in cands)


# ── Search label / ceiling tests ─────────────────────────────────────────────


def _make_point(kf: float, ppl_ratio: float, speedup: float, failed: bool = False) -> FrontierPoint:
    return FrontierPoint(
        keep_frac=kf, ppl_w103=ppl_ratio * 10.0, ppl_w2=ppl_ratio * 10.0,
        prefill_tps=speedup * 100.0, decode_tps=speedup * 50.0,
        prefill_speedup=speedup, decode_speedup=speedup * 0.5,
        wall_time_s=60.0, ppl_ratio=ppl_ratio, failed=failed,
    )


class TestCeilingEnforcement:
    def test_is_safe_respects_ceiling(self):
        pt_ok = _make_point(0.8, 1.5, 1.2)
        pt_over = _make_point(0.6, 2.5, 1.8)
        assert _is_safe(pt_ok, 2.0)
        assert not _is_safe(pt_over, 2.0)

    def test_failed_point_never_safe(self):
        pt = _make_point(0.9, 1.0, 1.0, failed=True)
        assert not _is_safe(pt, 10.0)

    def test_assign_labels_no_balanced_above_ceiling(self):
        ceiling = 2.0
        pts = [
            _make_point(0.9, 1.2, 1.1),
            _make_point(0.7, 1.8, 1.5),
            _make_point(0.5, 3.0, 2.0),  # above ceiling
        ]
        _assign_labels(pts, ceiling)
        for pt in pts:
            if "balanced" in pt.label:
                assert pt.ppl_ratio <= ceiling, (
                    f"balanced label on point with ppl_ratio={pt.ppl_ratio} > ceiling={ceiling}"
                )

    def test_single_point_labeled(self):
        pts = [_make_point(0.8, 1.5, 1.2)]
        _assign_labels(pts, 2.0)
        assert pts[0].label.startswith("frontier_0")

    def test_all_points_get_labels(self):
        pts = [_make_point(0.9, 1.1, 1.0), _make_point(0.7, 1.5, 1.3)]
        _assign_labels(pts, 2.0)
        for pt in pts:
            assert pt.label, "every point must have a label"
