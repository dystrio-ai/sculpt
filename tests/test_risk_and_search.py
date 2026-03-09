"""Tests for structural risk scoring, Thompson Sampling search, and layer ordering."""

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
    protected_layers,
    risk_weighted_keep_schedule,
    DEFAULT_PROTECTION_THRESHOLD,
)
from dystrio_sculpt.search import (
    BetaArm, FrontierPoint, _is_safe, _assign_labels, _safety_reward,
    blended_speedup, SPEED_PROFILES,
)
from dystrio_sculpt.selectors.structural import (
    CrossLayerNoveltyTracker, DEFAULT_NOVELTY_LAMBDA,
)


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

    def test_three_points_ordered_by_keep_frac(self):
        pts = [
            _make_point(0.88, 1.07, 1.04),
            _make_point(0.82, 1.25, 1.09),
            _make_point(0.75, 1.47, 1.16),
        ]
        _assign_labels(pts, 2.0)
        labels = {pt.keep_frac: pt.label for pt in pts}
        assert "conservative" in labels[0.88]
        assert "balanced" in labels[0.82]
        assert "aggressive" in labels[0.75]

    def test_two_points_conservative_and_balanced(self):
        pts = [
            _make_point(0.88, 1.07, 1.04),
            _make_point(0.75, 1.47, 1.16),
        ]
        _assign_labels(pts, 2.0)
        labels = {pt.keep_frac: pt.label for pt in pts}
        assert "conservative" in labels[0.88]
        assert "balanced" in labels[0.75]

    def test_above_ceiling_gets_generic_label(self):
        pts = [
            _make_point(0.9, 1.1, 1.0),
            _make_point(0.7, 1.5, 1.3),
            _make_point(0.5, 2.5, 1.8),  # above 2.0 ceiling
        ]
        _assign_labels(pts, 2.0)
        above = [p for p in pts if p.keep_frac == 0.5][0]
        assert "point" in above.label
        assert "conservative" not in above.label
        assert "balanced" not in above.label
        assert "aggressive" not in above.label

    def test_label_indices_are_sequential(self):
        pts = [
            _make_point(0.9, 1.1, 1.0),
            _make_point(0.8, 1.3, 1.15),
            _make_point(0.7, 1.5, 1.3),
        ]
        _assign_labels(pts, 2.0)
        assert pts[0].label.startswith("frontier_0")
        assert pts[1].label.startswith("frontier_1")
        assert pts[2].label.startswith("frontier_2")


# ── BetaArm (Thompson Sampling) tests ────────────────────────────────────────


class TestBetaArm:
    def test_jeffreys_prior(self):
        arm = BetaArm()
        assert arm.a == 0.5
        assert arm.b == 0.5
        assert arm.n_obs == 0.0

    def test_sample_in_unit_interval(self):
        arm = BetaArm()
        rng = np.random.RandomState(42)
        for _ in range(100):
            s = arm.sample(rng)
            assert 0.0 <= s <= 1.0

    def test_deterministic_with_same_rng(self):
        arm = BetaArm(a=3.0, b=2.0)
        rng1 = np.random.RandomState(7)
        rng2 = np.random.RandomState(7)
        assert arm.sample(rng1) == arm.sample(rng2)

    def test_update_adds_mass(self):
        arm = BetaArm()
        arm.update(1.0)
        assert arm.a == 1.5
        assert arm.b == 0.5
        arm.update(0.0)
        assert arm.a == 1.5
        assert arm.b == 1.5

    def test_success_increases_mean(self):
        arm = BetaArm()
        before = arm.mean
        arm.update(1.0)
        assert arm.mean > before

    def test_failure_decreases_mean(self):
        arm = BetaArm()
        before = arm.mean
        arm.update(0.0)
        assert arm.mean < before

    def test_fractional_reward(self):
        arm = BetaArm(a=1.0, b=1.0)
        arm.update(0.6)
        assert arm.a == pytest.approx(1.6)
        assert arm.b == pytest.approx(1.4)

    def test_mean_converges_with_observations(self):
        arm = BetaArm()
        rng = np.random.RandomState(0)
        true_p = 0.7
        for _ in range(200):
            arm.update(1.0 if rng.random() < true_p else 0.0)
        assert abs(arm.mean - true_p) < 0.05

    def test_n_obs_tracks_updates(self):
        arm = BetaArm()
        for _ in range(10):
            arm.update(0.5)
        assert arm.n_obs == pytest.approx(10.0)


# ── Safety reward tests ──────────────────────────────────────────────────────


class TestSafetyReward:
    def test_failed_point_gives_zero(self):
        pt = _make_point(0.8, 1.0, 1.0, failed=True)
        assert _safety_reward(pt, 2.0) == 0.0

    def test_baseline_quality_gives_one(self):
        pt = _make_point(0.9, 1.0, 1.1)
        assert _safety_reward(pt, 2.0) == 1.0

    def test_below_baseline_gives_one(self):
        pt = _make_point(0.9, 0.95, 1.1)
        assert _safety_reward(pt, 2.0) == 1.0

    def test_at_upper_limit_gives_zero(self):
        ceiling = 2.0
        pt = _make_point(0.5, ceiling * 1.5, 2.0)
        assert _safety_reward(pt, ceiling) == 0.0

    def test_above_upper_limit_gives_zero(self):
        ceiling = 2.0
        pt = _make_point(0.4, ceiling * 2.0, 2.5)
        assert _safety_reward(pt, ceiling) == 0.0

    def test_monotonically_decreasing(self):
        ceiling = 2.0
        rewards = []
        for ratio in [1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]:
            pt = _make_point(0.7, ratio, 1.3)
            rewards.append(_safety_reward(pt, ceiling))
        for i in range(len(rewards) - 1):
            assert rewards[i] >= rewards[i + 1], (
                f"reward not monotonic at ratio step {i}: {rewards}"
            )

    def test_midpoint_reward(self):
        ceiling = 2.0
        upper = ceiling * 1.5  # 3.0
        mid_ratio = (1.0 + upper) / 2  # 2.0
        pt = _make_point(0.7, mid_ratio, 1.3)
        reward = _safety_reward(pt, ceiling)
        assert 0.3 < reward < 0.7, f"expected ~0.5 at midpoint, got {reward}"

    def test_reward_in_unit_interval(self):
        ceiling = 2.0
        for ratio in np.linspace(0.5, 5.0, 20):
            for failed in [True, False]:
                pt = _make_point(0.7, ratio, 1.3, failed=failed)
                r = _safety_reward(pt, ceiling)
                assert 0.0 <= r <= 1.0, f"reward={r} for ratio={ratio}, failed={failed}"


# ── Cross-layer novelty tracker tests ────────────────────────────────────────


class TestCrossLayerNoveltyTracker:
    def test_no_history_returns_none(self):
        tracker = CrossLayerNoveltyTracker()
        assert tracker.novelty_multiplier(10) is None

    def test_first_layer_no_novelty_applied(self):
        tracker = CrossLayerNoveltyTracker()
        mult = tracker.novelty_multiplier(8)
        assert mult is None

    def test_single_layer_recorded(self):
        tracker = CrossLayerNoveltyTracker(novelty_lambda=0.2)
        tracker.record([0, 1, 2], n_blocks=8)
        mult = tracker.novelty_multiplier(8)
        assert mult is not None
        assert mult.shape == (8,)

    def test_selected_blocks_get_lower_multiplier_no_adj(self):
        tracker = CrossLayerNoveltyTracker(novelty_lambda=0.2, conductance_diffusion=0.0)
        tracker.record([0, 1, 2], n_blocks=8)
        mult = tracker.novelty_multiplier(8)
        for b in [0, 1, 2]:
            assert mult[b] == pytest.approx(1.0)
        for b in [3, 4, 5, 6, 7]:
            assert mult[b] == pytest.approx(1.2)

    def test_all_blocks_selected_gives_uniform_multiplier(self):
        tracker = CrossLayerNoveltyTracker(novelty_lambda=0.15, conductance_diffusion=0.0)
        all_blocks = list(range(10))
        tracker.record(all_blocks, n_blocks=10)
        mult = tracker.novelty_multiplier(10)
        for b in range(10):
            assert mult[b] == pytest.approx(1.0)

    def test_frequency_scales_with_layers_no_adj(self):
        tracker = CrossLayerNoveltyTracker(novelty_lambda=0.2, conductance_diffusion=0.0)
        tracker.record([0, 1], n_blocks=4)
        tracker.record([0, 2], n_blocks=4)
        mult = tracker.novelty_multiplier(4)
        assert mult[0] == pytest.approx(1.0)
        assert mult[1] == pytest.approx(1.1)
        assert mult[2] == pytest.approx(1.1)
        assert mult[3] == pytest.approx(1.2)

    def test_multiplier_bounds(self):
        lam = 0.3
        tracker = CrossLayerNoveltyTracker(novelty_lambda=lam, conductance_diffusion=0.0)
        for _ in range(5):
            tracker.record([0], n_blocks=6)
        mult = tracker.novelty_multiplier(6)
        assert all(1.0 <= m <= 1.0 + lam for m in mult)

    def test_n_layers_property(self):
        tracker = CrossLayerNoveltyTracker()
        assert tracker.n_layers == 0
        tracker.record([0, 1], n_blocks=4)
        assert tracker.n_layers == 1
        tracker.record([2, 3], n_blocks=4)
        assert tracker.n_layers == 2

    def test_block_counts_returns_copy(self):
        tracker = CrossLayerNoveltyTracker()
        tracker.record([0, 1], n_blocks=4)
        counts = tracker.block_counts
        assert counts is not None
        counts[0] = 999.0
        assert tracker.block_counts[0] != 999.0

    def test_out_of_range_blocks_ignored(self):
        tracker = CrossLayerNoveltyTracker()
        tracker.record([0, 1, 100], n_blocks=4)
        counts = tracker.block_counts
        assert counts[0] == 1.0
        assert counts[1] == 1.0
        assert len(counts) == 4

    # ── Conductance-aware (hybrid) tests ──────────────────────────────

    def test_conductance_diffusion_penalizes_neighbours(self):
        """Block 1 is coupled to block 0 via conductance — selecting block 0
        frequently should also penalize block 1."""
        n = 4
        adj = np.zeros((n, n), dtype=np.float64)
        adj[0, 1] = 1.0
        adj[1, 0] = 1.0
        adj_norm = adj / (adj.max(axis=1, keepdims=True) + 1e-30)

        tracker = CrossLayerNoveltyTracker(novelty_lambda=0.2, conductance_diffusion=0.5)
        tracker.record([0], n_blocks=n, block_adj=adj_norm)
        mult = tracker.novelty_multiplier(n)

        # Block 0 was selected → lowest multiplier
        # Block 1 is coupled to block 0 → also penalized (lower than blocks 2,3)
        assert mult[1] < mult[2], "coupled neighbour should have lower multiplier"
        assert mult[1] < mult[3], "coupled neighbour should have lower multiplier"

    def test_no_conductance_falls_back_to_frequency(self):
        """Without adjacency data, behaviour matches plain frequency counting."""
        tracker_plain = CrossLayerNoveltyTracker(novelty_lambda=0.2, conductance_diffusion=0.0)
        tracker_cond = CrossLayerNoveltyTracker(novelty_lambda=0.2, conductance_diffusion=0.5)

        tracker_plain.record([0, 1], n_blocks=4)
        tracker_cond.record([0, 1], n_blocks=4)

        m_plain = tracker_plain.novelty_multiplier(4)
        m_cond = tracker_cond.novelty_multiplier(4)
        np.testing.assert_array_almost_equal(m_plain, m_cond)

    def test_diffusion_zero_matches_pure_frequency(self):
        """With diffusion=0, conductance adjacency has no effect."""
        n = 4
        adj = np.ones((n, n), dtype=np.float64)
        np.fill_diagonal(adj, 0.0)
        adj_norm = adj / (adj.max(axis=1, keepdims=True) + 1e-30)

        tracker = CrossLayerNoveltyTracker(novelty_lambda=0.2, conductance_diffusion=0.0)
        tracker.record([0], n_blocks=n, block_adj=adj_norm)
        mult = tracker.novelty_multiplier(n)

        # Block 0 selected, blocks 1-3 not — pure frequency
        assert mult[0] == pytest.approx(1.0)
        assert mult[1] == pytest.approx(1.2)

    def test_high_diffusion_spreads_penalty_widely(self):
        """With high diffusion and full coupling, all blocks should be penalized
        when one is frequently selected."""
        n = 4
        adj = np.ones((n, n), dtype=np.float64)
        np.fill_diagonal(adj, 0.0)
        adj_norm = adj / (adj.max(axis=1, keepdims=True) + 1e-30)

        tracker = CrossLayerNoveltyTracker(novelty_lambda=0.3, conductance_diffusion=1.0)
        for _ in range(5):
            tracker.record([0], n_blocks=n, block_adj=adj_norm)
        mult = tracker.novelty_multiplier(n)

        # With full connectivity, block 0's penalty diffuses perfectly to
        # all neighbours — everyone gets equally penalized.
        assert mult[1] < 1.0 + 0.3, "diffusion should reduce neighbour novelty"
        assert mult[0] == pytest.approx(mult[1]), (
            "full connectivity + full diffusion → uniform penalty"
        )

    def test_multiplier_always_in_bounds_with_conductance(self):
        """Multipliers stay in [1.0, 1.0 + lambda] even with conductance."""
        lam = 0.25
        n = 6
        rng = np.random.RandomState(42)
        adj = rng.rand(n, n)
        adj = (adj + adj.T) / 2
        np.fill_diagonal(adj, 0.0)
        adj_norm = adj / (adj.max(axis=1, keepdims=True) + 1e-30)

        tracker = CrossLayerNoveltyTracker(novelty_lambda=lam, conductance_diffusion=0.5)
        for _ in range(8):
            blocks = rng.choice(n, size=3, replace=False).tolist()
            tracker.record(blocks, n_blocks=n, block_adj=adj_norm)
        mult = tracker.novelty_multiplier(n)
        assert all(1.0 <= m <= 1.0 + lam + 1e-9 for m in mult), (
            f"multiplier out of bounds: {mult}"
        )


# ── Layer protection tests ───────────────────────────────────────────────────


def _make_prescan_cache(n_layers: int = 8, dim: int = 16) -> Dict[int, Dict[str, Any]]:
    """Build a synthetic prescan cache with increasing risk by layer index."""
    cache: Dict[int, Dict[str, Any]] = {}
    for li in range(n_layers):
        scale = (li + 1) / n_layers
        bs = torch.full((dim,), scale, dtype=torch.float64)
        D = torch.eye(dim, dtype=torch.float64) * (1.0 + scale)
        cache[li] = {"block_sensitivity": bs, "D": D, "block_energy": bs * 0.5}
    return cache


class TestProtectedLayers:
    def test_no_layers_protected_low_threshold(self):
        cache = _make_prescan_cache(4)
        result = protected_layers(cache, threshold=0.99)
        assert result == []

    def test_high_risk_layers_protected(self):
        cache = _make_prescan_cache(8)
        result = protected_layers(cache, threshold=0.30)
        assert len(result) > 0
        for li in result:
            pre = cache[li]
            risk, _ = layer_risk_score(pre["block_sensitivity"], pre["D"])
            assert risk >= 0.30

    def test_all_layers_protected_at_zero_threshold(self):
        cache = _make_prescan_cache(4)
        result = protected_layers(cache, threshold=0.0)
        assert len(result) == 4

    def test_returns_sorted(self):
        cache = _make_prescan_cache(8)
        result = protected_layers(cache, threshold=0.3)
        assert result == sorted(result)


class TestRiskWeightedKeepSchedule:
    def test_returns_all_layers(self):
        cache = _make_prescan_cache(8)
        schedule = risk_weighted_keep_schedule(cache, aggressiveness=0.5)
        assert len(schedule) == 8

    def test_high_risk_layers_get_higher_keep(self):
        cache = _make_prescan_cache(8)
        schedule = risk_weighted_keep_schedule(cache, aggressiveness=0.5)
        assert schedule[0] <= schedule[7]

    def test_zero_aggressiveness_all_ceiling(self):
        cache = _make_prescan_cache(4)
        schedule = risk_weighted_keep_schedule(cache, aggressiveness=0.0)
        for kf in schedule.values():
            assert kf == pytest.approx(1.0)

    def test_protected_layers_always_1(self):
        cache = _make_prescan_cache(8)
        schedule = risk_weighted_keep_schedule(
            cache, aggressiveness=1.0, protection_threshold=0.5,
        )
        prot = protected_layers(cache, threshold=0.5)
        for li in prot:
            assert schedule[li] == 1.0

    def test_keep_frac_bounds(self):
        cache = _make_prescan_cache(8)
        schedule = risk_weighted_keep_schedule(
            cache, aggressiveness=0.8, floor=0.30, ceiling=1.0,
        )
        for kf in schedule.values():
            assert 0.30 <= kf <= 1.0

    def test_whole_model_mean_near_target(self):
        cache = _make_prescan_cache(16)
        for target_kf in [0.85, 0.70, 0.55]:
            agg = 1.0 - target_kf
            schedule = risk_weighted_keep_schedule(
                cache, aggressiveness=agg, protection_threshold=0.90,
            )
            whole_model_mean = sum(schedule.values()) / len(schedule)
            assert abs(whole_model_mean - target_kf) < 0.10, (
                f"target={target_kf} but whole-model mean={whole_model_mean:.3f}"
            )


# ── Speed profile tests ──────────────────────────────────────────────────────


class TestBlendedSpeedup:
    def test_equal_weights_averages(self):
        result = blended_speedup(1.2, 1.0, 0.5, 0.5)
        assert result == pytest.approx(1.1)

    def test_prefill_heavy(self):
        result = blended_speedup(1.5, 1.0, 0.8, 0.2)
        assert result == pytest.approx(1.4)

    def test_decode_heavy(self):
        result = blended_speedup(1.5, 1.0, 0.2, 0.8)
        assert result == pytest.approx(1.1)

    def test_all_profiles_exist(self):
        expected = {"balanced", "prefill_heavy", "decode_heavy", "rag", "chatbot", "throughput", "latency"}
        assert expected.issubset(set(SPEED_PROFILES.keys()))

    def test_profiles_weights_sum_to_one(self):
        for name, profile in SPEED_PROFILES.items():
            total = profile["prefill_weight"] + profile["decode_weight"]
            assert total == pytest.approx(1.0), f"profile {name} weights sum to {total}"

    def test_zero_weights_falls_back(self):
        result = blended_speedup(1.5, 1.0, 0.0, 0.0)
        assert result == 1.5
