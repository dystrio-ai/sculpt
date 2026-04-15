"""Tests for the cohesion-based Physarum selector and improved solver."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from dystrio_sculpt.selectors.structural import (
    build_graph_from_cov,
    physarum_conductance,
    physarum_conductance_v2,
    select_blocks_cohesion,
    select_blocks_structural,
)


def _make_block_cov(n_blocks: int, F: int = 3, coupled_pairs=None):
    """Build a synthetic block covariance with optional strong coupling."""
    dim = F * n_blocks
    D = torch.eye(dim, dtype=torch.float64) * 5.0
    D += 0.05 * torch.randn(dim, dim, dtype=torch.float64)
    D = (D + D.T) / 2

    for a, b in (coupled_pairs or []):
        for fi in range(F):
            for fj in range(F):
                D[a * F + fi, b * F + fj] += 3.0
                D[b * F + fj, a * F + fi] += 3.0

    return D


class TestPhysarumV2Solver:
    """Verify the improved solver produces valid conductances."""

    def test_output_shape_matches_edges(self):
        D = _make_block_cov(8)
        u, v, w = build_graph_from_cov(D, k=5)
        k = physarum_conductance_v2(u, v, w, D.shape[0], n_iters=50)
        assert k.shape == w.shape

    def test_conductances_are_positive(self):
        D = _make_block_cov(8)
        u, v, w = build_graph_from_cov(D, k=5)
        k = physarum_conductance_v2(u, v, w, D.shape[0], n_iters=50)
        assert np.all(k > 0)

    def test_deterministic_with_seed(self):
        D = _make_block_cov(8)
        u, v, w = build_graph_from_cov(D, k=5)
        rng1 = np.random.RandomState(42)
        rng2 = np.random.RandomState(42)
        k1 = physarum_conductance_v2(u, v, w, D.shape[0], n_iters=50, rng=rng1)
        k2 = physarum_conductance_v2(u, v, w, D.shape[0], n_iters=50, rng=rng2)
        np.testing.assert_array_equal(k1, k2)

    def test_empty_graph(self):
        u = np.array([], dtype=np.int64)
        v = np.array([], dtype=np.int64)
        w = np.array([], dtype=np.float64)
        k = physarum_conductance_v2(u, v, w, 10, n_iters=50)
        assert len(k) == 0

    def test_mu_affects_concentration(self):
        """Higher mu should produce more concentrated conductances."""
        D = _make_block_cov(12, coupled_pairs=[(0, 1), (2, 3)])
        u, v, w = build_graph_from_cov(D, k=5)
        rng1 = np.random.RandomState(7)
        rng2 = np.random.RandomState(7)
        k_linear = physarum_conductance_v2(u, v, w, D.shape[0], n_iters=100, mu=1.0, rng=rng1)
        k_super = physarum_conductance_v2(u, v, w, D.shape[0], n_iters=100, mu=2.0, rng=rng2)
        cv_linear = k_linear.std() / (k_linear.mean() + 1e-30)
        cv_super = k_super.std() / (k_super.mean() + 1e-30)
        assert cv_super > cv_linear, "mu=2.0 should concentrate conductance more than mu=1.0"


class TestCohesionSelector:
    """Verify the cohesion selector's basic contract and coupling behavior."""

    def test_returns_correct_count(self):
        D = _make_block_cov(10)
        sens = torch.rand(10, dtype=torch.float64)
        kept, idx, arts = select_blocks_cohesion(
            D, keep_frac=0.5, block_size=128,
            block_sensitivity=sens, feature_multiplier=3,
        )
        assert len(kept) == 5

    def test_artifacts_have_expected_keys(self):
        D = _make_block_cov(8)
        sens = torch.rand(8, dtype=torch.float64)
        kept, idx, arts = select_blocks_cohesion(
            D, keep_frac=0.5, block_size=1,
            block_sensitivity=sens, feature_multiplier=3,
        )
        assert "block_scores" in arts
        assert "block_adj_norm" in arts
        assert "k_edge" in arts
        assert arts["method"] == "cohesion"

    def test_deterministic_with_seed(self):
        D = _make_block_cov(10)
        sens = torch.rand(10, dtype=torch.float64)
        rng1 = np.random.RandomState(99)
        rng2 = np.random.RandomState(99)
        kept1, _, _ = select_blocks_cohesion(
            D, keep_frac=0.5, block_size=128,
            block_sensitivity=sens, feature_multiplier=3, rng=rng1,
        )
        kept2, _, _ = select_blocks_cohesion(
            D, keep_frac=0.5, block_size=128,
            block_sensitivity=sens, feature_multiplier=3, rng=rng2,
        )
        assert kept1 == kept2

    def test_cohesion_keeps_coupled_blocks_together(self):
        """When two blocks are strongly coupled and one is important, the
        cohesion selector should keep both; the old diversity selector should
        be more likely to separate them."""
        n_blocks = 10
        F = 3
        D = _make_block_cov(n_blocks, F=F, coupled_pairs=[(0, 1)])

        sens = torch.zeros(n_blocks, dtype=torch.float64)
        sens[0] = 1.0  # block 0 is very important
        sens[1] = 0.6  # block 1 is moderately important, coupled to 0
        for i in range(2, n_blocks):
            sens[i] = 0.7  # other blocks: individually more important than 1

        rng_c = np.random.RandomState(42)
        kept_cohesion, _, _ = select_blocks_cohesion(
            D, keep_frac=0.5, block_size=1,
            block_sensitivity=sens, feature_multiplier=F,
            rng=rng_c, cohesion_lambda=0.3,
        )

        rng_d = np.random.RandomState(42)
        kept_diversity, _, _ = select_blocks_structural(
            D, keep_frac=0.5, block_size=1,
            block_sensitivity=sens, feature_multiplier=F,
            rng=rng_d, diversity_lambda=0.3,
        )

        # Cohesion should keep block 1 (coupled to the important block 0).
        # Diversity should penalize block 1 for being coupled to 0.
        assert 0 in kept_cohesion and 1 in kept_cohesion, (
            f"Cohesion should keep coupled pair (0,1), got {kept_cohesion}"
        )
        # The diversity selector may or may not drop block 1, but with
        # diversity_lambda=0.3 and strong coupling, it should penalize block 1
        # enough that some other block beats it.
        assert 0 in kept_diversity, "Both should keep the most important block"

    def test_works_with_expert_level_block_size_1(self):
        """MoE-style usage: block_size=1, each block is one expert."""
        n_experts = 8
        F = 3
        D = _make_block_cov(n_experts, F=F)
        kept, idx, arts = select_blocks_cohesion(
            D, keep_frac=0.5, block_size=1,
            feature_multiplier=F,
        )
        assert len(kept) == 4
        assert arts["block_scores"].shape[0] == n_experts

    def test_cross_layer_novelty_modulation(self):
        """Cross-layer novelty should modulate base scores."""
        D = _make_block_cov(8)
        sens = torch.ones(8, dtype=torch.float64)
        novelty = np.ones(8, dtype=np.float64)
        novelty[0] = 0.01  # heavily penalize block 0

        rng = np.random.RandomState(42)
        kept, _, _ = select_blocks_cohesion(
            D, keep_frac=0.5, block_size=1,
            block_sensitivity=sens, feature_multiplier=3,
            rng=rng, cross_layer_novelty=novelty,
        )
        assert 0 not in kept, "Block 0 should be rejected by novelty penalty"

    def test_cohesion_lambda_zero_matches_sensitivity_order(self):
        """With cohesion_lambda=0, selection should match plain sensitivity."""
        D = _make_block_cov(8)
        sens = torch.tensor([0.1, 0.9, 0.3, 0.8, 0.2, 0.7, 0.4, 0.6],
                            dtype=torch.float64)
        rng = np.random.RandomState(42)
        kept, _, _ = select_blocks_cohesion(
            D, keep_frac=0.5, block_size=1,
            block_sensitivity=sens, feature_multiplier=3,
            rng=rng, cohesion_lambda=0.0,
        )
        # Top 4 by sensitivity: indices 1(0.9), 3(0.8), 5(0.7), 7(0.6)
        assert sorted(kept) == [1, 3, 5, 7]
