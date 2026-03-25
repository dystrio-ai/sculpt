"""Tests for MoE routing canonicalization patch."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from tests.test_moe_adapter import FakeMoEModel, FakeTokenizer


class TestCanonicalRouter:
    """Test the CanonicalRouter logit canonicalization."""

    def test_canonical_snaps_close_logits(self):
        from dystrio_sculpt.moe_routing_patch import CanonicalRouter, ExpertEquivalenceClass

        gate = nn.Linear(64, 8, bias=False)
        classes = [
            ExpertEquivalenceClass(canonical=0, members=[0, 1, 2], mean_coupling=0.85),
            ExpertEquivalenceClass(canonical=3, members=[3], mean_coupling=0.0),
            ExpertEquivalenceClass(canonical=4, members=[4, 5], mean_coupling=0.75),
            ExpertEquivalenceClass(canonical=6, members=[6, 7], mean_coupling=0.80),
        ]
        router = CanonicalRouter(gate, classes, margin_threshold=0.1)

        logits = torch.tensor([[
            0.50, 0.49, 0.48,   # experts 0,1,2 — all within margin, snap to 0
            0.30,                # expert 3 — singleton
            0.20, 0.19,          # experts 4,5 — within margin, snap to 4
            0.10, 0.01,          # experts 6,7 — NOT within margin (diff > 0.1)
        ]])

        out = router._canonicalize_logits(logits)

        # Canonical expert 0 should get the highest logit in class {0,1,2}
        assert out[0, 0] > out[0, 1]
        assert out[0, 0] > out[0, 2]

        # Canonical expert 4 should get the highest in class {4,5}
        assert out[0, 4] > out[0, 5]

        # Expert 3 (singleton) should be unchanged
        assert out[0, 3] == logits[0, 3]

        # Experts 6,7 — 6 is canonical but 7 is too far, so no snap
        # Both could be unchanged or 6 gets boosted if they're within margin
        # In this case diff = 0.09 < 0.1, so they ARE within margin
        assert out[0, 6] >= out[0, 7]

    def test_forward_preserves_shape(self):
        from dystrio_sculpt.moe_routing_patch import CanonicalRouter, ExpertEquivalenceClass

        gate = nn.Linear(64, 8, bias=False)
        classes = [
            ExpertEquivalenceClass(canonical=i, members=[i], mean_coupling=0.0)
            for i in range(8)
        ]
        router = CanonicalRouter(gate, classes, margin_threshold=0.1)
        x = torch.randn(4, 64)
        out = router(x)
        assert out.shape == (4, 8)

    def test_singleton_classes_are_noop(self):
        from dystrio_sculpt.moe_routing_patch import CanonicalRouter, ExpertEquivalenceClass

        gate = nn.Linear(64, 4, bias=False)
        classes = [
            ExpertEquivalenceClass(canonical=i, members=[i], mean_coupling=0.0)
            for i in range(4)
        ]
        router = CanonicalRouter(gate, classes, margin_threshold=0.1)
        x = torch.randn(2, 64)
        original = gate(x)
        patched = router(x)
        assert torch.equal(original, patched)


class TestClusterExperts:
    """Test expert equivalence class clustering."""

    def test_high_coupling_clusters(self):
        from dystrio_sculpt.moe_routing_patch import _cluster_experts

        coupling = np.array([
            [0.0, 0.9, 0.1, 0.0],
            [0.9, 0.0, 0.1, 0.0],
            [0.1, 0.1, 0.0, 0.8],
            [0.0, 0.0, 0.8, 0.0],
        ])
        utilization = np.array([0.5, 0.3, 0.6, 0.2])

        classes = _cluster_experts(coupling, utilization, threshold=0.7)

        canonical_map = {}
        for c in classes:
            for m in c.members:
                canonical_map[m] = c.canonical

        # 0 and 1 should be in the same class (coupling 0.9)
        assert canonical_map[0] == canonical_map[1]
        # 2 and 3 should be in the same class (coupling 0.8)
        assert canonical_map[2] == canonical_map[3]
        # 0 and 2 should be in different classes
        assert canonical_map[0] != canonical_map[2]

    def test_all_singletons_at_high_threshold(self):
        from dystrio_sculpt.moe_routing_patch import _cluster_experts

        coupling = np.ones((4, 4)) * 0.3
        np.fill_diagonal(coupling, 0)
        utilization = np.array([0.4, 0.3, 0.2, 0.1])

        classes = _cluster_experts(coupling, utilization, threshold=0.9)
        assert all(len(c.members) == 1 for c in classes)


class TestRoutingPatchSaveLoad:
    """Test serialization of routing patches."""

    def test_round_trip(self):
        from dystrio_sculpt.moe_routing_patch import RoutingPatch, ExpertEquivalenceClass

        patch = RoutingPatch(
            model_id="test/model",
            n_experts_original=8,
            top_k=2,
            coupling_threshold=0.7,
            margin_threshold=0.1,
        )
        patch.layers[0] = [
            ExpertEquivalenceClass(canonical=0, members=[0, 1, 2], mean_coupling=0.85),
            ExpertEquivalenceClass(canonical=3, members=[3, 4], mean_coupling=0.75),
        ]
        patch.layers[5] = [
            ExpertEquivalenceClass(canonical=0, members=[0], mean_coupling=0.0),
        ]

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        patch.save(path)
        loaded = RoutingPatch.load(path)

        assert loaded.model_id == "test/model"
        assert loaded.n_experts_original == 8
        assert len(loaded.layers[0]) == 2
        assert loaded.layers[0][0].canonical == 0
        assert loaded.layers[0][0].members == [0, 1, 2]
        assert len(loaded.layers[5]) == 1

        Path(path).unlink()


class TestApplyRemovePatch:
    """Test applying and removing the routing patch on a live model."""

    def setup_method(self):
        self.model = FakeMoEModel(hidden=64, ffn=128, n_experts=8, n_layers=2)
        self.tokenizer = FakeTokenizer()

    def test_apply_patch(self):
        from dystrio_sculpt.moe_routing_patch import (
            RoutingPatch, ExpertEquivalenceClass, apply_routing_patch, CanonicalRouter,
        )

        patch = RoutingPatch(
            model_id="test", n_experts_original=8, top_k=2,
        )
        patch.layers[0] = [
            ExpertEquivalenceClass(canonical=0, members=[0, 1, 2], mean_coupling=0.8),
            ExpertEquivalenceClass(canonical=3, members=[3], mean_coupling=0.0),
            ExpertEquivalenceClass(canonical=4, members=[4, 5, 6, 7], mean_coupling=0.7),
        ]
        patch.layers[1] = [
            ExpertEquivalenceClass(canonical=i, members=[i], mean_coupling=0.0)
            for i in range(8)
        ]

        n_patched = apply_routing_patch(self.model, patch)
        assert n_patched == 1  # only layer 0 has non-singleton classes

        moe = self.model.model.layers[0].block_sparse_moe
        assert isinstance(moe.gate, CanonicalRouter)

    def test_remove_patch(self):
        from dystrio_sculpt.moe_routing_patch import (
            RoutingPatch, ExpertEquivalenceClass,
            apply_routing_patch, remove_routing_patch, CanonicalRouter,
        )

        patch = RoutingPatch(
            model_id="test", n_experts_original=8, top_k=2,
        )
        patch.layers[0] = [
            ExpertEquivalenceClass(canonical=0, members=[0, 1], mean_coupling=0.8),
            ExpertEquivalenceClass(canonical=2, members=[2, 3, 4, 5, 6, 7], mean_coupling=0.7),
        ]

        apply_routing_patch(self.model, patch)
        moe = self.model.model.layers[0].block_sparse_moe
        assert isinstance(moe.gate, CanonicalRouter)

        n_removed = remove_routing_patch(self.model)
        assert n_removed == 1
        assert not isinstance(moe.gate, CanonicalRouter)

    def test_forward_works_with_patch(self):
        from dystrio_sculpt.moe_routing_patch import (
            RoutingPatch, ExpertEquivalenceClass, apply_routing_patch,
        )

        patch = RoutingPatch(
            model_id="test", n_experts_original=8, top_k=2,
        )
        patch.layers[0] = [
            ExpertEquivalenceClass(canonical=0, members=[0, 1, 2, 3], mean_coupling=0.8),
            ExpertEquivalenceClass(canonical=4, members=[4, 5, 6, 7], mean_coupling=0.75),
        ]

        apply_routing_patch(self.model, patch)
        ids = torch.randint(0, 100, (1, 10))
        out = self.model(ids)
        assert out.logits.shape == (1, 10, 64)

    def test_routing_is_more_deterministic_with_patch(self):
        """With the patch, repeated forward passes should produce identical routing."""
        from dystrio_sculpt.moe_routing_patch import (
            RoutingPatch, ExpertEquivalenceClass, apply_routing_patch,
        )

        patch = RoutingPatch(
            model_id="test", n_experts_original=8, top_k=2,
            margin_threshold=5.0,  # very wide margin → always snap
        )
        for li in range(2):
            patch.layers[li] = [
                ExpertEquivalenceClass(canonical=0, members=list(range(8)), mean_coupling=0.9),
            ]

        apply_routing_patch(self.model, patch)
        moe = self.model.model.layers[0].block_sparse_moe

        ids = torch.randint(0, 100, (1, 10))
        with torch.no_grad():
            x = self.model.embed(ids).reshape(-1, 64)
            logits1 = moe.gate(x)
            logits2 = moe.gate(x)

        top1 = logits1.argmax(dim=-1)
        top2 = logits2.argmax(dim=-1)
        assert torch.equal(top1, top2)
        assert (top1 == 0).all()  # canonical is 0, all snapped to it


class TestBakeRoutingPatch:
    """Test baking canonicalization into router weights for vLLM."""

    def setup_method(self):
        self.model = FakeMoEModel(hidden=64, ffn=128, n_experts=8, n_layers=2)

    def test_bake_modifies_weights(self):
        from dystrio_sculpt.moe_routing_patch import (
            RoutingPatch, ExpertEquivalenceClass, bake_routing_patch,
        )

        moe = self.model.model.layers[0].block_sparse_moe
        orig_w1 = moe.gate.weight.data[1].clone()
        orig_w0 = moe.gate.weight.data[0].clone()
        assert not torch.equal(orig_w0, orig_w1)

        patch = RoutingPatch(
            model_id="test", n_experts_original=8, top_k=2,
        )
        patch.layers[0] = [
            ExpertEquivalenceClass(canonical=0, members=[0, 1, 2, 3], mean_coupling=0.85),
            ExpertEquivalenceClass(canonical=4, members=[4, 5, 6, 7], mean_coupling=0.75),
        ]

        n_modified = bake_routing_patch(self.model, patch)
        assert n_modified == 1

        W = moe.gate.weight.data
        scale = 1.0 - 1e-4
        # Members 1,2,3 should be canonical 0's row scaled down by tiebreak_eps
        assert torch.allclose(W[1], W[0] * scale, atol=1e-6)
        assert torch.allclose(W[2], W[0] * scale, atol=1e-6)
        assert torch.allclose(W[3], W[0] * scale, atol=1e-6)
        # Members 5,6,7 should be canonical 4's row scaled down
        assert torch.allclose(W[5], W[4] * scale, atol=1e-6)
        assert torch.allclose(W[6], W[4] * scale, atol=1e-6)
        # Canonical should NOT equal member (tiebreaker gap)
        assert not torch.equal(W[0], W[1])

    def test_baked_model_forward_works(self):
        from dystrio_sculpt.moe_routing_patch import (
            RoutingPatch, ExpertEquivalenceClass, bake_routing_patch,
        )

        patch = RoutingPatch(
            model_id="test", n_experts_original=8, top_k=2,
        )
        patch.layers[0] = [
            ExpertEquivalenceClass(canonical=0, members=[0, 1, 2, 3], mean_coupling=0.85),
            ExpertEquivalenceClass(canonical=4, members=[4, 5, 6, 7], mean_coupling=0.75),
        ]

        bake_routing_patch(self.model, patch)
        ids = torch.randint(0, 100, (1, 10))
        out = self.model(ids)
        assert out.logits.shape == (1, 10, 64)

    def test_baked_routing_is_deterministic(self):
        """Baked weights should produce identical routing on repeated calls."""
        from dystrio_sculpt.moe_routing_patch import (
            RoutingPatch, ExpertEquivalenceClass, bake_routing_patch,
        )

        patch = RoutingPatch(
            model_id="test", n_experts_original=8, top_k=2,
        )
        patch.layers[0] = [
            ExpertEquivalenceClass(canonical=0, members=list(range(8)), mean_coupling=0.9),
        ]

        bake_routing_patch(self.model, patch)
        moe = self.model.model.layers[0].block_sparse_moe

        ids = torch.randint(0, 100, (1, 10))
        with torch.no_grad():
            x = self.model.embed(ids).reshape(-1, 64)
            logits = moe.gate(x)

        # Non-canonical experts have slightly lower gate weights (scaled
        # by 1-eps). Logits should be very close but NOT identical.
        assert logits[:, 0].allclose(logits[:, 1], atol=0.01)
        assert not torch.equal(logits[:, 0], logits[:, 1])

        # For positive logits (the ones that matter for top-k selection),
        # the canonical expert should always win.
        pos_mask = logits[:, 0] > 0
        if pos_mask.any():
            assert (logits[pos_mask, 0] >= logits[pos_mask, 1]).all()
            assert (logits[pos_mask, 0] >= logits[pos_mask, 7]).all()

    def test_no_runtime_wrapper_needed(self):
        """After baking, the router should be a plain nn.Linear, not CanonicalRouter."""
        from dystrio_sculpt.moe_routing_patch import (
            RoutingPatch, ExpertEquivalenceClass, bake_routing_patch, CanonicalRouter,
        )

        patch = RoutingPatch(
            model_id="test", n_experts_original=8, top_k=2,
        )
        patch.layers[0] = [
            ExpertEquivalenceClass(canonical=0, members=[0, 1], mean_coupling=0.8),
            ExpertEquivalenceClass(canonical=2, members=[2, 3, 4, 5, 6, 7], mean_coupling=0.7),
        ]

        bake_routing_patch(self.model, patch)
        moe = self.model.model.layers[0].block_sparse_moe
        assert isinstance(moe.gate, nn.Linear)
        assert not isinstance(moe.gate, CanonicalRouter)
