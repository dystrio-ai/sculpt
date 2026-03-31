"""Tests for MoE expert-level calibration, selection, and compression."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn


class FakeExpert(nn.Module):
    """Minimal SwiGLU expert with gate_proj, up_proj, down_proj."""

    def __init__(self, hidden: int, ffn: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, ffn, bias=False)
        self.up_proj = nn.Linear(hidden, ffn, bias=False)
        self.down_proj = nn.Linear(ffn, hidden, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class FakeMoE(nn.Module):
    """Minimal MoE module with experts + gate."""

    def __init__(self, hidden: int, ffn: int, n_experts: int, top_k: int = 2):
        super().__init__()
        self.experts = nn.ModuleList([FakeExpert(hidden, ffn) for _ in range(n_experts)])
        self.gate = nn.Linear(hidden, n_experts, bias=False)
        self.num_experts_per_tok = top_k

    def forward(self, hidden_states):
        if hidden_states.dim() == 3:
            B, T, H = hidden_states.shape
            hidden_states = hidden_states.reshape(B * T, H)
        logits = self.gate(hidden_states)
        weights = torch.softmax(logits, dim=-1)
        topk_weights, topk_indices = torch.topk(weights, self.num_experts_per_tok, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        out = torch.zeros_like(hidden_states)
        for k in range(self.num_experts_per_tok):
            for eidx in range(len(self.experts)):
                mask = topk_indices[:, k] == eidx
                if mask.any():
                    exp_out = self.experts[eidx](hidden_states[mask])
                    out[mask] += topk_weights[mask, k : k + 1] * exp_out
        return out


class FakeLayer(nn.Module):
    def __init__(self, hidden: int, ffn: int, n_experts: int):
        super().__init__()
        self.block_sparse_moe = FakeMoE(hidden, ffn, n_experts)


class FakeMoEModel(nn.Module):
    """Minimal model with MoE layers for testing."""

    def __init__(self, hidden: int = 64, ffn: int = 128, n_experts: int = 8,
                 n_layers: int = 2, vocab: int = 100):
        super().__init__()
        self.config = SimpleNamespace(
            num_hidden_layers=n_layers,
            hidden_size=hidden,
            intermediate_size=ffn,
            num_local_experts=n_experts,
            num_experts_per_tok=2,
            vocab_size=vocab,
        )
        self.embed = nn.Embedding(vocab, hidden)
        self.model = SimpleNamespace(
            layers=nn.ModuleList([FakeLayer(hidden, ffn, n_experts) for _ in range(n_layers)])
        )

    def forward(self, input_ids, **kwargs):
        x = self.embed(input_ids)
        for layer in self.model.layers:
            x = x + layer.block_sparse_moe(x)
        return SimpleNamespace(logits=x)


class FakeTokenizer:
    def __call__(self, text, return_tensors="pt", truncation=True, max_length=256):
        ids = torch.randint(0, 100, (1, min(len(text.split()) + 5, max_length)))
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


# ── Adapter tests ─────────────────────────────────────────────────────────


class TestSwiGLUMoEAdapterAccess:
    """Test layer access methods."""

    def setup_method(self):
        self.model = FakeMoEModel(hidden=64, ffn=128, n_experts=8, n_layers=2)

    def test_get_num_layers(self):
        from dystrio_sculpt.architectures.swiglu_moe import SwiGLUMoEAdapter
        adapter = SwiGLUMoEAdapter()
        assert adapter.get_num_layers(self.model) == 2

    def test_get_ffn_size_returns_num_experts(self):
        """For MoE, get_ffn_size returns num_experts so the engine
        treats each expert as one selectable block."""
        from dystrio_sculpt.architectures.swiglu_moe import SwiGLUMoEAdapter
        adapter = SwiGLUMoEAdapter()
        assert adapter.get_ffn_size(self.model, 0) == 8
        assert adapter.get_ffn_size(self.model, 0) == adapter.get_num_experts(self.model, 0)

    def test_get_num_experts(self):
        from dystrio_sculpt.architectures.swiglu_moe import SwiGLUMoEAdapter
        adapter = SwiGLUMoEAdapter()
        assert adapter.get_num_experts(self.model, 0) == 8

    def test_get_mlp_returns_moe(self):
        from dystrio_sculpt.architectures.swiglu_moe import SwiGLUMoEAdapter
        adapter = SwiGLUMoEAdapter()
        moe = adapter.get_mlp(self.model, 0)
        assert hasattr(moe, "experts")
        assert hasattr(moe, "gate")


class TestMoECompression:
    """Test expert dropping, merging, and router rescaling."""

    def setup_method(self):
        self.model = FakeMoEModel(hidden=64, ffn=128, n_experts=8, n_layers=2)

    def test_drop_experts(self):
        from dystrio_sculpt.architectures.swiglu_moe import SwiGLUMoEAdapter
        adapter = SwiGLUMoEAdapter()
        kept = torch.tensor([0, 1, 2, 3])
        result = adapter.compress_layer(
            self.model, 0, kept, torch.bfloat16, "cpu", merge=False,
        )
        assert result["n_experts_orig"] == 8
        assert result["n_experts_kept"] == 4
        moe = self.model.model.layers[0].block_sparse_moe
        assert len(moe.experts) == 4

    def test_router_resized(self):
        from dystrio_sculpt.architectures.swiglu_moe import SwiGLUMoEAdapter
        adapter = SwiGLUMoEAdapter()
        kept = torch.tensor([0, 2, 4, 6])
        adapter.compress_layer(self.model, 0, kept, torch.bfloat16, "cpu", merge=False)
        moe = self.model.model.layers[0].block_sparse_moe
        assert moe.gate.out_features == 4

    def test_merge_preserves_expert_count(self):
        from dystrio_sculpt.architectures.swiglu_moe import SwiGLUMoEAdapter
        adapter = SwiGLUMoEAdapter()
        coupling = np.eye(8) * 0.1
        coupling[4, 0] = coupling[0, 4] = 0.9
        coupling[5, 1] = coupling[1, 5] = 0.8
        kept = torch.tensor([0, 1, 2, 3])
        result = adapter.compress_layer(
            self.model, 0, kept, torch.bfloat16, "cpu",
            merge=True, coupling_matrix=coupling,
        )
        assert result["n_experts_kept"] == 4
        assert result["n_merged"] == 4

    def test_config_updated(self):
        from dystrio_sculpt.architectures.swiglu_moe import SwiGLUMoEAdapter
        adapter = SwiGLUMoEAdapter()
        kept = torch.tensor([0, 1, 2, 3])
        adapter.compress_layer(self.model, 0, kept, torch.bfloat16, "cpu", merge=False)
        assert self.model.config.num_local_experts == 4

    def test_forward_after_compression(self):
        """Model should still produce output after expert dropping."""
        from dystrio_sculpt.architectures.swiglu_moe import SwiGLUMoEAdapter
        adapter = SwiGLUMoEAdapter()
        kept = torch.tensor([0, 1, 2, 3])
        adapter.compress_layer(self.model, 0, kept, torch.float32, "cpu", merge=False)
        # Update the MoE's internal count so forward pass routes correctly
        self.model.model.layers[0].block_sparse_moe.num_experts_per_tok = 2
        ids = torch.randint(0, 100, (1, 10))
        out = self.model(ids)
        assert out.logits.shape == (1, 10, 64)


class TestMoERepairSupport:
    """Test snapshot/restore/trainable for MoE layers."""

    def setup_method(self):
        self.model = FakeMoEModel(hidden=64, ffn=128, n_experts=4, n_layers=2)

    def test_snapshot_has_all_params(self):
        from dystrio_sculpt.architectures.swiglu_moe import SwiGLUMoEAdapter
        adapter = SwiGLUMoEAdapter()
        snap = adapter.snapshot_trainable(self.model, [0])
        expert_keys = [k for k in snap if "expert" in k]
        gate_keys = [k for k in snap if "gate" in k]
        assert len(expert_keys) > 0
        assert len(gate_keys) > 0

    def test_restore_recovers_weights(self):
        from dystrio_sculpt.architectures.swiglu_moe import SwiGLUMoEAdapter
        adapter = SwiGLUMoEAdapter()
        snap = adapter.snapshot_trainable(self.model, [0])
        moe = self.model.model.layers[0].block_sparse_moe
        original_w = moe.experts[0].gate_proj.weight.data.clone()
        moe.experts[0].gate_proj.weight.data.zero_()
        assert not torch.equal(moe.experts[0].gate_proj.weight.data, original_w)
        adapter.restore_trainable(self.model, [0], snap)
        assert torch.equal(moe.experts[0].gate_proj.weight.data, original_w)

    def test_trainable_params_count(self):
        from dystrio_sculpt.architectures.swiglu_moe import SwiGLUMoEAdapter
        adapter = SwiGLUMoEAdapter()
        params = adapter.get_trainable_params(self.model, [0])
        # 4 experts × 3 proj × 1 weight each + gate weight = 13
        assert len(params) == 13


class TestMoECalibration:
    """Test expert-level calibration functions."""

    def setup_method(self):
        self.model = FakeMoEModel(hidden=64, ffn=128, n_experts=8, n_layers=2)
        self.tokenizer = FakeTokenizer()
        self.texts = ["hello world test"] * 5

    def test_expert_utilization_shape(self):
        from dystrio_sculpt._calibrate_moe import collect_expert_utilization
        result = collect_expert_utilization(
            self.model, self.tokenizer, 0, self.texts, 64, "cpu",
            max_tokens=200,
        )
        assert result["expert_frequency"].shape == (8,)
        assert result["expert_avg_weight"].shape == (8,)
        assert result["n_experts"] == 8
        assert (result["expert_frequency"] >= 0).all()

    def test_expert_sensitivity_shape(self):
        from dystrio_sculpt._calibrate_moe import collect_expert_sensitivity
        result = collect_expert_sensitivity(
            self.model, self.tokenizer, 0, self.texts, 64, "cpu",
            max_tokens=200,
        )
        assert result["expert_sensitivity"].shape == (8,)
        assert result["expert_energy"].shape == (8,)
        assert (result["expert_sensitivity"] >= 0).all()

    def test_expert_covariance_shape(self):
        from dystrio_sculpt._calibrate_moe import collect_expert_covariance
        result = collect_expert_covariance(
            self.model, self.tokenizer, 0, self.texts, 64, "cpu",
            max_tokens=200,
        )
        assert result["D"].shape == (24, 24)  # 3 features × 8 experts
        assert result["n_blocks"] == 8
        assert result["feature_multiplier"] == 3

    def test_adapter_collect_block_geometry(self):
        """Adapter's collect_block_geometry should return Physarum-compatible output."""
        from dystrio_sculpt.architectures.swiglu_moe import SwiGLUMoEAdapter
        adapter = SwiGLUMoEAdapter()
        result = adapter.collect_block_geometry(
            self.model, self.tokenizer, 0, self.texts, 64, "cpu",
            max_tokens=200,
        )
        assert "D" in result
        assert "block_energy" in result
        assert result["n_blocks"] == 8

    def test_adapter_collect_block_sensitivity(self):
        from dystrio_sculpt.architectures.swiglu_moe import SwiGLUMoEAdapter
        adapter = SwiGLUMoEAdapter()
        result = adapter.collect_block_sensitivity(
            self.model, self.tokenizer, 0, self.texts, 64, "cpu",
            max_tokens=200,
        )
        assert "block_sensitivity" in result
        assert "block_energy" in result
        assert result["n_blocks"] == 8


class FakeFusedExperts(nn.Module):
    """Mimics OlmoeExperts: 3D Parameter tensors, NOT a ModuleList."""

    def __init__(self, n_experts: int, hidden: int, ffn: int):
        super().__init__()
        self.num_experts = n_experts
        self.hidden_dim = hidden
        self.intermediate_dim = ffn
        self.gate_up_proj = nn.Parameter(torch.randn(n_experts, 2 * ffn, hidden))
        self.down_proj = nn.Parameter(torch.randn(n_experts, hidden, ffn))
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states)
        expert_mask = torch.nn.functional.one_hot(
            top_k_index, num_classes=self.num_experts,
        )
        expert_mask = expert_mask.permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current = hidden_states[token_idx]
            gate, up = nn.functional.linear(
                current, self.gate_up_proj[expert_idx],
            ).chunk(2, dim=-1)
            out = self.act_fn(gate) * up
            out = nn.functional.linear(out, self.down_proj[expert_idx])
            out = out * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, out)
        return final


class FakeFusedRouter(nn.Module):
    """Mimics OlmoeTopKRouter: returns (logits, scores, indices)."""

    def __init__(self, n_experts: int, hidden: int, top_k: int = 2):
        super().__init__()
        self.num_experts = n_experts
        self.top_k = top_k
        self.hidden_dim = hidden
        self.weight = nn.Parameter(torch.randn(n_experts, hidden))

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        logits = nn.functional.linear(hidden_states, self.weight)
        probs = torch.softmax(logits, dim=-1)
        top_v, top_i = torch.topk(probs, self.top_k, dim=-1)
        top_v = top_v / top_v.sum(dim=-1, keepdim=True)
        return logits, top_v, top_i


class FakeFusedMoE(nn.Module):
    """Mimics OlmoeSparseMoeBlock with fused experts + custom router."""

    def __init__(self, hidden: int, ffn: int, n_experts: int, top_k: int = 2):
        super().__init__()
        self.gate = FakeFusedRouter(n_experts, hidden, top_k)
        self.experts = FakeFusedExperts(n_experts, hidden, ffn)

    def forward(self, hidden_states):
        B, T, H = hidden_states.shape
        h = hidden_states.view(-1, H)
        _, top_k_w, top_k_i = self.gate(h)
        return self.experts(h, top_k_i, top_k_w).reshape(B, T, H)


class FakeFusedLayer(nn.Module):
    def __init__(self, hidden: int, ffn: int, n_experts: int):
        super().__init__()
        self.mlp = FakeFusedMoE(hidden, ffn, n_experts)


class FakeFusedMoEModel(nn.Module):
    """Mimics OLMoE model structure with fused experts."""

    def __init__(self, hidden: int = 64, ffn: int = 128, n_experts: int = 8,
                 n_layers: int = 2, vocab: int = 100):
        super().__init__()
        self.config = SimpleNamespace(
            num_hidden_layers=n_layers,
            hidden_size=hidden,
            intermediate_size=ffn,
            num_local_experts=n_experts,
            num_experts=n_experts,
            num_experts_per_tok=2,
            vocab_size=vocab,
        )
        self.num_experts = n_experts
        self.embed = nn.Embedding(vocab, hidden)
        self.model = SimpleNamespace(
            layers=nn.ModuleList(
                [FakeFusedLayer(hidden, ffn, n_experts) for _ in range(n_layers)]
            )
        )

    def forward(self, input_ids, **kwargs):
        x = self.embed(input_ids)
        for layer in self.model.layers:
            x = x + layer.mlp(x)
        return SimpleNamespace(logits=x)


class TestFusedExpertCompression:
    """Test compression for OLMoE-style fused experts with custom router."""

    def setup_method(self):
        self.model = FakeFusedMoEModel(hidden=64, ffn=128, n_experts=8, n_layers=2)

    def test_fused_detection(self):
        from dystrio_sculpt._calibrate_moe import _experts_are_iterable
        moe = self.model.model.layers[0].mlp
        assert not _experts_are_iterable(moe), "Fused experts should NOT be iterable"

    def test_compress_slices_3d_tensors(self):
        from dystrio_sculpt.architectures.swiglu_moe import SwiGLUMoEAdapter
        adapter = SwiGLUMoEAdapter()
        kept = torch.tensor([0, 1, 2, 3])
        result = adapter.compress_layer(
            self.model, 0, kept, torch.float32, "cpu", merge=False,
        )
        experts = self.model.model.layers[0].mlp.experts
        assert result["n_experts_orig"] == 8
        assert result["n_experts_kept"] == 4
        assert experts.gate_up_proj.shape[0] == 4
        assert experts.down_proj.shape[0] == 4

    def test_experts_num_experts_updated(self):
        """Critical: OlmoeExperts.forward() uses self.num_experts for one_hot."""
        from dystrio_sculpt.architectures.swiglu_moe import SwiGLUMoEAdapter
        adapter = SwiGLUMoEAdapter()
        kept = torch.tensor([0, 1, 2, 3])
        adapter.compress_layer(self.model, 0, kept, torch.float32, "cpu", merge=False)
        experts = self.model.model.layers[0].mlp.experts
        assert experts.num_experts == 4

    def test_router_preserved_not_replaced(self):
        """Custom router must be preserved (not replaced with nn.Linear)."""
        from dystrio_sculpt.architectures.swiglu_moe import SwiGLUMoEAdapter
        adapter = SwiGLUMoEAdapter()
        gate_before = self.model.model.layers[0].mlp.gate
        kept = torch.tensor([0, 1, 2, 3])
        adapter.compress_layer(self.model, 0, kept, torch.float32, "cpu", merge=False)
        gate_after = self.model.model.layers[0].mlp.gate
        assert type(gate_after) is FakeFusedRouter
        assert gate_after.weight.shape[0] == 4
        assert gate_after.num_experts == 4

    def test_router_forward_still_returns_3_values(self):
        from dystrio_sculpt.architectures.swiglu_moe import SwiGLUMoEAdapter
        adapter = SwiGLUMoEAdapter()
        kept = torch.tensor([0, 1, 2, 3])
        adapter.compress_layer(self.model, 0, kept, torch.float32, "cpu", merge=False)
        gate = self.model.model.layers[0].mlp.gate
        h = torch.randn(5, 64)
        result = gate(h)
        assert len(result) == 3, f"Expected 3 return values, got {len(result)}"

    def test_forward_after_fused_compression(self):
        """Full forward pass must work after fused expert compression."""
        from dystrio_sculpt.architectures.swiglu_moe import SwiGLUMoEAdapter
        adapter = SwiGLUMoEAdapter()
        kept = torch.tensor([0, 1, 2, 3])
        adapter.compress_layer(self.model, 0, kept, torch.float32, "cpu", merge=False)
        ids = torch.randint(0, 100, (1, 10))
        out = self.model(ids)
        assert out.logits.shape == (1, 10, 64)

    def test_fused_merge(self):
        from dystrio_sculpt.architectures.swiglu_moe import SwiGLUMoEAdapter
        adapter = SwiGLUMoEAdapter()
        experts = self.model.model.layers[0].mlp.experts
        orig_w = experts.gate_up_proj.data[0].clone()
        coupling = np.eye(8) * 0.1
        coupling[4, 0] = coupling[0, 4] = 0.9
        kept = torch.tensor([0, 1, 2, 3])
        result = adapter.compress_layer(
            self.model, 0, kept, torch.float32, "cpu",
            merge=True, coupling_matrix=coupling,
        )
        assert result["n_merged"] == 4
        assert not torch.equal(
            self.model.model.layers[0].mlp.experts.gate_up_proj.data[0], orig_w
        )

    def test_config_and_model_attrs_updated(self):
        from dystrio_sculpt.architectures.swiglu_moe import SwiGLUMoEAdapter
        adapter = SwiGLUMoEAdapter()
        kept = torch.tensor([0, 1, 2, 3])
        adapter.compress_layer(self.model, 0, kept, torch.float32, "cpu", merge=False)
        assert self.model.config.num_local_experts == 4
        assert self.model.config.num_experts == 4
        assert self.model.num_experts == 4


class TestPhysarumOnExperts:
    """Verify the existing Physarum pipeline works on expert covariance."""

    def test_structural_selection_on_expert_D(self):
        """select_blocks_structural should pick experts from expert covariance."""
        from dystrio_sculpt.selectors.structural import select_blocks_structural
        n_experts = 8
        F = 3
        D = torch.randn(F * n_experts, F * n_experts, dtype=torch.float64)
        D = (D + D.T) / 2
        D += torch.eye(F * n_experts, dtype=torch.float64) * 10

        kept_blocks, kept_idx, arts = select_blocks_structural(
            D, keep_frac=0.5, block_size=1,
            feature_multiplier=F,
        )
        assert len(kept_blocks) == 4
        assert "block_scores" in arts
        assert arts["block_scores"].shape[0] == n_experts
