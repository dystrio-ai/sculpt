"""Tests for the MiniCPM-o/V multimodal adapter.

Builds a fake MiniCPM-o model with the same module hierarchy:
  model.llm.model.layers[i].mlp.{gate_proj, up_proj, down_proj}
and verifies that the adapter correctly navigates through the
multimodal wrapper to prune only the LLM backbone.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn


# ── Fake model that mimics MiniCPM-o module hierarchy ────────────────────


class FakeSwiGLUMLP(nn.Module):
    def __init__(self, hidden: int, ffn: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, ffn, bias=False)
        self.up_proj = nn.Linear(hidden, ffn, bias=False)
        self.down_proj = nn.Linear(ffn, hidden, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class FakeTransformerLayer(nn.Module):
    def __init__(self, hidden: int, ffn: int):
        super().__init__()
        self.mlp = FakeSwiGLUMLP(hidden, ffn)

    def forward(self, x, **kw):
        return x + self.mlp(x)


class FakeQwenModel(nn.Module):
    """Mimics model.llm.model (the inner Qwen3Model)."""

    def __init__(self, hidden: int, ffn: int, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [FakeTransformerLayer(hidden, ffn) for _ in range(n_layers)]
        )


class FakeQwenForCausalLM(nn.Module):
    """Mimics model.llm (Qwen3ForCausalLM)."""

    def __init__(self, hidden: int, ffn: int, n_layers: int, vocab: int):
        super().__init__()
        self.config = SimpleNamespace(
            num_hidden_layers=n_layers,
            hidden_size=hidden,
            intermediate_size=ffn,
            vocab_size=vocab,
        )
        self.model = FakeQwenModel(hidden, ffn, n_layers)
        self.embed = nn.Embedding(vocab, hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids=None, attention_mask=None, **kw):
        x = self.embed(input_ids)
        for layer in self.model.layers:
            x = layer(x)
        logits = self.lm_head(x)
        return SimpleNamespace(logits=logits)


class FakeVisionEncoder(nn.Module):
    """Placeholder vision encoder (not pruned)."""

    def __init__(self):
        super().__init__()
        self.dummy = nn.Linear(10, 10)


class FakeMiniCPMO(nn.Module):
    """Full MiniCPM-o wrapper with model.llm, model.vpm, etc."""

    def __init__(
        self, hidden: int = 64, ffn: int = 128,
        n_layers: int = 4, vocab: int = 200,
    ):
        super().__init__()
        self.config = SimpleNamespace(
            model_type="minicpmo",
            num_hidden_layers=n_layers,
            hidden_size=hidden,
            intermediate_size=ffn,
            vocab_size=vocab,
            hidden_act="silu",
        )
        self.llm = FakeQwenForCausalLM(hidden, ffn, n_layers, vocab)
        self.vpm = FakeVisionEncoder()

    def forward(self, input_ids=None, **kw):
        return self.llm(input_ids=input_ids, **kw)


class FakeTokenizer:
    def __call__(self, text, return_tensors="pt", truncation=True, max_length=256):
        ids = torch.randint(0, 200, (1, min(len(text.split()) + 5, max_length)))
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


# ── Tests ─────────────────────────────────────────────────────────────────


class TestMiniCPMAdapterAccess:
    """Verify the adapter correctly navigates model.llm.model.layers."""

    def setup_method(self):
        self.model = FakeMiniCPMO(hidden=64, ffn=128, n_layers=4)

    def test_get_num_layers(self):
        from dystrio_sculpt.architectures.minicpm import MiniCPMAdapter

        adapter = MiniCPMAdapter()
        assert adapter.get_num_layers(self.model) == 4

    def test_get_mlp(self):
        from dystrio_sculpt.architectures.minicpm import MiniCPMAdapter

        adapter = MiniCPMAdapter()
        mlp = adapter.get_mlp(self.model, 0)
        assert hasattr(mlp, "gate_proj")
        assert hasattr(mlp, "up_proj")
        assert hasattr(mlp, "down_proj")

    def test_get_ffn_size(self):
        from dystrio_sculpt.architectures.minicpm import MiniCPMAdapter

        adapter = MiniCPMAdapter()
        assert adapter.get_ffn_size(self.model, 0) == 128

    def test_supported_targets(self):
        from dystrio_sculpt.architectures.minicpm import MiniCPMAdapter
        from dystrio_sculpt.architectures.descriptor import OptimizationTarget

        adapter = MiniCPMAdapter()
        assert OptimizationTarget.MLP_BLOCK in adapter.supported_targets()


class TestMiniCPMCompression:
    """Test in-place SwiGLU compression via the adapter."""

    def setup_method(self):
        self.model = FakeMiniCPMO(hidden=64, ffn=128, n_layers=4)

    def test_compress_reduces_ffn(self):
        from dystrio_sculpt.architectures.minicpm import MiniCPMAdapter

        adapter = MiniCPMAdapter()
        kept_idx = torch.arange(0, 64)  # keep half
        result = adapter.compress_layer(
            self.model, 0, kept_idx, torch.float32, "cpu",
        )
        assert result["hidden"] == 64
        assert result["ffn_kept"] == 64
        mlp = self.model.llm.model.layers[0].mlp
        assert mlp.gate_proj.out_features == 64
        assert mlp.up_proj.out_features == 64
        assert mlp.down_proj.in_features == 64

    def test_vision_encoder_untouched(self):
        from dystrio_sculpt.architectures.minicpm import MiniCPMAdapter

        adapter = MiniCPMAdapter()
        orig_weight = self.model.vpm.dummy.weight.data.clone()
        kept_idx = torch.arange(0, 64)
        adapter.compress_layer(self.model, 0, kept_idx, torch.float32, "cpu")
        assert torch.equal(self.model.vpm.dummy.weight.data, orig_weight)

    def test_forward_after_compression(self):
        from dystrio_sculpt.architectures.minicpm import MiniCPMAdapter

        adapter = MiniCPMAdapter()
        kept_idx = torch.arange(0, 64)
        adapter.compress_layer(self.model, 0, kept_idx, torch.float32, "cpu")
        ids = torch.randint(0, 200, (1, 10))
        out = self.model(ids)
        assert out.logits.shape == (1, 10, 200)

    def test_other_layers_unchanged(self):
        from dystrio_sculpt.architectures.minicpm import MiniCPMAdapter

        adapter = MiniCPMAdapter()
        orig_gate = self.model.llm.model.layers[1].mlp.gate_proj.weight.clone()
        kept_idx = torch.arange(0, 64)
        adapter.compress_layer(self.model, 0, kept_idx, torch.float32, "cpu")
        assert torch.equal(
            self.model.llm.model.layers[1].mlp.gate_proj.weight, orig_gate,
        )


class TestMiniCPMRepairSupport:
    """Test snapshot, restore, and trainable param extraction."""

    def setup_method(self):
        self.model = FakeMiniCPMO(hidden=64, ffn=128, n_layers=4)

    def test_snapshot_has_mlp_params(self):
        from dystrio_sculpt.architectures.minicpm import MiniCPMAdapter

        adapter = MiniCPMAdapter()
        snap = adapter.snapshot_trainable(self.model, [0, 1])
        gate_keys = [k for k in snap if "gate_proj" in k]
        up_keys = [k for k in snap if "up_proj" in k]
        down_keys = [k for k in snap if "down_proj" in k]
        assert len(gate_keys) == 2  # one per layer
        assert len(up_keys) == 2
        assert len(down_keys) == 2

    def test_restore_recovers_weights(self):
        from dystrio_sculpt.architectures.minicpm import MiniCPMAdapter

        adapter = MiniCPMAdapter()
        snap = adapter.snapshot_trainable(self.model, [0])
        mlp = self.model.llm.model.layers[0].mlp
        orig = mlp.gate_proj.weight.data.clone()
        mlp.gate_proj.weight.data.zero_()
        assert not torch.equal(mlp.gate_proj.weight.data, orig)
        adapter.restore_trainable(self.model, [0], snap)
        assert torch.equal(mlp.gate_proj.weight.data, orig)

    def test_trainable_params_count(self):
        from dystrio_sculpt.architectures.minicpm import MiniCPMAdapter

        adapter = MiniCPMAdapter()
        params = adapter.get_trainable_params(self.model, [0])
        assert len(params) == 3  # gate, up, down (no bias)


class TestMiniCPMCalibration:
    """Test that calibration hooks fire through model.llm forward pass."""

    def setup_method(self):
        self.model = FakeMiniCPMO(hidden=64, ffn=128, n_layers=4)
        self.tokenizer = FakeTokenizer()
        self.texts = ["hello world test sentence"] * 5

    def test_importance_shape(self):
        from dystrio_sculpt.architectures.minicpm import MiniCPMAdapter

        adapter = MiniCPMAdapter()
        imp = adapter.collect_importance(
            self.model, self.tokenizer, 0, self.texts, 64, "cpu",
        )
        assert imp.shape == (128,)
        assert imp.sum() > 0

    def test_block_geometry_shape(self):
        from dystrio_sculpt.architectures.minicpm import MiniCPMAdapter

        adapter = MiniCPMAdapter()
        result = adapter.collect_block_geometry(
            self.model, self.tokenizer, 0, self.texts, 64, "cpu",
            block_size=16, max_tokens=200,
        )
        assert "D" in result
        assert "block_energy" in result
        assert result["n_blocks"] == 8  # 128 / 16

    def test_block_sensitivity_shape(self):
        from dystrio_sculpt.architectures.minicpm import MiniCPMAdapter

        adapter = MiniCPMAdapter()
        result = adapter.collect_block_sensitivity(
            self.model, self.tokenizer, 0, self.texts, 64, "cpu",
            block_size=16, max_tokens=200,
        )
        assert "block_sensitivity" in result
        assert result["block_sensitivity"].shape[0] == 8


class TestMiniCPMFallback:
    """Adapter falls back gracefully when model doesn't have .llm."""

    def test_works_on_plain_qwen(self):
        from dystrio_sculpt.architectures.minicpm import MiniCPMAdapter

        llm = FakeQwenForCausalLM(hidden=64, ffn=128, n_layers=2, vocab=200)
        adapter = MiniCPMAdapter()
        assert adapter.get_num_layers(llm) == 2
        assert adapter.get_ffn_size(llm, 0) == 128


class TestRegistryIntegration:
    """Verify the adapter is registered and retrievable."""

    def test_registry_has_minicpm(self):
        from dystrio_sculpt.architectures import _REGISTRY

        assert "minicpm" in _REGISTRY

    def test_get_adapter_returns_minicpm(self):
        from dystrio_sculpt.architectures import get_adapter
        from dystrio_sculpt.architectures.descriptor import (
            ArchitectureDescriptor,
            SupportState,
        )
        from dystrio_sculpt.architectures.minicpm import MiniCPMAdapter

        desc = ArchitectureDescriptor(
            family="minicpm",
            model_type="minicpmo",
            support_state=SupportState.SUPPORTED,
        )
        adapter = get_adapter(desc)
        assert isinstance(adapter, MiniCPMAdapter)


class TestFindLayersIntegration:
    """Verify _find_layers discovers model.llm.model.layers."""

    def test_find_layers_resolves_llm_path(self):
        from dystrio_sculpt._model import _find_layers

        model = FakeMiniCPMO(hidden=64, ffn=128, n_layers=4)
        layers = _find_layers(model)
        assert len(layers) == 4
        assert hasattr(layers[0], "mlp")
