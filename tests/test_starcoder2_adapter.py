"""Tests for the StarCoder2 plain-MLP adapter.

Builds a fake StarCoder2 model with c_fc/c_proj naming and verifies
that calibration, compression, and repair work correctly.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn


# ── Fake model mimicking StarCoder2 module hierarchy ──────────────────────


class FakePlainMLP(nn.Module):
    def __init__(self, hidden: int, ffn: int):
        super().__init__()
        self.c_fc = nn.Linear(hidden, ffn, bias=True)
        self.c_proj = nn.Linear(ffn, hidden, bias=True)
        self.act = nn.GELU(approximate="tanh")
        self.residual_dropout = 0.0

    def forward(self, x):
        return self.c_proj(self.act(self.c_fc(x)))


class FakeTransformerLayer(nn.Module):
    def __init__(self, hidden: int, ffn: int):
        super().__init__()
        self.mlp = FakePlainMLP(hidden, ffn)
        self.self_attn = nn.Linear(hidden, hidden)
        self.input_layernorm = nn.LayerNorm(hidden)
        self.post_attention_layernorm = nn.LayerNorm(hidden)

    def forward(self, x, **kw):
        return x + self.mlp(self.input_layernorm(x))


class FakeStarcoder2Model(nn.Module):
    def __init__(self, hidden: int, ffn: int, n_layers: int, vocab: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [FakeTransformerLayer(hidden, ffn) for _ in range(n_layers)]
        )
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.norm = nn.LayerNorm(hidden)


class FakeStarcoder2ForCausalLM(nn.Module):
    """Mimics Starcoder2ForCausalLM: model.model.layers[i].mlp.{c_fc, c_proj}."""

    def __init__(self, hidden: int = 64, ffn: int = 256, n_layers: int = 4, vocab: int = 100):
        super().__init__()
        self.config = SimpleNamespace(
            num_hidden_layers=n_layers,
            hidden_size=hidden,
            intermediate_size=ffn,
            vocab_size=vocab,
            model_type="starcoder2",
            hidden_act="gelu_pytorch_tanh",
            use_bias=True,
        )
        self.model = FakeStarcoder2Model(hidden, ffn, n_layers, vocab)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids=None, **kw):
        x = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            x = layer(x)
        x = self.model.norm(x)
        logits = self.lm_head(x)
        return SimpleNamespace(logits=logits)


def _make_model(hidden=64, ffn=256, n_layers=4):
    return FakeStarcoder2ForCausalLM(hidden=hidden, ffn=ffn, n_layers=n_layers)


def _make_tokenizer():
    class FakeTok:
        pad_token_id = 0
        eos_token_id = 0
        def __call__(self, text, return_tensors="pt", truncation=True, max_length=128):
            ids = torch.randint(1, 99, (1, min(16, max_length)))
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    return FakeTok()


# ── Adapter import ────────────────────────────────────────────────────────


@pytest.fixture
def adapter():
    from dystrio_sculpt.architectures.starcoder2 import Starcoder2Adapter
    return Starcoder2Adapter()


@pytest.fixture
def model():
    return _make_model()


@pytest.fixture
def tokenizer():
    return _make_tokenizer()


# ── Tests ─────────────────────────────────────────────────────────────────


class TestLayerAccess:
    def test_get_num_layers(self, adapter, model):
        assert adapter.get_num_layers(model) == 4

    def test_get_mlp(self, adapter, model):
        mlp = adapter.get_mlp(model, 0)
        assert hasattr(mlp, "c_fc")
        assert hasattr(mlp, "c_proj")
        assert not hasattr(mlp, "gate_proj")

    def test_get_ffn_size(self, adapter, model):
        assert adapter.get_ffn_size(model, 0) == 256


class TestCompression:
    def test_compress_reduces_size(self, adapter, model):
        kept_idx = torch.tensor([0, 1, 2, 3, 10, 20, 50, 100])
        info = adapter.compress_layer(model, 0, kept_idx, torch.float32, "cpu")
        assert info["ffn_kept"] == 8
        assert info["hidden"] == 64
        mlp = model.model.layers[0].mlp
        assert mlp.c_fc.out_features == 8
        assert mlp.c_proj.in_features == 8
        assert mlp.c_fc.in_features == 64
        assert mlp.c_proj.out_features == 64

    def test_compress_preserves_bias(self, adapter, model):
        kept_idx = torch.arange(32)
        adapter.compress_layer(model, 1, kept_idx, torch.float32, "cpu")
        mlp = model.model.layers[1].mlp
        assert mlp.c_fc.bias is not None
        assert mlp.c_fc.bias.shape == (32,)
        assert mlp.c_proj.bias is not None
        assert mlp.c_proj.bias.shape == (64,)

    def test_compress_leaves_other_layers_untouched(self, adapter, model):
        kept_idx = torch.arange(16)
        adapter.compress_layer(model, 0, kept_idx, torch.float32, "cpu")
        assert model.model.layers[1].mlp.c_fc.out_features == 256
        assert model.model.layers[2].mlp.c_fc.out_features == 256


class TestCalibration:
    def test_collect_importance(self, adapter, model, tokenizer):
        imp = adapter.collect_importance(
            model, tokenizer, 0, ["hello world"], 64, "cpu",
        )
        assert imp.shape == (256,)
        assert imp.dtype == torch.float32

    def test_collect_block_geometry(self, adapter, model, tokenizer):
        geom = adapter.collect_block_geometry(
            model, tokenizer, 0, ["hello world"], 64, "cpu",
            block_size=64,
        )
        assert "D" in geom
        assert "block_energy" in geom
        assert "n_blocks" in geom
        assert "feature_multiplier" in geom
        assert geom["n_blocks"] == 4  # 256 / 64
        assert geom["feature_multiplier"] == 3
        assert geom["D"].shape == (12, 12)  # 4 blocks * 3 features

    def test_collect_block_sensitivity(self, adapter, model, tokenizer):
        sens = adapter.collect_block_sensitivity(
            model, tokenizer, 0, ["hello world"], 64, "cpu",
            block_size=64,
        )
        assert "block_sensitivity" in sens
        assert "block_energy" in sens
        assert sens["n_blocks"] == 4


class TestRepair:
    def test_snapshot_and_restore(self, adapter, model):
        snap = adapter.snapshot_trainable(model, [0, 1])
        assert len(snap) > 0
        assert all("layers." in k for k in snap)

        original_w = model.model.layers[0].mlp.c_fc.weight.clone()
        model.model.layers[0].mlp.c_fc.weight.data.zero_()
        assert (model.model.layers[0].mlp.c_fc.weight == 0).all()

        adapter.restore_trainable(model, [0, 1], snap)
        assert torch.equal(model.model.layers[0].mlp.c_fc.weight, original_w)

    def test_get_trainable_params(self, adapter, model):
        params = adapter.get_trainable_params(model, [0])
        assert len(params) == 4  # c_fc.weight, c_fc.bias, c_proj.weight, c_proj.bias
        assert all(isinstance(p, torch.nn.Parameter) for p in params)


class TestRegistry:
    def test_starcoder_family_uses_starcoder2_adapter(self):
        from dystrio_sculpt.architectures import _REGISTRY
        from dystrio_sculpt.architectures.starcoder2 import Starcoder2Adapter
        assert _REGISTRY.get("starcoder") is Starcoder2Adapter

    def test_fingerprint_starcoder2_is_plain(self):
        from dystrio_sculpt.architectures.fingerprint import _KNOWN_ARCHITECTURES
        from dystrio_sculpt.architectures.descriptor import MlpType
        entry = _KNOWN_ARCHITECTURES["starcoder2"]
        assert entry[0] == "starcoder"
        assert entry[1] == MlpType.PLAIN
        assert entry[2] is False  # gating = False


class TestFindLayers:
    def test_find_layers_resolves_starcoder2(self, model):
        from dystrio_sculpt._model import get_layers
        layers = get_layers(model)
        assert len(layers) == 4
        assert hasattr(layers[0].mlp, "c_fc")


class TestForwardPass:
    def test_compressed_model_forward(self, adapter, model, tokenizer):
        """Verify the model produces output after compression."""
        kept_idx = torch.arange(64)
        adapter.compress_layer(model, 0, kept_idx, torch.float32, "cpu")
        adapter.compress_layer(model, 1, kept_idx, torch.float32, "cpu")

        inp = tokenizer("test input")
        inp = {k: v for k, v in inp.items()}
        out = model(**inp)
        assert out.logits.shape[-1] == 100  # vocab size
