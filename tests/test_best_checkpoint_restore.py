"""Tests for best-checkpoint restore and never-worse invariant in repair."""

from __future__ import annotations

import math
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from dystrio_sculpt.repair import repair_layers, _snapshot_trainable, _restore_trainable


class TinyMLP(nn.Module):
    def __init__(self, dim: int = 16):
        super().__init__()
        self.gate_proj = nn.Linear(dim, dim * 2, bias=False)
        self.up_proj = nn.Linear(dim, dim * 2, bias=False)
        self.down_proj = nn.Linear(dim * 2, dim, bias=False)

    def forward(self, x):
        return self.down_proj(self.gate_proj(x) * self.up_proj(x))


class TinyLayer(nn.Module):
    def __init__(self, dim: int = 16):
        super().__init__()
        self.mlp = TinyMLP(dim)


class TinyModel(nn.Module):
    """Minimal mock that looks enough like a HF causal LM for repair."""

    def __init__(self, dim: int = 16, n_layers: int = 2, vocab: int = 64):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([TinyLayer(dim) for _ in range(n_layers)])
        self.head = nn.Linear(dim, vocab, bias=False)
        self.config = MagicMock()
        self.config.vocab_size = vocab

    def forward(self, input_ids, **kwargs):
        x = self.embed(input_ids)
        for layer in self.model.layers:
            x = x + layer.mlp(x)
        logits = self.head(x)
        return MagicMock(logits=logits)


class FakeTokenizer:
    def __call__(self, text, return_tensors="pt", truncation=True, max_length=256):
        ids = torch.randint(0, 64, (1, 20))
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


@pytest.fixture
def tiny_setup():
    model = TinyModel().to("cpu")
    tok = FakeTokenizer()
    texts = [f"text_{i}" for i in range(100)]
    return model, tok, texts


class TestSnapshotRestore:
    def test_snapshot_roundtrip(self, tiny_setup):
        model, _, _ = tiny_setup
        snap = _snapshot_trainable(model, [0, 1])
        assert len(snap) > 0
        # Mutate weights
        for li in [0, 1]:
            for p in model.model.layers[li].mlp.parameters():
                p.data.fill_(999.0)
        # Restore
        _restore_trainable(model, [0, 1], snap)
        for key, original in snap.items():
            li = int(key.split(".")[1])
            name = key.split("mlp.")[1]
            current = dict(model.model.layers[li].mlp.named_parameters())[name]
            assert torch.allclose(current.data.cpu(), original), f"mismatch at {key}"


class TestBestCheckpoint:
    def test_best_checkpoint_restored_on_worsening(self, tiny_setup):
        """If metric worsens late, best checkpoint from earlier is restored."""
        model, tok, texts = tiny_setup

        # Simulate a curve that improves then worsens
        call_count = [0]
        def curve_fn(step):
            call_count[0] += 1
            n = call_count[0]
            if n <= 2:
                return {"ppl_w103_valid": 10.0 - n}  # improves
            return {"ppl_w103_valid": 15.0 + n}  # worsens

        result = repair_layers(
            model=model, tokenizer=tok, texts_train=texts,
            layers=[0], steps=20, lr=1e-3, warmup=0,
            device="cpu", curve_fn=curve_fn, curve_every=3,
            early_stop_patience=2, save_best=True,
        )

        assert result["best_metric"] <= 10.0
        assert result["repaired_ok"] is True

    def test_never_worse_rollback(self, tiny_setup):
        """If repair never beats pre-repair, weights are rolled back."""
        model, tok, texts = tiny_setup

        pre_ppl = 5.0

        # curve_fn always returns worse than pre_ppl
        def curve_fn(step):
            return {"ppl_w103_valid": 20.0}

        result = repair_layers(
            model=model, tokenizer=tok, texts_train=texts,
            layers=[0], steps=10, lr=1e-3, warmup=0,
            device="cpu", curve_fn=curve_fn, curve_every=2,
            early_stop_patience=3, save_best=True,
            pre_repair_metric=pre_ppl, never_worse_eps=0.01,
        )

        assert result["repaired_ok"] is False

    def test_nan_metric_triggers_rollback(self, tiny_setup):
        model, tok, texts = tiny_setup

        call_count = [0]
        def curve_fn(step):
            call_count[0] += 1
            if call_count[0] > 2:
                return {"ppl_w103_valid": float("nan")}
            return {"ppl_w103_valid": 10.0}

        result = repair_layers(
            model=model, tokenizer=tok, texts_train=texts,
            layers=[0], steps=20, lr=1e-3, warmup=0,
            device="cpu", curve_fn=curve_fn, curve_every=3,
            early_stop_patience=5, save_best=True,
        )

        assert result["early_stopped"] is True
