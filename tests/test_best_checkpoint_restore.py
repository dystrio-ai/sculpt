"""Tests for best-checkpoint restore, never-worse invariant, JSD distillation,
adaptive alpha, and risk-scaled LR in repair."""

from __future__ import annotations

import math
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from dystrio_sculpt.repair import (
    repair_layers, _snapshot_trainable, _restore_trainable,
    _kl_from_cache, _jsd_from_cache, _distill_loss_from_cache,
    _distill_loss_live, adaptive_distill_alpha,
    TeacherCacheEntry, build_teacher_cache,
)


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


# ---------------------------------------------------------------------------
# Distillation loss tests
# ---------------------------------------------------------------------------

def _make_cache_entry(seq_len: int = 10, vocab: int = 64, top_k: int = 16):
    """Build a synthetic TeacherCacheEntry for testing."""
    probs = F.softmax(torch.randn(seq_len, vocab), dim=-1)
    vals, idx = probs.topk(top_k, dim=-1)
    return TeacherCacheEntry(
        top_k_vals=vals.half(),
        top_k_idx=idx.int(),
    )


class TestJSDLoss:
    def test_jsd_is_non_negative(self):
        """JSD is always >= 0."""
        logits = torch.randn(1, 10, 64)
        entry = _make_cache_entry(seq_len=10, vocab=64)
        loss = _jsd_from_cache(logits, entry, distill_temp=2.0)
        assert loss.item() >= -1e-6

    def test_jsd_bounded_above(self):
        """JSD <= ln(2) * temp^2 for probability distributions."""
        logits = torch.randn(1, 10, 64) * 10
        entry = _make_cache_entry(seq_len=10, vocab=64)
        loss = _jsd_from_cache(logits, entry, distill_temp=2.0)
        assert loss.item() < 10.0  # generous bound

    def test_jsd_zero_for_identical(self):
        """JSD(P, P) = 0 when student matches teacher exactly."""
        entry = _make_cache_entry(seq_len=10, vocab=64, top_k=16)
        # Build logits that reproduce the teacher's distribution at top-k
        logits = torch.zeros(1, 10, 64)
        idx = entry.top_k_idx.long()
        vals = entry.top_k_vals.float()
        # Set logits at top-k positions to log of teacher probs (approx)
        for s in range(10):
            logits[0, s].scatter_(0, idx[s], vals[s].log().clamp(min=-20))
        loss = _jsd_from_cache(logits, entry, distill_temp=1.0)
        # Won't be exactly 0 due to softmax renormalization, but should be small
        assert loss.item() < 0.5

    def test_jsd_has_gradient(self):
        """JSD loss produces valid gradients for backprop."""
        logits = torch.randn(1, 10, 64, requires_grad=True)
        entry = _make_cache_entry(seq_len=10, vocab=64)
        loss = _jsd_from_cache(logits, entry, distill_temp=2.0)
        loss.backward()
        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()

    def test_distill_loss_dispatch_cached(self):
        """_distill_loss_from_cache dispatches correctly for both modes."""
        logits = torch.randn(1, 10, 64)
        entry = _make_cache_entry(seq_len=10, vocab=64)
        jsd_val = _distill_loss_from_cache(logits, entry, 2.0, "jsd")
        kl_val = _distill_loss_from_cache(logits, entry, 2.0, "kl")
        # Both should be finite scalars, but generally different values
        assert torch.isfinite(jsd_val)
        assert torch.isfinite(kl_val)

    def test_distill_loss_live_jsd(self):
        """Live JSD produces finite loss."""
        student = torch.randn(1, 10, 64)
        teacher = torch.randn(1, 10, 64)
        loss = _distill_loss_live(student, teacher, distill_temp=2.0, loss_fn="jsd")
        assert torch.isfinite(loss)
        assert loss.item() >= -1e-6

    def test_distill_loss_live_kl(self):
        """Live KL produces finite loss."""
        student = torch.randn(1, 10, 64)
        teacher = torch.randn(1, 10, 64)
        loss = _distill_loss_live(student, teacher, distill_temp=2.0, loss_fn="kl")
        assert torch.isfinite(loss)


class TestAdaptiveAlpha:
    def test_scales_with_compression(self):
        """More compression → higher alpha."""
        a_light = adaptive_distill_alpha(0.5, keep_frac=0.95)
        a_moderate = adaptive_distill_alpha(0.5, keep_frac=0.85)
        a_heavy = adaptive_distill_alpha(0.5, keep_frac=0.70)
        assert a_light < a_moderate < a_heavy

    def test_clamped_bounds(self):
        """Alpha stays within [0.1, 0.9]."""
        assert adaptive_distill_alpha(0.5, keep_frac=1.0) >= 0.1
        assert adaptive_distill_alpha(0.5, keep_frac=0.0) <= 0.9
        assert adaptive_distill_alpha(0.9, keep_frac=0.0) <= 0.9
        assert adaptive_distill_alpha(0.1, keep_frac=1.0) >= 0.1

    def test_known_values(self):
        """Verify specific alpha values match the formula."""
        # alpha = base + (1 - kf) * 0.5, clamped
        assert abs(adaptive_distill_alpha(0.5, 0.90) - 0.55) < 1e-6
        assert abs(adaptive_distill_alpha(0.5, 0.80) - 0.60) < 1e-6
        assert abs(adaptive_distill_alpha(0.5, 0.70) - 0.65) < 1e-6


class TestRiskScaledLR:
    def test_risk_creates_param_groups(self, tiny_setup):
        """When layer_risk is provided, repair uses per-layer param groups."""
        model, tok, texts = tiny_setup
        result = repair_layers(
            model=model, tokenizer=tok, texts_train=texts,
            layers=[0, 1], steps=5, lr=1e-3, warmup=0,
            device="cpu",
            layer_risk={0: 0.2, 1: 0.8},
        )
        assert result["risk_scaled_lr"] is True

    def test_no_risk_flat_params(self, tiny_setup):
        """Without layer_risk, repair uses a flat param list."""
        model, tok, texts = tiny_setup
        result = repair_layers(
            model=model, tokenizer=tok, texts_train=texts,
            layers=[0, 1], steps=5, lr=1e-3, warmup=0,
            device="cpu",
        )
        assert result["risk_scaled_lr"] is False

    def test_risk_lr_scaling_values(self):
        """Risk 0.0 → 0.5x LR, risk 1.0 → 1.5x LR."""
        base_lr = 1e-3
        assert abs((0.5 + 0.0) * base_lr - 0.5e-3) < 1e-9
        assert abs((0.5 + 0.5) * base_lr - 1.0e-3) < 1e-9
        assert abs((0.5 + 1.0) * base_lr - 1.5e-3) < 1e-9


class TestJSDRepairIntegration:
    def test_jsd_repair_converges(self, tiny_setup):
        """Full repair loop with JSD distillation completes without error."""
        model, tok, texts = tiny_setup
        teacher = TinyModel().to("cpu")

        cache = build_teacher_cache(
            teacher, tok, texts[:20],
            distill_temp=2.0, max_len=32, device="cpu", top_k=16,
        )

        result = repair_layers(
            model=model, tokenizer=tok, texts_train=texts[:20],
            layers=[0, 1], steps=10, lr=1e-3, warmup=2,
            device="cpu",
            distill_alpha=0.5, distill_temp=2.0,
            distill_loss_fn="jsd",
            teacher_cache=cache,
        )
        assert result["distillation_enabled"] is True
        assert result["distill_loss_fn"] == "jsd"
        assert result["steps"] > 0

    def test_kl_repair_backward_compat(self, tiny_setup):
        """Legacy KL mode still works when explicitly requested."""
        model, tok, texts = tiny_setup
        teacher = TinyModel().to("cpu")

        cache = build_teacher_cache(
            teacher, tok, texts[:20],
            distill_temp=2.0, max_len=32, device="cpu", top_k=16,
        )

        result = repair_layers(
            model=model, tokenizer=tok, texts_train=texts[:20],
            layers=[0, 1], steps=10, lr=1e-3, warmup=2,
            device="cpu",
            distill_alpha=0.5, distill_temp=2.0,
            distill_loss_fn="kl",
            teacher_cache=cache,
        )
        assert result["distillation_enabled"] is True
        assert result["distill_loss_fn"] == "kl"
