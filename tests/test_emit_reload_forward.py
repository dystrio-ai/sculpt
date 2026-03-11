"""Tests for emit + validate round-trip with a mock model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn


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


class TinyCausalLM(nn.Module):
    def __init__(self, dim: int = 16, n_layers: int = 1, vocab: int = 64):
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

    def save_pretrained(self, path: str):
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), str(Path(path) / "model.pt"))
        config_data = {"vocab_size": self.config.vocab_size}
        with open(str(Path(path) / "config.json"), "w") as f:
            json.dump(config_data, f)


class FakeTokenizer:
    def __call__(self, text, return_tensors="pt", truncation=True, max_length=256):
        ids = torch.randint(0, 64, (1, 10))
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def save_pretrained(self, path: str):
        Path(path).mkdir(parents=True, exist_ok=True)
        with open(str(Path(path) / "tokenizer_config.json"), "w") as f:
            json.dump({"model_type": "test"}, f)


class TinyCompressedCausalLM(TinyCausalLM):
    """A TinyCausalLM whose layer-0 MLP has been 'compressed' to a smaller FFN."""

    def __init__(self, dim: int = 16, compressed_ffn: int = 24, n_layers: int = 1, vocab: int = 64):
        super().__init__(dim=dim, n_layers=n_layers, vocab=vocab)
        mlp = self.model.layers[0].mlp
        mlp.gate_proj = nn.Linear(dim, compressed_ffn, bias=False)
        mlp.up_proj = nn.Linear(dim, compressed_ffn, bias=False)
        mlp.down_proj = nn.Linear(compressed_ffn, dim, bias=False)
        self.config.intermediate_size = dim * 2  # stale — original width


class TestEmitArtifacts:
    def test_emit_creates_expected_files(self, tmp_path):
        from dystrio_sculpt.emit import emit_frontier_point

        model = TinyCausalLM()
        tok = FakeTokenizer()

        point_dir = emit_frontier_point(
            model=model, tokenizer=tok,
            outdir=tmp_path, label="frontier_0_default",
            keep_frac=0.75,
            metrics={
                "ppl_w2_test": 12.5, "ppl_w103_valid": 14.0,
                "prefill_tokens_per_sec": 1000.0, "decode_tokens_per_sec": 500.0,
            },
            baseline_metrics={
                "ppl_w103_valid": 10.0, "prefill_tokens_per_sec": 800.0,
                "decode_tokens_per_sec": 400.0,
            },
            compile_report={"0": {"kept_blocks": 10}},
            config={"model_id": "test", "keep_frac": 0.75, "seed": 0},
            wall_time_s=42.0,
        )

        assert (point_dir / "model").exists()
        assert (point_dir / "metrics.json").exists()
        assert (point_dir / "compile_report.json").exists()
        assert (point_dir / "manifest.json").exists()

        metrics = json.loads((point_dir / "metrics.json").read_text())
        assert metrics["keep_frac"] == 0.75
        assert metrics["label"] == "frontier_0_default"
        assert metrics["ppl_ratio"] > 1.0

    def test_summary_csv_appended(self, tmp_path):
        from dystrio_sculpt.emit import emit_frontier_point, append_summary_csv

        model = TinyCausalLM()
        tok = FakeTokenizer()
        baseline = {
            "ppl_w103_valid": 10.0, "prefill_tokens_per_sec": 800.0,
            "decode_tokens_per_sec": 400.0,
        }

        for i in range(3):
            emit_frontier_point(
                model=model, tokenizer=tok,
                outdir=tmp_path, label=f"frontier_{i}_test",
                keep_frac=0.5 + i * 0.1,
                metrics={
                    "ppl_w2_test": 12.0, "ppl_w103_valid": 13.0,
                    "prefill_tokens_per_sec": 1000.0, "decode_tokens_per_sec": 500.0,
                },
                baseline_metrics=baseline,
                compile_report={},
                config={"model_id": "test", "keep_frac": 0.5 + i * 0.1, "seed": 0},
                wall_time_s=10.0 + i,
            )

        csv_path = tmp_path / "summary.csv"
        assert csv_path.exists()
        lines = csv_path.read_text().strip().split("\n")
        assert len(lines) == 4  # header + 3 rows

    def test_manifest_contains_policy(self, tmp_path):
        from dystrio_sculpt.emit import emit_frontier_point

        model = TinyCausalLM()
        tok = FakeTokenizer()

        policy_info = {"name": "ss4_lr1e-4_p8_s450", "stage_size": 4, "lr": 1e-4}
        emit_frontier_point(
            model=model, tokenizer=tok,
            outdir=tmp_path, label="frontier_0_default",
            keep_frac=0.8,
            metrics={
                "ppl_w2_test": 11.0, "ppl_w103_valid": 12.0,
                "prefill_tokens_per_sec": 900.0, "decode_tokens_per_sec": 450.0,
            },
            baseline_metrics={
                "ppl_w103_valid": 10.0, "prefill_tokens_per_sec": 800.0,
                "decode_tokens_per_sec": 400.0,
            },
            compile_report={},
            config={"model_id": "test", "policy": policy_info, "seed": 0},
            wall_time_s=30.0,
        )

        manifest = json.loads(
            (tmp_path / "frontier_0_default" / "manifest.json").read_text(),
        )
        assert "policy" in manifest
        assert manifest["policy"]["name"] == "ss4_lr1e-4_p8_s450"

    def test_config_intermediate_size_patched(self, tmp_path):
        """After emit, saved config.json has the compressed intermediate_size."""
        from dystrio_sculpt.emit import emit_frontier_point

        compressed_ffn = 24
        model = TinyCompressedCausalLM(dim=16, compressed_ffn=compressed_ffn)
        tok = FakeTokenizer()

        point_dir = emit_frontier_point(
            model=model, tokenizer=tok,
            outdir=tmp_path, label="frontier_0_default",
            keep_frac=0.75,
            metrics={
                "ppl_w2_test": 12.5, "ppl_w103_valid": 14.0,
                "prefill_tokens_per_sec": 1000.0, "decode_tokens_per_sec": 500.0,
            },
            baseline_metrics={
                "ppl_w103_valid": 10.0, "prefill_tokens_per_sec": 800.0,
                "decode_tokens_per_sec": 400.0,
            },
            compile_report={},
            config={"model_id": "test", "keep_frac": 0.75, "seed": 0},
            wall_time_s=10.0,
        )

        # Config on the in-memory model was patched
        assert model.config.intermediate_size == compressed_ffn

        # Manifest records both old and new
        manifest = json.loads((point_dir / "manifest.json").read_text())
        assert manifest["old_intermediate_size"] == 32  # dim * 2
        assert manifest["new_intermediate_size"] == compressed_ffn
