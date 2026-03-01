"""Tests for deterministic subset selection in _data.py."""

from __future__ import annotations

import pytest

from dystrio_sculpt._data import deterministic_subset


class TestDeterministicSubset:
    def test_same_seed_same_result(self):
        texts = [f"text_{i}" for i in range(500)]
        a = deterministic_subset(texts, 50, seed=42)
        b = deterministic_subset(texts, 50, seed=42)
        assert a == b

    def test_different_seed_different_result(self):
        texts = [f"text_{i}" for i in range(500)]
        a = deterministic_subset(texts, 50, seed=42)
        b = deterministic_subset(texts, 50, seed=99)
        assert a != b

    def test_correct_length(self):
        texts = [f"t_{i}" for i in range(200)]
        subset = deterministic_subset(texts, 30, seed=0)
        assert len(subset) == 30

    def test_full_list_when_n_exceeds_length(self):
        texts = [f"t_{i}" for i in range(10)]
        subset = deterministic_subset(texts, 100, seed=0)
        assert len(subset) == 10
        assert subset == list(texts)

    def test_subset_contains_only_originals(self):
        texts = [f"unique_{i}" for i in range(100)]
        subset = deterministic_subset(texts, 20, seed=7)
        for item in subset:
            assert item in texts

    def test_indices_are_sorted(self):
        """The returned subset preserves original ordering (sorted indices)."""
        texts = [f"item_{i:04d}" for i in range(300)]
        subset = deterministic_subset(texts, 50, seed=13)
        original_indices = [texts.index(t) for t in subset]
        assert original_indices == sorted(original_indices)

    def test_stability_across_calls(self):
        """Multiple calls with the same seed are stable (no global state leak)."""
        texts = [f"x_{i}" for i in range(1000)]
        results = [deterministic_subset(texts, 100, seed=5) for _ in range(5)]
        for r in results[1:]:
            assert r == results[0]
