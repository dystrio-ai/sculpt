"""Tests for calibration corpus configuration and metadata recording."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


class TestCalibConfigDefaults:
    """Default CalibConfig must reproduce the original wikitext behavior."""

    def test_default_values(self):
        from dystrio_sculpt._data import (
            CalibConfig,
            DEFAULT_CALIB_DATASET,
            DEFAULT_CALIB_CONFIG,
            DEFAULT_CALIB_SPLIT,
            DEFAULT_CALIB_TEXT_FIELD,
        )

        cfg = CalibConfig()
        assert cfg.dataset == DEFAULT_CALIB_DATASET
        assert cfg.config == DEFAULT_CALIB_CONFIG
        assert cfg.split == DEFAULT_CALIB_SPLIT
        assert cfg.text_field == DEFAULT_CALIB_TEXT_FIELD
        assert cfg.num_samples is None
        assert cfg.seq_len is None
        assert cfg.seed == 0

    def test_default_is_wikitext(self):
        from dystrio_sculpt._data import CalibConfig

        cfg = CalibConfig()
        assert cfg.dataset == "wikitext"
        assert cfg.config == "wikitext-2-raw-v1"
        assert cfg.split == "train"

    def test_to_dict_keys(self):
        from dystrio_sculpt._data import CalibConfig

        d = CalibConfig().to_dict()
        expected_keys = {
            "calib_dataset", "calib_config", "calib_split",
            "calib_text_field", "calib_num_samples", "calib_seq_len", "calib_seed",
        }
        assert set(d.keys()) == expected_keys


class TestCalibConfigCustom:
    """Custom CalibConfig changes the dataset id recorded in metadata."""

    def test_custom_dataset(self):
        from dystrio_sculpt._data import CalibConfig

        cfg = CalibConfig(dataset="allenai/c4", config="en", split="train")
        d = cfg.to_dict()
        assert d["calib_dataset"] == "allenai/c4"
        assert d["calib_config"] == "en"

    def test_custom_num_samples_and_seed(self):
        from dystrio_sculpt._data import CalibConfig

        cfg = CalibConfig(num_samples=500, seed=42)
        d = cfg.to_dict()
        assert d["calib_num_samples"] == 500
        assert d["calib_seed"] == 42


class TestCalibMetadataRecording:
    """Calib params should flow into run_metadata.json."""

    def test_emit_run_metadata_includes_calib(self, tmp_path):
        from dystrio_sculpt.emit import emit_run_metadata
        from dystrio_sculpt._data import CalibConfig

        cfg = CalibConfig(dataset="allenai/c4", config="en", split="train", seed=99)
        emit_run_metadata(tmp_path, {
            "deterministic": True, "seed": 0, "dtype": "bf16",
            **cfg.to_dict(),
        })
        meta_path = tmp_path / "run_metadata.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["calib_dataset"] == "allenai/c4"
        assert meta["calib_config"] == "en"
        assert meta["calib_seed"] == 99

    def test_emit_run_metadata_default_calib(self, tmp_path):
        from dystrio_sculpt.emit import emit_run_metadata
        from dystrio_sculpt._data import CalibConfig

        cfg = CalibConfig()
        emit_run_metadata(tmp_path, {
            "deterministic": False, "seed": 0, "dtype": "bf16",
            **cfg.to_dict(),
        })
        meta = json.loads((tmp_path / "run_metadata.json").read_text())
        assert meta["calib_dataset"] == "wikitext"
        assert meta["calib_config"] == "wikitext-2-raw-v1"

    def test_emit_run_metadata_without_calib_keys(self, tmp_path):
        """Bench runs don't pass calib keys — should not crash."""
        from dystrio_sculpt.emit import emit_run_metadata

        emit_run_metadata(tmp_path, {"deterministic": False, "seed": 0, "dtype": "bf16"})
        meta = json.loads((tmp_path / "run_metadata.json").read_text())
        assert "calib_dataset" not in meta


class TestCalibDataHelpers:
    """Test internal data helpers used by calibration loading."""

    def test_collect_texts_filters_empty(self):
        from dystrio_sculpt._data import _collect_texts

        class FakeDS:
            def __init__(self, data):
                self._data = data
            def __len__(self):
                return len(self._data)
            def __getitem__(self, i):
                return self._data[i]

        ds = FakeDS([{"text": "hello"}, {"text": ""}, {"text": "  "}, {"text": "world"}])
        result = _collect_texts(ds, 10)
        assert result == ["hello", "world"]

    def test_collect_texts_respects_n(self):
        from dystrio_sculpt._data import _collect_texts

        class FakeDS:
            def __init__(self, data):
                self._data = data
            def __len__(self):
                return len(self._data)
            def __getitem__(self, i):
                return self._data[i]

        ds = FakeDS([{"text": f"line {i}"} for i in range(100)])
        result = _collect_texts(ds, 5)
        assert len(result) == 5

    def test_collect_texts_custom_field(self):
        from dystrio_sculpt._data import _collect_texts

        class FakeDS:
            def __init__(self, data):
                self._data = data
            def __len__(self):
                return len(self._data)
            def __getitem__(self, i):
                return self._data[i]

        ds = FakeDS([{"content": "hello"}, {"content": "world"}])
        result = _collect_texts(ds, 10, field="content")
        assert result == ["hello", "world"]

    def test_deterministic_sample_stable(self):
        from dystrio_sculpt._data import _deterministic_sample

        texts = [f"text_{i}" for i in range(100)]
        s1 = _deterministic_sample(texts, 10, seed=42)
        s2 = _deterministic_sample(texts, 10, seed=42)
        assert s1 == s2
        assert len(s1) == 10

    def test_deterministic_sample_different_seeds(self):
        from dystrio_sculpt._data import _deterministic_sample

        texts = [f"text_{i}" for i in range(100)]
        s1 = _deterministic_sample(texts, 10, seed=0)
        s2 = _deterministic_sample(texts, 10, seed=1)
        assert s1 != s2

    def test_deterministic_sample_n_greater_than_pool(self):
        from dystrio_sculpt._data import _deterministic_sample

        texts = ["a", "b", "c"]
        result = _deterministic_sample(texts, 100, seed=0)
        assert result == texts


class TestCalibConfigFrontierSearchWiring:
    """CalibConfig is accepted by FrontierSearch constructor."""

    def test_search_accepts_calib(self):
        from dystrio_sculpt._data import CalibConfig

        # Just verify import and constructor signature accepts calib kwarg
        from dystrio_sculpt.search import FrontierSearch
        import inspect
        sig = inspect.signature(FrontierSearch.__init__)
        assert "calib" in sig.parameters

    def test_engine_compile_accepts_calib(self):
        from dystrio_sculpt.engine import compile_model
        import inspect
        sig = inspect.signature(compile_model)
        assert "calib" in sig.parameters
