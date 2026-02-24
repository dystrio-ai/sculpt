"""Configuration: EngineConfig dataclass + YAML loading + CLI overrides."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List

import torch
import yaml


CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
DEFAULT_CONFIG = CONFIG_DIR / "default.yaml"


@dataclass
class EngineConfig:
    model_id: str = "Qwen/Qwen2-0.5B"
    layers: List[int] = field(default_factory=lambda: [3])

    # compression
    block_size: int = 128
    keep_frac: float = 0.50

    # data
    max_len: int = 256
    n_texts_cal: int = 400
    n_texts_train: int = 2500
    n_texts_eval: int = 300
    max_eval_tokens: int = 40000

    # repair
    lr: float = 3e-4
    warmup: int = 100
    weight_decay: float = 0.01
    repair_steps: int = 2000

    # benchmark
    bench_texts: int = 200
    bench_warmup_iters: int = 20
    bench_iters: int = 80

    # runtime
    device: str = "cuda"
    dtype: str = "bf16"  # bf16 | fp16 | fp32
    seed: int = 0
    allow_tf32: bool = True


def resolve_dtype(dtype_str: str) -> torch.dtype:
    s = dtype_str.lower()
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16"):
        return torch.float16
    if s in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unknown dtype: {dtype_str}")


# ── YAML loading ─────────────────────────────────────────────────────────────


def _load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _from_yaml_dict(d: dict) -> EngineConfig:
    """Map a (possibly nested) YAML dict into a flat EngineConfig.

    Supports both flat layout (field names match dataclass) and the nested
    layout from configs/default.yaml.
    """
    # If the YAML already uses flat keys that match EngineConfig, just use them.
    # Otherwise, pull from nested sections.
    model = d.get("model", {})
    hw = d.get("hardware", {})
    comp = d.get("compression", {})
    cal = d.get("calibration", {})
    data = d.get("data", {})
    rep = d.get("repair", {})
    ev = d.get("eval", {})
    bench = d.get("bench", {})

    def _get(key, *sources, default=None):
        """First check top-level d[key], then each nested source for key."""
        if key in d:
            return d[key]
        for src in sources:
            if key in src:
                return src[key]
        return default

    layers = _get("layers", comp, default=[3])
    if isinstance(layers, tuple):
        layers = list(layers)

    _S = object()  # sentinel distinguishing "not found" from 0/False/""

    def _get_or(key, *sources, fallback_src=None, fallback_key=None, default=None):
        """Like _get but uses a sentinel so 0/False/"" aren't swallowed."""
        v = _get(key, *sources, default=_S)
        if v is not _S:
            return v
        if fallback_src is not None and fallback_key is not None:
            return fallback_src.get(fallback_key, default)
        return default

    return EngineConfig(
        model_id=_get_or(
            "model_id", model, fallback_src=model, fallback_key="name",
            default=EngineConfig.model_id,
        ),
        layers=layers,
        block_size=_get("block_size", comp, default=EngineConfig.block_size),
        keep_frac=_get("keep_frac", comp, default=EngineConfig.keep_frac),
        max_len=_get("max_len", cal, rep, ev, bench, default=EngineConfig.max_len),
        n_texts_cal=_get_or(
            "n_texts_cal", cal, fallback_src=cal, fallback_key="n_texts",
            default=EngineConfig.n_texts_cal,
        ),
        n_texts_train=_get("n_texts_train", data, default=EngineConfig.n_texts_train),
        n_texts_eval=_get("n_texts_eval", data, default=EngineConfig.n_texts_eval),
        max_eval_tokens=_get("max_eval_tokens", ev, default=EngineConfig.max_eval_tokens),
        lr=_get("lr", rep, default=EngineConfig.lr),
        warmup=_get("warmup", rep, default=EngineConfig.warmup),
        weight_decay=_get("weight_decay", rep, default=EngineConfig.weight_decay),
        repair_steps=_get_or(
            "repair_steps", fallback_src=rep, fallback_key="steps",
            default=EngineConfig.repair_steps,
        ),
        bench_texts=_get_or(
            "bench_texts", fallback_src=bench, fallback_key="n_texts",
            default=EngineConfig.bench_texts,
        ),
        bench_warmup_iters=_get_or(
            "bench_warmup_iters", bench, fallback_src=bench, fallback_key="warmup_iters",
            default=EngineConfig.bench_warmup_iters,
        ),
        bench_iters=_get_or(
            "bench_iters", bench, fallback_src=bench, fallback_key="iters",
            default=EngineConfig.bench_iters,
        ),
        device=_get("device", hw, default=EngineConfig.device),
        dtype=_get("dtype", hw, default=EngineConfig.dtype),
        seed=_get("seed", default=EngineConfig.seed),
        allow_tf32=_get("allow_tf32", hw, default=EngineConfig.allow_tf32),
    )


# ── CLI override helpers ─────────────────────────────────────────────────────


def deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge *overrides* into a copy of *base*."""
    out = copy.deepcopy(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def parse_cli_overrides(pairs: list[str]) -> dict:
    """Turn key=value pairs (optionally dotted) into a nested dict.

    Examples: ["keep_frac=0.6", "repair.lr=3e-5", "layers=[3,5]"]
    """
    out: dict[str, Any] = {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        if not value:
            raise ValueError(f"Override must be key=value, got {pair!r}")
        parts = key.split(".")
        d = out
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = _auto_cast(value)
    return out


def _auto_cast(v: str) -> Any:
    """Best-effort cast: bool → int → float → list → str."""
    if v.lower() in ("true", "yes"):
        return True
    if v.lower() in ("false", "no"):
        return False
    if v.lower() in ("null", "none"):
        return None
    if v.startswith("[") and v.endswith("]"):
        return [_auto_cast(x.strip()) for x in v[1:-1].split(",") if x.strip()]
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def load_config(
    config_path: str | Path | None = None,
    overrides: list[str] | None = None,
) -> EngineConfig:
    """Load YAML config, merge CLI overrides, return an EngineConfig."""
    raw = _load_yaml(config_path or DEFAULT_CONFIG)
    if overrides:
        raw = deep_merge(raw, parse_cli_overrides(overrides))
    return _from_yaml_dict(raw)
