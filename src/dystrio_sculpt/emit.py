"""Artifact emission: save compiled models, metrics, manifests, and summary CSV."""

from __future__ import annotations

import csv
import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from . import __version__
from .policy import E2E_PROFILES, compute_e2e_speedup
from ._bench import LATENCY_WARMUP, LATENCY_MEASURE

_log = logging.getLogger(__name__)


def _get_git_sha() -> Optional[str]:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def _bytes_to_gib(b: Optional[int]) -> Optional[float]:
    """Convert bytes to GiB, returning None if input is None."""
    if b is None:
        return None
    return round(b / (1024 ** 3), 2)


def _safe_pct(baseline: Optional[float], current: Optional[float]) -> Optional[float]:
    """100 * (baseline - current) / baseline, or None if inputs are missing/zero."""
    if baseline is None or current is None or baseline == 0:
        return None
    return round(100.0 * (baseline - current) / baseline, 2)


def _safe_ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return round(numerator / denominator, 3)


def _safe_throughput_gain_pct(
    current: Optional[float], baseline: Optional[float],
) -> Optional[float]:
    """100 * (current - baseline) / baseline."""
    if baseline is None or current is None or baseline == 0:
        return None
    return round(100.0 * (current - baseline) / baseline, 2)


def _gpu_hour_reduction_pct(e2e_speedup: Optional[float]) -> Optional[float]:
    """100 * (1 - 1/speedup)."""
    if e2e_speedup is None or e2e_speedup <= 0:
        return None
    return round(100.0 * (1.0 - 1.0 / e2e_speedup), 2)


_LATENCY_KEYS = [
    "prefill_latency_ms_p50", "prefill_latency_ms_p95", "prefill_latency_ms_p99",
    "prefill_latency_ms_mean", "prefill_latency_ms_std",
    "decode_ms_per_token_p50", "decode_ms_per_token_p95", "decode_ms_per_token_p99",
    "decode_ms_per_token_mean", "decode_ms_per_token_std",
]

_BASELINE_LATENCY_KEYS = [f"baseline_{k}" for k in _LATENCY_KEYS]

_DERIVED_KEYS = [
    "prefill_p95_latency_ratio", "decode_p95_latency_ratio",
    "prefill_p95_latency_improvement_pct", "decode_p95_latency_improvement_pct",
    "prefill_throughput_gain_pct", "decode_throughput_gain_pct",
    "gpu_hour_reduction_chat_pct", "gpu_hour_reduction_rag_pct",
    "gpu_hour_reduction_batch_pct",
    "baseline_steady_state_alloc_gb", "steady_state_memory_reduction_pct",
    "weights_memory_reduction_pct",
    "repair_steps_per_stage", "compile_minutes",
]

_SUMMARY_COLUMNS = [
    "name", "keep_frac", "ppl_w103", "ppl_ratio",
    "prefill_speedup", "decode_speedup", "risk_score", "compile_time_s",
    "e2e_speedup_chat", "e2e_speedup_rag", "e2e_speedup_batch",
    "prefill_ms_p95", "decode_ms_per_tok_p95",
    "num_params", "weights_gb",
    "peak_compile_alloc_gb", "peak_bench_alloc_gb", "steady_state_alloc_gb",
    # whitepaper-grade additions
    "baseline_prefill_ms_p95", "baseline_decode_ms_per_tok_p95",
    "prefill_p95_latency_improvement_pct", "decode_p95_latency_improvement_pct",
    "prefill_throughput_gain_pct", "decode_throughput_gain_pct",
    "gpu_hour_reduction_rag_pct",
    "weights_memory_reduction_pct", "steady_state_memory_reduction_pct",
    "compile_minutes",
]


def append_summary_csv(
    outdir: Path,
    name: str,
    keep_frac: float,
    ppl_w103: float,
    baseline_ppl_w103: float,
    prefill_speedup: float,
    decode_speedup: float,
    compile_time_s: float,
    risk_score: float = 0.0,
    e2e_speedup_chat: Optional[float] = None,
    e2e_speedup_rag: Optional[float] = None,
    e2e_speedup_batch: Optional[float] = None,
    prefill_ms_p95: Optional[float] = None,
    decode_ms_per_tok_p95: Optional[float] = None,
    num_params: Optional[int] = None,
    weights_gb: Optional[float] = None,
    peak_compile_alloc_gb: Optional[float] = None,
    peak_bench_alloc_gb: Optional[float] = None,
    steady_state_alloc_gb: Optional[float] = None,
    baseline_prefill_ms_p95: Optional[float] = None,
    baseline_decode_ms_per_tok_p95: Optional[float] = None,
    prefill_p95_latency_improvement_pct: Optional[float] = None,
    decode_p95_latency_improvement_pct: Optional[float] = None,
    prefill_throughput_gain_pct: Optional[float] = None,
    decode_throughput_gain_pct: Optional[float] = None,
    gpu_hour_reduction_rag_pct: Optional[float] = None,
    weights_memory_reduction_pct: Optional[float] = None,
    steady_state_memory_reduction_pct: Optional[float] = None,
    compile_minutes: Optional[float] = None,
) -> None:
    """Append one row to the root-level summary.csv (creates header on first call)."""
    csv_path = outdir / "summary.csv"
    write_header = not csv_path.exists()
    outdir.mkdir(parents=True, exist_ok=True)

    def _fmt(val, fmt_str):
        return fmt_str.format(val) if val is not None else ""

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_SUMMARY_COLUMNS)
        if write_header:
            writer.writeheader()
        ppl_ratio = ppl_w103 / max(1e-9, baseline_ppl_w103)
        writer.writerow({
            "name": name,
            "keep_frac": f"{keep_frac:.3f}",
            "ppl_w103": f"{ppl_w103:.4f}",
            "ppl_ratio": f"{ppl_ratio:.4f}",
            "prefill_speedup": f"{prefill_speedup:.3f}",
            "decode_speedup": f"{decode_speedup:.3f}",
            "risk_score": f"{risk_score:.4f}",
            "compile_time_s": f"{compile_time_s:.1f}",
            "e2e_speedup_chat": _fmt(e2e_speedup_chat, "{:.3f}"),
            "e2e_speedup_rag": _fmt(e2e_speedup_rag, "{:.3f}"),
            "e2e_speedup_batch": _fmt(e2e_speedup_batch, "{:.3f}"),
            "prefill_ms_p95": _fmt(prefill_ms_p95, "{:.1f}"),
            "decode_ms_per_tok_p95": _fmt(decode_ms_per_tok_p95, "{:.3f}"),
            "num_params": num_params if num_params is not None else "",
            "weights_gb": _fmt(weights_gb, "{:.3f}"),
            "peak_compile_alloc_gb": _fmt(peak_compile_alloc_gb, "{:.2f}"),
            "peak_bench_alloc_gb": _fmt(peak_bench_alloc_gb, "{:.2f}"),
            "steady_state_alloc_gb": _fmt(steady_state_alloc_gb, "{:.2f}"),
            "baseline_prefill_ms_p95": _fmt(baseline_prefill_ms_p95, "{:.1f}"),
            "baseline_decode_ms_per_tok_p95": _fmt(baseline_decode_ms_per_tok_p95, "{:.3f}"),
            "prefill_p95_latency_improvement_pct": _fmt(prefill_p95_latency_improvement_pct, "{:.1f}"),
            "decode_p95_latency_improvement_pct": _fmt(decode_p95_latency_improvement_pct, "{:.1f}"),
            "prefill_throughput_gain_pct": _fmt(prefill_throughput_gain_pct, "{:.1f}"),
            "decode_throughput_gain_pct": _fmt(decode_throughput_gain_pct, "{:.1f}"),
            "gpu_hour_reduction_rag_pct": _fmt(gpu_hour_reduction_rag_pct, "{:.1f}"),
            "weights_memory_reduction_pct": _fmt(weights_memory_reduction_pct, "{:.1f}"),
            "steady_state_memory_reduction_pct": _fmt(steady_state_memory_reduction_pct, "{:.1f}"),
            "compile_minutes": _fmt(compile_minutes, "{:.1f}"),
        })


def emit_frontier_point(
    model,
    tokenizer,
    outdir: Path,
    label: str,
    keep_frac: float,
    metrics: Dict[str, float],
    baseline_metrics: Dict[str, float],
    compile_report: Dict[str, Any],
    config: Dict[str, Any],
    wall_time_s: float,
    pilot_report: Optional[Dict[str, Any]] = None,
    risk_score: float = 0.0,
    peak_cuda_allocated_compile_bytes: Optional[int] = None,
    peak_cuda_reserved_compile_bytes: Optional[int] = None,
    peak_cuda_allocated_bench_bytes: Optional[int] = None,
    peak_cuda_reserved_bench_bytes: Optional[int] = None,
    cuda_allocated_end_bytes: Optional[int] = None,
    cuda_reserved_end_bytes: Optional[int] = None,
    num_params: Optional[int] = None,
    weights_bytes: Optional[int] = None,
    baseline_num_params: Optional[int] = None,
    baseline_weights_bytes: Optional[int] = None,
) -> Path:
    """Save a single frontier point: model weights, metrics, and manifest.

    *label* is a human-friendly name like ``frontier_0_conservative``.
    Returns the path to the emitted directory.
    """
    point_dir = outdir / label
    model_dir = point_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Patch config.intermediate_size to match actual (compressed) FFN width.
    old_intermediate = getattr(model.config, "intermediate_size", None)
    new_intermediate = None
    if hasattr(model, "model") and hasattr(model.model, "layers") and len(model.model.layers) > 0:
        gate = model.model.layers[0].mlp.gate_proj
        new_intermediate = gate.out_features if hasattr(gate, "out_features") else gate.weight.shape[0]
    if new_intermediate is not None and old_intermediate != new_intermediate:
        _log.info(
            "patching config.intermediate_size: %s -> %s",
            old_intermediate, new_intermediate,
        )
        model.config.intermediate_size = int(new_intermediate)
        if hasattr(model.config, "text_config") and model.config.text_config is not None:
            model.config.text_config.intermediate_size = int(new_intermediate)

    _log.info("saving model to %s", model_dir)
    model.save_pretrained(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))

    base_prefill = baseline_metrics.get("prefill_tokens_per_sec", 1.0)
    base_decode = baseline_metrics.get("decode_tokens_per_sec", 1.0)
    base_ppl = baseline_metrics.get("ppl_w103_valid", 1.0)

    prefill_speedup = metrics.get("prefill_tokens_per_sec", 0.0) / max(1e-9, base_prefill)
    decode_speedup = metrics.get("decode_tokens_per_sec", 0.0) / max(1e-9, base_decode)

    # E2E workload-modeled speedups
    e2e_speedups = {}
    for profile_name, profile in E2E_PROFILES.items():
        e2e_speedups[profile_name] = round(
            compute_e2e_speedup(prefill_speedup, decode_speedup, profile["P"], profile["D"]),
            3,
        )

    # Weights-only memory (deterministic, runtime-independent)
    weights_gb = round(weights_bytes / (1024 ** 3), 6) if weights_bytes is not None else None
    baseline_weights_gb = round(baseline_weights_bytes / (1024 ** 3), 6) if baseline_weights_bytes is not None else None

    # VRAM: compile peaks, bench peaks, steady-state (bytes -> GiB)
    vram = {
        "num_params": num_params,
        "weights_gb": weights_gb,
        "baseline_num_params": baseline_num_params,
        "baseline_weights_gb": baseline_weights_gb,
        "peak_compile_alloc_gb": _bytes_to_gib(peak_cuda_allocated_compile_bytes),
        "peak_compile_reserved_gb": _bytes_to_gib(peak_cuda_reserved_compile_bytes),
        "peak_bench_alloc_gb": _bytes_to_gib(peak_cuda_allocated_bench_bytes),
        "peak_bench_reserved_gb": _bytes_to_gib(peak_cuda_reserved_bench_bytes),
        "steady_state_alloc_gb": _bytes_to_gib(cuda_allocated_end_bytes),
        "steady_state_reserved_gb": _bytes_to_gib(cuda_reserved_end_bytes),
    }

    # ── Baseline latency + steady-state memory ────────────────────────────────
    base_prefill_p95 = baseline_metrics.get("prefill_latency_ms_p95")
    base_decode_p95 = baseline_metrics.get("decode_ms_per_token_p95")
    sculpt_prefill_p95 = metrics.get("prefill_latency_ms_p95")
    sculpt_decode_p95 = metrics.get("decode_ms_per_token_p95")

    baseline_steady_bytes = baseline_metrics.get("cuda_allocated_baseline_bytes")
    baseline_steady_gb = _bytes_to_gib(baseline_steady_bytes)

    # ── Derived metrics (whitepaper-grade) ────────────────────────────────────
    derived: Dict[str, Optional[float]] = {}

    derived["prefill_p95_latency_ratio"] = _safe_ratio(base_prefill_p95, sculpt_prefill_p95)
    derived["decode_p95_latency_ratio"] = _safe_ratio(base_decode_p95, sculpt_decode_p95)
    derived["prefill_p95_latency_improvement_pct"] = _safe_pct(base_prefill_p95, sculpt_prefill_p95)
    derived["decode_p95_latency_improvement_pct"] = _safe_pct(base_decode_p95, sculpt_decode_p95)

    derived["prefill_throughput_gain_pct"] = _safe_throughput_gain_pct(
        metrics.get("prefill_tokens_per_sec"), base_prefill,
    )
    derived["decode_throughput_gain_pct"] = _safe_throughput_gain_pct(
        metrics.get("decode_tokens_per_sec"), base_decode,
    )

    derived["gpu_hour_reduction_chat_pct"] = _gpu_hour_reduction_pct(e2e_speedups.get("chat"))
    derived["gpu_hour_reduction_rag_pct"] = _gpu_hour_reduction_pct(e2e_speedups.get("rag"))
    derived["gpu_hour_reduction_batch_pct"] = _gpu_hour_reduction_pct(e2e_speedups.get("batch"))

    derived["baseline_steady_state_alloc_gb"] = baseline_steady_gb
    derived["steady_state_memory_reduction_pct"] = _safe_pct(
        baseline_steady_gb, vram.get("steady_state_alloc_gb"),
    )
    derived["weights_memory_reduction_pct"] = _safe_pct(baseline_weights_gb, weights_gb)

    total_repair_steps = config.get("total_repair_steps", 0)
    n_stages = max(1, len(config.get("stage_stats", [])))
    derived["repair_steps_per_stage"] = round(total_repair_steps / n_stages, 1)
    derived["compile_minutes"] = round(wall_time_s / 60.0, 1)

    # ── Build metrics.json ────────────────────────────────────────────────────
    metrics_out: Dict[str, Any] = {
        "keep_frac": keep_frac,
        "label": label,
        "ppl_w2_test": round(metrics.get("ppl_w2_test", 0.0), 4),
        "ppl_w103_valid": round(metrics.get("ppl_w103_valid", 0.0), 4),
        "ppl_ratio": round(metrics.get("ppl_w103_valid", 0.0) / max(1e-9, base_ppl), 4),
        "prefill_tokens_per_sec": round(metrics.get("prefill_tokens_per_sec", 0.0), 1),
        "decode_tokens_per_sec": round(metrics.get("decode_tokens_per_sec", 0.0), 1),
        "prefill_speedup": round(prefill_speedup, 3),
        "decode_speedup": round(decode_speedup, 3),
        "baseline_ppl_w103": round(base_ppl, 4),
        "baseline_prefill_tps": round(base_prefill, 1),
        "baseline_decode_tps": round(base_decode, 1),
        "risk_score": round(risk_score, 4),
        "e2e_speedup_chat": e2e_speedups.get("chat"),
        "e2e_speedup_rag": e2e_speedups.get("rag"),
        "e2e_speedup_batch": e2e_speedups.get("batch"),
    }

    # Sculpted model latency percentiles
    for k in _LATENCY_KEYS:
        val = metrics.get(k)
        if val is not None:
            metrics_out[k] = val

    # Baseline latency percentiles
    for k in _LATENCY_KEYS:
        val = baseline_metrics.get(k)
        if val is not None:
            metrics_out[f"baseline_{k}"] = val

    # VRAM GiB fields
    for k, v in vram.items():
        if v is not None:
            metrics_out[k] = v

    # Derived whitepaper-grade fields
    for k, v in derived.items():
        if v is not None:
            metrics_out[k] = v

    _write_json(point_dir / "metrics.json", metrics_out)
    _write_json(point_dir / "compile_report.json", compile_report)

    try:
        import transformers
        transformers_version = transformers.__version__
    except Exception:
        transformers_version = "unknown"

    manifest = {
        "dystrio_sculpt_version": __version__,
        "model_id": config.get("model_id", ""),
        "keep_frac": keep_frac,
        "label": label,
        "block_size": config.get("block_size", 128),
        "seed": config.get("seed", 0),
        "deterministic": config.get("deterministic", False),
        "device": config.get("device", "cuda"),
        "dtype": config.get("dtype", "bf16"),
        "selector": config.get("selector", "structural"),
        "num_layers": config.get("num_layers", 0),
        "layers_compressed": config.get("layers_compressed", 0),
        "policy": config.get("policy", {}),
        "total_repair_steps": config.get("total_repair_steps", 0),
        "risk_score": round(risk_score, 4),
        "old_intermediate_size": old_intermediate,
        "new_intermediate_size": new_intermediate if new_intermediate is not None else old_intermediate,
        "compile_wall_time_s": round(wall_time_s, 2),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "cuda_device": (
            torch.cuda.get_device_name() if torch.cuda.is_available() else None
        ),
        "transformers_version": transformers_version,
        "git_sha": _get_git_sha(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
    }
    if pilot_report is not None:
        manifest["pilot_report"] = pilot_report
    _write_json(point_dir / "manifest.json", manifest)

    # Append to incremental summary.csv
    append_summary_csv(
        outdir=outdir,
        name=label,
        keep_frac=keep_frac,
        ppl_w103=metrics.get("ppl_w103_valid", 0.0),
        baseline_ppl_w103=base_ppl,
        prefill_speedup=prefill_speedup,
        decode_speedup=decode_speedup,
        compile_time_s=wall_time_s,
        risk_score=risk_score,
        e2e_speedup_chat=e2e_speedups.get("chat"),
        e2e_speedup_rag=e2e_speedups.get("rag"),
        e2e_speedup_batch=e2e_speedups.get("batch"),
        prefill_ms_p95=sculpt_prefill_p95,
        decode_ms_per_tok_p95=sculpt_decode_p95,
        num_params=num_params,
        weights_gb=weights_gb,
        peak_compile_alloc_gb=vram["peak_compile_alloc_gb"],
        peak_bench_alloc_gb=vram["peak_bench_alloc_gb"],
        steady_state_alloc_gb=vram["steady_state_alloc_gb"],
        baseline_prefill_ms_p95=base_prefill_p95,
        baseline_decode_ms_per_tok_p95=base_decode_p95,
        prefill_p95_latency_improvement_pct=derived.get("prefill_p95_latency_improvement_pct"),
        decode_p95_latency_improvement_pct=derived.get("decode_p95_latency_improvement_pct"),
        prefill_throughput_gain_pct=derived.get("prefill_throughput_gain_pct"),
        decode_throughput_gain_pct=derived.get("decode_throughput_gain_pct"),
        gpu_hour_reduction_rag_pct=derived.get("gpu_hour_reduction_rag_pct"),
        weights_memory_reduction_pct=derived.get("weights_memory_reduction_pct"),
        steady_state_memory_reduction_pct=derived.get("steady_state_memory_reduction_pct"),
        compile_minutes=derived.get("compile_minutes"),
    )

    _log.info(
        "emitted %s: keep_frac=%.2f  ppl_w103=%.2f  speedup=%.2fx  e2e_rag=%.2fx",
        label, keep_frac,
        metrics_out["ppl_w103_valid"], metrics_out["prefill_speedup"],
        e2e_speedups.get("rag", 0.0),
    )
    return point_dir


# ── Run provenance ────────────────────────────────────────────────────────────

def emit_run_metadata(
    outdir: Path,
    config: Dict[str, Any],
) -> None:
    """Write run_metadata.json, gpu_info.txt, and pip_freeze.txt to outdir.

    Failures are logged but never crash the run.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        import transformers
        transformers_version = transformers.__version__
    except Exception:
        transformers_version = "unknown"

    gpu_name: Optional[str] = None
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name()
        except Exception:
            pass

    metadata = {
        "git_commit": _get_git_sha(),
        "torch_version": torch.__version__,
        "transformers_version": transformers_version,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": gpu_name,
        "deterministic_flag": config.get("deterministic", False),
        "seed": config.get("seed", 0),
        "dtype": config.get("dtype", "bf16"),
        "tf32_enabled": getattr(torch.backends.cuda.matmul, "allow_tf32", None),
        "warmup_iters": LATENCY_WARMUP,
        "measure_iters": LATENCY_MEASURE,
        "decode_steps": 32,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    # Calibration corpus provenance (written by sculpt, absent for bench)
    for ck in (
        "calib_dataset", "calib_config", "calib_split",
        "calib_text_field", "calib_num_samples", "calib_seq_len", "calib_seed",
    ):
        if ck in config:
            metadata[ck] = config[ck]
    try:
        _write_json(outdir / "run_metadata.json", metadata)
    except Exception as exc:
        _log.warning("failed to write run_metadata.json: %s", exc)

    # gpu_info.txt
    try:
        r = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0:
            (outdir / "gpu_info.txt").write_text(r.stdout)
    except Exception:
        _log.debug("nvidia-smi not available")

    # pip_freeze.txt
    try:
        import sys
        r = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode == 0:
            (outdir / "pip_freeze.txt").write_text(r.stdout)
    except Exception:
        _log.debug("pip freeze not available")
