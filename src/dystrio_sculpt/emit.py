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


_LATENCY_KEYS = [
    "prefill_latency_ms_p50", "prefill_latency_ms_p95", "prefill_latency_ms_p99",
    "prefill_latency_ms_mean", "prefill_latency_ms_std",
    "decode_ms_per_token_p50", "decode_ms_per_token_p95", "decode_ms_per_token_p99",
    "decode_ms_per_token_mean", "decode_ms_per_token_std",
]

_SUMMARY_COLUMNS = [
    "name", "keep_frac", "ppl_w103", "ppl_ratio",
    "prefill_speedup", "decode_speedup", "risk_score", "compile_time_s",
    "e2e_speedup_chat", "e2e_speedup_rag", "e2e_speedup_batch",
    "prefill_ms_p95", "decode_ms_per_tok_p95",
    "peak_compile_alloc_gb", "peak_bench_alloc_gb", "steady_state_alloc_gb",
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
    peak_compile_alloc_gb: Optional[float] = None,
    peak_bench_alloc_gb: Optional[float] = None,
    steady_state_alloc_gb: Optional[float] = None,
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
            "peak_compile_alloc_gb": _fmt(peak_compile_alloc_gb, "{:.2f}"),
            "peak_bench_alloc_gb": _fmt(peak_bench_alloc_gb, "{:.2f}"),
            "steady_state_alloc_gb": _fmt(steady_state_alloc_gb, "{:.2f}"),
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

    # VRAM: compile peaks, bench peaks, steady-state (bytes -> GiB)
    vram = {
        "peak_compile_alloc_gb": _bytes_to_gib(peak_cuda_allocated_compile_bytes),
        "peak_compile_reserved_gb": _bytes_to_gib(peak_cuda_reserved_compile_bytes),
        "peak_bench_alloc_gb": _bytes_to_gib(peak_cuda_allocated_bench_bytes),
        "peak_bench_reserved_gb": _bytes_to_gib(peak_cuda_reserved_bench_bytes),
        "steady_state_alloc_gb": _bytes_to_gib(cuda_allocated_end_bytes),
        "steady_state_reserved_gb": _bytes_to_gib(cuda_reserved_end_bytes),
    }

    metrics_out = {
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
    for k in _LATENCY_KEYS:
        val = metrics.get(k)
        if val is not None:
            metrics_out[k] = val
    for k, v in vram.items():
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
        prefill_ms_p95=metrics.get("prefill_latency_ms_p95"),
        decode_ms_per_tok_p95=metrics.get("decode_ms_per_token_p95"),
        peak_compile_alloc_gb=vram["peak_compile_alloc_gb"],
        peak_bench_alloc_gb=vram["peak_bench_alloc_gb"],
        steady_state_alloc_gb=vram["steady_state_alloc_gb"],
    )

    _log.info(
        "emitted %s: keep_frac=%.2f  ppl_w103=%.2f  speedup=%.2fx  e2e_rag=%.2fx",
        label, keep_frac,
        metrics_out["ppl_w103_valid"], metrics_out["prefill_speedup"],
        e2e_speedups.get("rag", 0.0),
    )
    return point_dir
