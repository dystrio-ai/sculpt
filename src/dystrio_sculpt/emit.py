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


_SUMMARY_COLUMNS = [
    "name", "keep_frac", "ppl_w103", "ppl_ratio",
    "prefill_speedup", "decode_speedup", "compile_time_s",
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
) -> None:
    """Append one row to the root-level summary.csv (creates header on first call)."""
    csv_path = outdir / "summary.csv"
    write_header = not csv_path.exists()
    outdir.mkdir(parents=True, exist_ok=True)
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
            "compile_time_s": f"{compile_time_s:.1f}",
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
) -> Path:
    """Save a single frontier point: model weights, metrics, and manifest.

    *label* is a human-friendly name like ``frontier_0_conservative``.
    Returns the path to the emitted directory.
    """
    point_dir = outdir / label
    model_dir = point_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Patch config.intermediate_size to match actual (compressed) FFN width.
    # Without this, from_pretrained reconstructs the original (wider) shapes
    # and crashes on weight size mismatch.
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
    }
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
    )

    _log.info(
        "emitted %s: keep_frac=%.2f  ppl_w103=%.2f  speedup=%.2fx",
        label, keep_frac,
        metrics_out["ppl_w103_valid"], metrics_out["prefill_speedup"],
    )
    return point_dir
