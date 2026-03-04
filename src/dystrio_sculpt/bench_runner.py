"""Benchmark runner: per-model per-workload orchestration."""

from __future__ import annotations

import csv
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from ._bench import (
    bench_prefill_tps,
    bench_decode_tps,
    bench_prefill_latency_ms,
    bench_decode_latency_ms,
    bench_ttft_per_prompt,
    compute_latency_percentiles,
)
from .emit import _bytes_to_gib, _write_json, emit_run_metadata
from .prompt_packs import load_prompt_pack, prompt_pack_hash

_log = logging.getLogger(__name__)

MAX_LEN = 2048
BENCH_BATCH = 32
BENCH_WARMUP = 10
BENCH_ITERS = 40
DECODE_STEPS = 64
DECODE_WARMUP = 3
DECODE_ITERS = 10
PER_PROMPT_WARMUP = 5

BENCHMARKS_CSV_COLUMNS = [
    "model_id", "workload", "num_prompts",
    "ppl_wikitext", "ppl_ratio",
    "prefill_tokens_per_sec", "decode_tokens_per_sec",
    # Microbench latency (batched iteration percentiles — internal reference)
    "microbench_prefill_ms_p50", "microbench_prefill_ms_p95", "microbench_prefill_ms_p99",
    "microbench_decode_ms_per_tok_p50", "microbench_decode_ms_per_tok_p95",
    "microbench_decode_ms_per_tok_p99",
    # Request-level latency (per-prompt, publishable)
    "first_decode_step_ms_p50", "first_decode_step_ms_p95", "first_decode_step_ms_p99",
    "ttft_ms_p50", "ttft_ms_p95", "ttft_ms_p99",
    "prefill_ms_p50", "prefill_ms_p95", "prefill_ms_p99",
    # Memory: deterministic (weights-only) and runtime
    "num_params", "weights_gb", "cold_alloc_gb",
    "peak_alloc_gb", "steady_state_alloc_gb",
    "errors_skipped_prompts",
]

PER_PROMPT_CSV_COLUMNS = [
    "id", "prompt_tokens", "max_new_tokens",
    "prefill_ms", "first_decode_step_ms", "ttft_ms",
    "is_warmup", "error",
]


def _compute_model_weight_stats(model: torch.nn.Module) -> Dict[str, Any]:
    """Return num_params (int) and weights_gb (float GiB) for a model."""
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return {
        "num_params": sum(p.numel() for p in model.parameters()),
        "weights_gb": round(total_bytes / (1024 ** 3), 6),
    }


def sanitize_model_id(model_id: str) -> str:
    """Replace ``/`` with ``__`` and remaining non-alphanumeric chars with ``_``."""
    s = model_id.replace("/", "__")
    return re.sub(r"[^a-zA-Z0-9_]", "_", s)


def model_shortname(model_id: str) -> str:
    """Derive a display-friendly short name from a model id."""
    base = model_id.rsplit("/", 1)[-1].lower()
    for tag in ("baseline", "conservative", "balanced", "aggressive"):
        if tag in base:
            return tag
    parts = base.replace("-", "_").split("_")
    return "_".join(parts[:3]) if len(parts) > 3 else base


# ── model loading ─────────────────────────────────────────────────────────────

def _load_model(model_id: str, device: str, dtype_str: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map.get(dtype_str, torch.bfloat16)

    _log.info("loading %s  dtype=%s  device=%s", model_id, dtype_str, device)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, device_map=device, trust_remote_code=True,
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


# ── wikitext workload ─────────────────────────────────────────────────────────

def _run_wikitext(
    model, tokenizer, device: str,
    max_len: int = 256, max_eval_tokens: int = 40_000,
) -> Dict[str, Any]:
    from datasets import load_dataset
    from ._eval import eval_perplexity

    ds = load_dataset("wikitext", "wikitext-103-v1", split="validation")
    texts = [t for t in ds["text"] if len(t.strip()) > 50][:500]
    ppl = eval_perplexity(model, tokenizer, texts, max_len, device, max_eval_tokens)
    return {"ppl_wikitext": round(ppl, 4), "num_prompts": len(texts)}


# ── prompt-based workloads ────────────────────────────────────────────────────

def _run_prompt_workload(
    model, tokenizer,
    prompts: List[Dict[str, Any]],
    device: str,
    workload: str,
    max_len: int = MAX_LEN,
) -> Dict[str, Any]:
    texts = [p["prompt"] for p in prompts]
    batch_texts = texts[:BENCH_BATCH]
    decode_text = texts[0] if texts else ""

    # Microbench throughput (aggregate, existing functions)
    prefill_tps = bench_prefill_tps(
        model, tokenizer, batch_texts, max_len, device, BENCH_WARMUP, BENCH_ITERS,
    )
    decode_tps = bench_decode_tps(
        model, tokenizer, decode_text, max_len, device,
        DECODE_STEPS, DECODE_WARMUP, DECODE_ITERS,
    )

    # Microbench latency percentiles (batched iteration — internal reference)
    pf_pct = compute_latency_percentiles(
        bench_prefill_latency_ms(model, tokenizer, batch_texts, max_len, device),
    )
    dc_pct = compute_latency_percentiles(
        bench_decode_latency_ms(model, tokenizer, decode_text, max_len, device),
    )

    # Per-prompt request-level latency (publishable, warmup-excluded)
    ttft_results = bench_ttft_per_prompt(
        model, tokenizer, prompts, max_len, device, warmup=PER_PROMPT_WARMUP,
    )
    measured = [r for r in ttft_results if not r.get("is_warmup") and not r.get("error")]
    errors = sum(1 for r in ttft_results if r.get("error") and not r.get("is_warmup"))

    fd_vals = [r["first_decode_step_ms"] for r in measured if r["first_decode_step_ms"] is not None]
    ttft_vals = [r["ttft_ms"] for r in measured if r["ttft_ms"] is not None]
    pf_req_vals = [r["prefill_ms"] for r in measured if r["prefill_ms"] is not None]

    fd_pct = compute_latency_percentiles(fd_vals)
    ttft_pct = compute_latency_percentiles(ttft_vals)
    pf_req_pct = compute_latency_percentiles(pf_req_vals)

    peak_gb = None
    ss_gb = None
    if torch.cuda.is_available():
        peak_gb = _bytes_to_gib(torch.cuda.max_memory_allocated())
        ss_gb = _bytes_to_gib(torch.cuda.memory_allocated())

    num_measured = len(prompts) - PER_PROMPT_WARMUP

    metrics: Dict[str, Any] = {
        "workload": workload,
        "num_prompts": num_measured,
        "prefill_tokens_per_sec": round(prefill_tps, 1),
        "decode_tokens_per_sec": round(decode_tps, 1),
        # Microbench (internal reference, not publishable)
        "microbench_prefill_ms_p50": pf_pct.get("p50"),
        "microbench_prefill_ms_p95": pf_pct.get("p95"),
        "microbench_prefill_ms_p99": pf_pct.get("p99"),
        "microbench_decode_ms_per_tok_p50": dc_pct.get("p50"),
        "microbench_decode_ms_per_tok_p95": dc_pct.get("p95"),
        "microbench_decode_ms_per_tok_p99": dc_pct.get("p99"),
        # Request-level (per-prompt, publishable)
        "first_decode_step_ms_p50": fd_pct.get("p50"),
        "first_decode_step_ms_p95": fd_pct.get("p95"),
        "first_decode_step_ms_p99": fd_pct.get("p99"),
        "ttft_ms_p50": ttft_pct.get("p50"),
        "ttft_ms_p95": ttft_pct.get("p95"),
        "ttft_ms_p99": ttft_pct.get("p99"),
        "prefill_ms_p50": pf_req_pct.get("p50"),
        "prefill_ms_p95": pf_req_pct.get("p95"),
        "prefill_ms_p99": pf_req_pct.get("p99"),
        "peak_alloc_gb": peak_gb,
        "steady_state_alloc_gb": ss_gb,
        "errors_skipped_prompts": errors,
    }
    return {"metrics": metrics, "per_prompt": ttft_results}


# ── per-model driver ──────────────────────────────────────────────────────────

def _resolve_prompt_pack(prompts_dir: Optional[Path], workload: str) -> Optional[Path]:
    if prompts_dir is None:
        return None
    exact = prompts_dir / f"{workload}.jsonl"
    if exact.exists():
        return exact
    for p in sorted(prompts_dir.glob(f"{workload}_*.jsonl")):
        return p
    return None


def bench_model(
    model_id: str,
    workloads: List[str],
    prompts_dir: Optional[Path],
    results_dir: Path,
    device: str = "cuda",
    dtype_str: str = "bf16",
    seed: int = 0,
    deterministic: bool = False,
) -> Dict[str, Dict[str, Any]]:
    from .engine import setup_determinism
    setup_determinism(seed, deterministic)

    safe_id = sanitize_model_id(model_id)
    model_root = results_dir / safe_id
    model, tokenizer = _load_model(model_id, device, dtype_str)

    weight_stats = _compute_model_weight_stats(model)

    cold_alloc_gb: Optional[float] = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        cold_alloc_gb = round(torch.cuda.memory_allocated() / (1024 ** 3), 6)

    all_metrics: Dict[str, Dict[str, Any]] = {}

    for wl in workloads:
        _log.info("[bench] %s / %s", model_id, wl)
        wl_dir = model_root / wl
        wl_dir.mkdir(parents=True, exist_ok=True)

        try:
            if wl == "wikitext":
                result = _run_wikitext(model, tokenizer, device)
                metrics: Dict[str, Any] = {
                    "model_id": model_id, "workload": wl, **result,
                    **weight_stats,
                    "cold_alloc_gb": cold_alloc_gb,
                }
                _write_json(wl_dir / "metrics.json", metrics)
                all_metrics[wl] = metrics
            else:
                pack_path = _resolve_prompt_pack(prompts_dir, wl)
                if pack_path is None:
                    _log.warning("no prompt pack for %s — skipping", wl)
                    continue
                prompts = load_prompt_pack(pack_path)
                pack_h = prompt_pack_hash(pack_path)
                result = _run_prompt_workload(model, tokenizer, prompts, device, wl)
                metrics = {
                    "model_id": model_id,
                    **result["metrics"],
                    **weight_stats,
                    "cold_alloc_gb": cold_alloc_gb,
                }
                _write_json(wl_dir / "metrics.json", metrics)

                if result["per_prompt"]:
                    _write_per_prompt_csv(wl_dir / "per_prompt.csv", result["per_prompt"])

                _write_workload_metadata(
                    wl_dir, model_id, wl, dtype_str, device,
                    seed, deterministic, pack_h,
                )
                all_metrics[wl] = metrics
        except Exception as exc:
            _log.error("[bench] %s/%s failed: %s", model_id, wl, exc, exc_info=True)
            all_metrics[wl] = {"model_id": model_id, "workload": wl, "error": str(exc)}

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return all_metrics


def _write_per_prompt_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PER_PROMPT_CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in PER_PROMPT_CSV_COLUMNS})


def _write_workload_metadata(
    wl_dir: Path, model_id: str, workload: str,
    dtype: str, device: str, seed: int, deterministic: bool,
    pack_hash: str,
) -> None:
    try:
        import transformers
        tv = transformers.__version__
    except Exception:
        tv = "unknown"
    meta = {
        "model_id": model_id,
        "workload": workload,
        "dtype": dtype,
        "device": device,
        "seed": seed,
        "deterministic": deterministic,
        "torch_version": torch.__version__,
        "transformers_version": tv,
        "gpu_name": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "promptpack_hash": pack_hash,
        "per_prompt_warmup": PER_PROMPT_WARMUP,
    }
    _write_json(wl_dir / "run_metadata.json", meta)


# ── aggregate CSV ─────────────────────────────────────────────────────────────

def write_benchmarks_csv(
    all_results: Dict[str, Dict[str, Dict[str, Any]]],
    outdir: Path,
    baseline_model: Optional[str] = None,
) -> Path:
    models = list(all_results)
    if baseline_model is None and models:
        baseline_model = models[0]

    baseline_ppl = None
    if baseline_model and baseline_model in all_results:
        baseline_ppl = all_results[baseline_model].get("wikitext", {}).get("ppl_wikitext")

    csv_path = outdir / "benchmarks.csv"
    outdir.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=BENCHMARKS_CSV_COLUMNS)
        writer.writeheader()
        for mid, workloads in all_results.items():
            model_ppl = workloads.get("wikitext", {}).get("ppl_wikitext")
            ppl_ratio = None
            if model_ppl is not None and baseline_ppl and baseline_ppl > 0:
                ppl_ratio = round(model_ppl / baseline_ppl, 4)
            for wl, m in workloads.items():
                if "error" in m and "workload" not in m:
                    continue
                row: Dict[str, Any] = {c: "" for c in BENCHMARKS_CSV_COLUMNS}
                row["model_id"] = mid
                row["workload"] = wl
                row["ppl_ratio"] = ppl_ratio if ppl_ratio is not None else ""
                for c in BENCHMARKS_CSV_COLUMNS:
                    if c in m and m[c] is not None:
                        row[c] = m[c]
                if wl != "wikitext":
                    row["ppl_wikitext"] = ""
                writer.writerow(row)
    _log.info("wrote %s", csv_path)
    return csv_path


# ── top-level entry ───────────────────────────────────────────────────────────

def run_bench(
    models: List[str],
    workloads: List[str],
    prompts_dir: Optional[Path],
    outdir: Path,
    device: str = "cuda",
    dtype_str: str = "bf16",
    seed: int = 0,
    deterministic: bool = False,
    baseline_model: Optional[str] = None,
) -> Path:
    results_dir = outdir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    resolved_baseline = baseline_model if baseline_model else (models[0] if models else None)
    emit_run_metadata(outdir, {
        "deterministic": deterministic, "seed": seed, "dtype": dtype_str,
        "baseline_model_id": resolved_baseline,
    })

    all_results: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for mid in models:
        all_results[mid] = bench_model(
            mid, workloads, prompts_dir, results_dir,
            device, dtype_str, seed, deterministic,
        )

    csv_path = write_benchmarks_csv(all_results, outdir, baseline_model)
    _log.info("[bench] complete — %d models × %d workloads", len(models), len(workloads))
    return csv_path
