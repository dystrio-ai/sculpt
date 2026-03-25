#!/usr/bin/env python3
"""Clean prefix caching benchmark with proper methodology.

Measures absolute throughput for original vs patched MoE models with and
without prefix caching enabled. Designed to produce defensible numbers:

  - Both models downloaded to local disk first (same I/O path)
  - 20-prompt warmup (discarded) before each measurement
  - 3 measurement rounds of 200 prompts each, report median ± range
  - Shared-prefix workload (the scenario where routing matters)
  - Clear side-by-side comparison table

Usage:
    python scripts/bench_prefix_cache.py \
        --original Qwen/Qwen3.5-122B-A10B-FP8 \
        --patched  dystrio/Qwen3.5-122B-A10B-FP8-CacheReady \
        --tp 2 \
        --output /ephemeral/clean_bench_results
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [bench] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("bench")

# ─── Workload ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a helpful AI assistant specialized in answering questions about "
    "science, technology, engineering, and mathematics. You provide clear, "
    "accurate, and concise answers. When appropriate, you include relevant "
    "examples and analogies to help explain complex concepts. You always "
    "cite your sources when possible and acknowledge uncertainty when you "
    "are not confident in your answer. Please respond thoughtfully."
)

QUERIES = [
    "What is the Heisenberg uncertainty principle?",
    "Explain how CRISPR gene editing works.",
    "What causes auroras (northern/southern lights)?",
    "How do quantum computers differ from classical computers?",
    "Explain the concept of entropy in thermodynamics.",
    "What is dark matter and why do scientists believe it exists?",
    "How does mRNA vaccine technology work?",
    "Explain the P vs NP problem in computer science.",
    "What is the standard model of particle physics?",
    "How do neural networks learn through backpropagation?",
    "What is the significance of the Higgs boson?",
    "Explain how blockchain consensus mechanisms work.",
    "What causes tectonic plates to move?",
    "How does LIGO detect gravitational waves?",
    "What is the holographic principle in theoretical physics?",
    "Explain the difference between supervised and unsupervised learning.",
    "What is quantum entanglement?",
    "How do black holes form and what happens at the event horizon?",
    "Explain the central dogma of molecular biology.",
    "What is the significance of Gödel's incompleteness theorems?",
    "How does photosynthesis convert light energy to chemical energy?",
    "What is the role of dopamine in the brain's reward system?",
    "Explain the concept of spacetime curvature in general relativity.",
    "How do antibiotics work and why is resistance a growing concern?",
    "What is the many-worlds interpretation of quantum mechanics?",
    "Explain how compilers optimize code during compilation.",
    "What are gravitational lensing effects and how are they observed?",
    "How does the immune system distinguish self from non-self?",
    "Explain the halting problem and its implications for computation.",
    "What is the cosmic microwave background radiation?",
    "How do enzymes catalyze biochemical reactions?",
    "What is topological quantum computing?",
    "Explain the role of telomeres in cellular aging.",
    "How does nuclear fusion work in stars?",
    "What are the key differences between TCP and UDP protocols?",
    "Explain the concept of herd immunity.",
    "What is the Riemann hypothesis and why does it matter?",
    "How do satellites maintain their orbits?",
    "Explain how transformers use attention mechanisms.",
    "What is the double-slit experiment and what does it demonstrate?",
]


def build_shared_prefix_prompts(n: int) -> List[str]:
    """Build n prompts that all share the same system prompt prefix."""
    prompts = []
    for i in range(n):
        q = QUERIES[i % len(QUERIES)]
        prompts.append(f"{SYSTEM_PROMPT}\n\nUser: {q}\nAssistant:")
    return prompts


# ─── Engine management ────────────────────────────────────────────────

def create_engine(
    model_path: str,
    tp: int,
    prefix_caching: bool,
    gpu_mem: float = 0.90,
    max_model_len: int = 2048,
):
    from vllm import LLM
    return LLM(
        model=model_path,
        trust_remote_code=True,
        tensor_parallel_size=tp,
        enable_prefix_caching=prefix_caching,
        gpu_memory_utilization=gpu_mem,
        max_model_len=max_model_len,
        enforce_eager=True,
    )


def destroy_engine(llm):
    try:
        if hasattr(llm, "llm_engine"):
            if hasattr(llm.llm_engine, "shutdown"):
                llm.llm_engine.shutdown()
            elif hasattr(llm.llm_engine, "model_executor"):
                ex = llm.llm_engine.model_executor
                if hasattr(ex, "shutdown"):
                    ex.shutdown()
    except Exception:
        pass
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    subprocess.run(
        "ps aux | grep '[m]ultiproc_executor' | awk '{print $2}' | xargs -r kill -9",
        shell=True, capture_output=True,
    )
    time.sleep(5)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def ensure_local(model_id: str) -> str:
    """Download model to local disk. Returns local path."""
    if Path(model_id).exists():
        return model_id
    from huggingface_hub import snapshot_download
    base = Path(os.environ.get("HF_HOME", Path.home() / "models"))
    local_dir = base / model_id.replace("/", "--")
    if local_dir.exists() and (local_dir / "config.json").exists():
        log.info("cached: %s", local_dir)
    else:
        log.info("downloading %s → %s", model_id, local_dir)
        snapshot_download(repo_id=model_id, local_dir=str(local_dir))
    return str(local_dir)


# ─── Measurement ──────────────────────────────────────────────────────

def measure_throughput(
    model_path: str,
    tp: int,
    prefix_caching: bool,
    n_prompts: int = 200,
    n_rounds: int = 3,
    n_warmup: int = 20,
    max_new_tokens: int = 64,
    mode: str = "batch",
) -> Dict[str, Any]:
    """Load model, warm up, measure throughput over multiple rounds.

    mode="batch":  submit all prompts at once via llm.generate()
    mode="online": submit prompts one-at-a-time to simulate sequential
                   serving where prefix caching actually matters.
    """
    from vllm import SamplingParams

    cache_label = "ON" if prefix_caching else "OFF"
    log.info("  loading engine (prefix_caching=%s, mode=%s)...", cache_label, mode)

    t_load = time.time()
    llm = create_engine(model_path, tp, prefix_caching)
    load_time = time.time() - t_load
    log.info("  engine loaded in %.1fs", load_time)

    params = SamplingParams(max_tokens=max_new_tokens, temperature=0)
    prompts = build_shared_prefix_prompts(n_prompts)
    warmup_prompts = build_shared_prefix_prompts(n_warmup)

    log.info("  warmup: %d prompts (discarded)...", n_warmup)
    if mode == "online":
        for wp in warmup_prompts:
            _ = llm.generate([wp], params)
    else:
        _ = llm.generate(warmup_prompts, params)

    round_throughputs = []
    round_times = []
    for r in range(n_rounds):
        t0 = time.time()
        if mode == "online":
            total_tokens = 0
            for prompt in prompts:
                outputs = llm.generate([prompt], params)
                total_tokens += len(outputs[0].outputs[0].token_ids)
        else:
            outputs = llm.generate(prompts, params)
            total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        elapsed = time.time() - t0
        tps = total_tokens / max(elapsed, 1e-9)
        round_throughputs.append(tps)
        round_times.append(elapsed)
        log.info(
            "  round %d/%d: %d tokens in %.1fs = %.0f tok/s",
            r + 1, n_rounds, total_tokens, elapsed, tps,
        )

    destroy_engine(llm)

    median_tps = statistics.median(round_throughputs)
    min_tps = min(round_throughputs)
    max_tps = max(round_throughputs)

    return {
        "prefix_caching": prefix_caching,
        "mode": mode,
        "n_prompts": n_prompts,
        "n_rounds": n_rounds,
        "n_warmup": n_warmup,
        "max_new_tokens": max_new_tokens,
        "load_time_s": round(load_time, 1),
        "round_throughputs_tok_s": [round(t, 1) for t in round_throughputs],
        "round_times_s": [round(t, 2) for t in round_times],
        "median_tok_s": round(median_tps, 1),
        "min_tok_s": round(min_tps, 1),
        "max_tok_s": round(max_tps, 1),
    }


# ─── Report ───────────────────────────────────────────────────────────

def generate_report(
    results: Dict[str, Any],
    output_dir: Path,
) -> str:
    lines = [
        "# Prefix Caching Benchmark Report",
        "",
        f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        f"**Hardware**: {results.get('gpu_info', 'N/A')}",
        f"**TP**: {results.get('tp', 'N/A')}",
        f"**Prompts per round (batch)**: {results.get('n_prompts', 200)}",
        f"**Rounds**: {results.get('n_rounds', 3)}",
        f"**Warmup prompts**: {results.get('n_warmup', 20)}",
        f"**Modes**: {', '.join(results.get('modes', ['batch']))}",
        "",
    ]

    orig = results.get("original", {})
    patch = results.get("patched", {})

    def _fmt(d):
        if not d or "error" in d:
            return "—", "—"
        med = d.get("median_tok_s", 0)
        lo = d.get("min_tok_s", 0)
        hi = d.get("max_tok_s", 0)
        spread = max(hi - med, med - lo)
        return f"{med:.0f}", f"±{spread:.0f}"

    def _effect(d_off, d_on):
        if not d_off or not d_on or "error" in d_off or "error" in d_on:
            return "—"
        off = d_off.get("median_tok_s", 0)
        on = d_on.get("median_tok_s", 0)
        if off <= 0:
            return "—"
        return f"{on / off:.2f}x"

    for bench_mode in results.get("modes", ["batch"]):
        off_key = f"{bench_mode}_cache_off"
        on_key = f"{bench_mode}_cache_on"

        # Fall back to old-style keys for single-mode runs
        orig_off = orig.get(off_key, orig.get("cache_off", {}))
        orig_on = orig.get(on_key, orig.get("cache_on", {}))
        patch_off = patch.get(off_key, patch.get("cache_off", {}))
        patch_on = patch.get(on_key, patch.get("cache_on", {}))

        label = "Batch" if bench_mode == "batch" else "Online (sequential)"

        lines.extend([
            f"## {label} — Shared-Prefix Throughput (tok/s)",
            "",
            "| Model | Cache OFF | Cache ON | Cache Effect |",
            "|-------|-----------|----------|--------------|",
        ])

        orig_off_med, orig_off_spread = _fmt(orig_off)
        orig_on_med, orig_on_spread = _fmt(orig_on)
        patch_off_med, patch_off_spread = _fmt(patch_off)
        patch_on_med, patch_on_spread = _fmt(patch_on)

        lines.append(
            f"| Original | {orig_off_med} {orig_off_spread} | "
            f"{orig_on_med} {orig_on_spread} | {_effect(orig_off, orig_on)} |"
        )
        lines.append(
            f"| **CacheReady** | **{patch_off_med} {patch_off_spread}** | "
            f"**{patch_on_med} {patch_on_spread}** | **{_effect(patch_off, patch_on)}** |"
        )
        lines.append("")

        orig_off_v = orig_off.get("median_tok_s", 0) if orig_off and "error" not in orig_off else 0
        orig_on_v = orig_on.get("median_tok_s", 0) if orig_on and "error" not in orig_on else 0
        patch_off_v = patch_off.get("median_tok_s", 0) if patch_off and "error" not in patch_off else 0
        patch_on_v = patch_on.get("median_tok_s", 0) if patch_on and "error" not in patch_on else 0

        lines.append(f"### {label} — Key Comparisons")
        lines.append("")

        if orig_off_v > 0 and patch_off_v > 0:
            overhead = (patch_off_v - orig_off_v) / orig_off_v * 100
            lines.append(f"- **Patch overhead** (cache OFF): {overhead:+.1f}% "
                         f"({orig_off_v:.0f} → {patch_off_v:.0f} tok/s)")

        if orig_off_v > 0 and orig_on_v > 0:
            pct = (orig_on_v - orig_off_v) / orig_off_v * 100
            lines.append(f"- **Original + cache**: {pct:+.1f}% "
                         f"({orig_off_v:.0f} → {orig_on_v:.0f} tok/s)")

        if patch_off_v > 0 and patch_on_v > 0:
            pct = (patch_on_v - patch_off_v) / patch_off_v * 100
            lines.append(f"- **CacheReady + cache**: {pct:+.1f}% "
                         f"({patch_off_v:.0f} → {patch_on_v:.0f} tok/s)")

        if orig_off_v > 0 and patch_on_v > 0:
            absolute = patch_on_v / orig_off_v
            lines.append(f"- **Absolute** (original no-cache vs CacheReady+cache): "
                         f"**{absolute:.2f}x** ({orig_off_v:.0f} → {patch_on_v:.0f} tok/s)")

        lines.append("")

        # Per-round details
        lines.append(f"### {label} — Per-Round Details")
        lines.append("")
        for model_label, model_data in [("Original", orig), ("CacheReady", patch)]:
            for cache_label_key, nice_label in [(off_key, "cache_off"), (on_key, "cache_on")]:
                cache_data = model_data.get(cache_label_key, {})
                if not cache_data or "error" in cache_data:
                    continue
                rounds = cache_data.get("round_throughputs_tok_s", [])
                if rounds:
                    round_str = ", ".join(f"{r:.0f}" for r in rounds)
                    lines.append(f"- **{model_label}** ({nice_label}): [{round_str}] tok/s")
        lines.append("")

    report = "\n".join(lines)
    report_path = output_dir / "prefix_cache_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    log.info("report: %s", report_path)
    return report


# ─── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Clean prefix caching benchmark")
    parser.add_argument("--original", required=True, help="Original model HF ID or local path")
    parser.add_argument("--patched", required=True, help="Patched model HF ID or local path")
    parser.add_argument("--tp", type=int, default=2, help="Tensor parallel size")
    parser.add_argument("--output", default="~/clean_bench_results", help="Output directory")
    parser.add_argument("--n-prompts", type=int, default=200, help="Prompts per round")
    parser.add_argument("--n-rounds", type=int, default=3, help="Measurement rounds")
    parser.add_argument("--n-warmup", type=int, default=20, help="Warmup prompts (discarded)")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Max tokens per prompt")
    parser.add_argument(
        "--mode", choices=["batch", "online", "both"], default="both",
        help="batch = submit all prompts at once (throughput test), "
             "online = submit one-at-a-time (latency/cache-hit test), "
             "both = run both modes sequentially (default)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    gpu_info = "unknown"
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True,
        )
        gpu_lines = r.stdout.strip().split("\n")
        gpu_info = f"{len(gpu_lines)}x {gpu_lines[0].strip()}" if gpu_lines else "unknown"
    except Exception:
        pass

    modes = ["batch", "online"] if args.mode == "both" else [args.mode]

    log.info("=" * 60)
    log.info("Prefix Caching Benchmark (clean methodology)")
    log.info("Original: %s", args.original)
    log.info("Patched:  %s", args.patched)
    log.info("TP: %d | Prompts: %d | Rounds: %d | Warmup: %d",
             args.tp, args.n_prompts, args.n_rounds, args.n_warmup)
    log.info("Modes: %s", ", ".join(modes))
    log.info("GPU: %s", gpu_info)
    log.info("=" * 60)

    log.info("--- Step 1: Download models to local disk ---")
    original_local = ensure_local(args.original)
    patched_local = ensure_local(args.patched)
    log.info("original: %s", original_local)
    log.info("patched:  %s", patched_local)

    all_results: Dict[str, Any] = {
        "original_model": args.original,
        "patched_model": args.patched,
        "original_local": original_local,
        "patched_local": patched_local,
        "tp": args.tp,
        "n_prompts": args.n_prompts,
        "n_rounds": args.n_rounds,
        "n_warmup": args.n_warmup,
        "max_new_tokens": args.max_new_tokens,
        "gpu_info": gpu_info,
        "modes": modes,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }

    t_total = time.time()

    for bench_mode in modes:
        log.info("=" * 40)
        log.info("MODE: %s", bench_mode.upper())
        log.info("=" * 40)

        n_prompts = args.n_prompts if bench_mode == "batch" else min(args.n_prompts, 60)
        n_warmup = args.n_warmup if bench_mode == "batch" else min(args.n_warmup, 10)

        for model_label, model_path in [("original", original_local), ("patched", patched_local)]:
            log.info("--- %s [%s]: %s ---", model_label.upper(), bench_mode, model_path)
            model_results = all_results.setdefault(model_label, {})

            for cache_enabled in [False, True]:
                cache_label = f"{bench_mode}_cache_{'on' if cache_enabled else 'off'}"
                log.info("[%s] prefix_caching=%s mode=%s", model_label, cache_enabled, bench_mode)
                try:
                    r = measure_throughput(
                        model_path=model_path,
                        tp=args.tp,
                        prefix_caching=cache_enabled,
                        n_prompts=n_prompts,
                        n_rounds=args.n_rounds,
                        n_warmup=n_warmup,
                        max_new_tokens=args.max_new_tokens,
                        mode=bench_mode,
                    )
                    model_results[cache_label] = r
                    log.info("[%s] %s: median %.0f tok/s (range: %.0f–%.0f)",
                             model_label, cache_label,
                             r["median_tok_s"], r["min_tok_s"], r["max_tok_s"])
                except Exception as e:
                    log.error("[%s] %s FAILED: %s", model_label, cache_label, e, exc_info=True)
                    model_results[cache_label] = {"error": str(e)}

    total_time = time.time() - t_total
    all_results["total_time_s"] = round(total_time, 1)
    all_results["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    raw_path = output_dir / "prefix_cache_results.json"
    with open(raw_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info("raw results: %s", raw_path)

    report = generate_report(all_results, output_dir)
    print("\n" + report)

    log.info("=" * 60)
    log.info("BENCHMARK COMPLETE in %.0f seconds (%.1f minutes)", total_time, total_time / 60)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
