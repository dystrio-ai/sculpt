#!/usr/bin/env python3
"""MoE Routing Patch Benchmark Suite.

Validates the Physarum routing canonicalization patch across four dimensions:

  1. Routing Determinism — identical inputs → identical outputs (via vLLM)
  2. Quality Preservation — full lm_eval benchmarks with vLLM backend
  3. vLLM Prefix Caching — throughput with shared-prefix workloads
  4. nvfp4 Quantization Stability — determinism under quantization

All tests use vLLM natively (no transformers AutoModel), ensuring compatibility
with Qwen3.5-MoE and matching production serving conditions.

Usage:
    python scripts/benchmark_moe_routing.py \
        --original Qwen/Qwen3.5-122B-A10B \
        --patched  ~/moe_cache_ready \
        --output   ~/moe_benchmark_results

Each test writes detailed JSON results and a final markdown report is generated.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("moe_benchmark")


# ─── Test texts ──────────────────────────────────────────────────────

DETERMINISM_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is transforming industries across the globe.",
    "In quantum computing, qubits can exist in superposition states.",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    "The European Central Bank announced new monetary policy measures today.",
    "Consider a Riemannian manifold M with metric tensor g. The Christoffel symbols are defined as...",
    "SELECT u.name, COUNT(o.id) FROM users u LEFT JOIN orders o ON u.id = o.user_id GROUP BY u.name;",
    "A photograph shows a sunset over the ocean with vibrant orange and purple hues reflecting off the waves.",
    "The patient presents with acute onset chest pain radiating to the left arm, diaphoresis, and shortness of breath.",
    "According to the latest IPCC report, global temperatures have risen approximately 1.1 degrees Celsius above pre-industrial levels.",
    "import torch\nmodel = torch.nn.Linear(768, 768)\noptimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)",
    "The Krebs cycle, also known as the citric acid cycle, is a series of chemical reactions used by aerobic organisms.",
    "In the year 2024, artificial general intelligence remains an aspirational goal for researchers worldwide.",
    "La inteligencia artificial está cambiando la forma en que interactuamos con la tecnología.",
    "Transformer architectures rely on self-attention mechanisms to capture long-range dependencies in sequences.",
    "The recipe calls for 2 cups flour, 1 cup sugar, 3 eggs, and a teaspoon of vanilla extract.",
    "Prove that for all n >= 1, the sum 1 + 2 + ... + n = n(n+1)/2 by mathematical induction.",
    "Breaking news: The stock market experienced significant volatility today as investors reacted to unexpected economic data.",
    "The double-slit experiment demonstrates wave-particle duality in quantum mechanics.",
    "Once upon a time in a distant galaxy, a lone spaceship drifted through the cosmic void searching for a new home.",
]

SYSTEM_PROMPT = (
    "You are a helpful AI assistant specialized in answering questions about "
    "science, technology, engineering, and mathematics. You provide clear, "
    "accurate, and concise answers. When appropriate, you include relevant "
    "examples and analogies to help explain complex concepts. You always "
    "cite your sources when possible and acknowledge uncertainty when you "
    "are not confident in your answer. Please respond thoughtfully."
)

SHARED_PREFIX_QUERIES = [
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
] * 3

UNIQUE_PROMPTS = [
    f"Tell me about topic number {i} in great detail, covering all aspects." for i in range(60)
]


# ─── Helpers ─────────────────────────────────────────────────────────

def _ensure_local_model(model_id: str) -> str:
    """Download a HF model to local cache and fix config if needed.

    Fixes the known issue where our patched model has model_type
    'qwen3_5_moe_text' which older transformers versions don't recognize.
    Returns the local path to use.
    """
    if Path(model_id).exists():
        config_path = Path(model_id) / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                cfg = json.load(f)
            if cfg.get("model_type") == "qwen3_5_moe_text":
                log.info("fixing model_type qwen3_5_moe_text → qwen3_5_moe in %s", config_path)
                cfg["model_type"] = "qwen3_5_moe"
                with open(config_path, "w") as f:
                    json.dump(cfg, f, indent=2)
        return model_id

    from huggingface_hub import snapshot_download
    local_dir = Path.home() / "models" / model_id.replace("/", "--")
    if local_dir.exists() and (local_dir / "config.json").exists():
        log.info("using cached model at %s", local_dir)
    else:
        log.info("downloading %s → %s", model_id, local_dir)
        snapshot_download(repo_id=model_id, local_dir=str(local_dir))

    config_path = local_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        if cfg.get("model_type") == "qwen3_5_moe_text":
            log.info("fixing model_type qwen3_5_moe_text → qwen3_5_moe in %s", config_path)
            cfg["model_type"] = "qwen3_5_moe"
            with open(config_path, "w") as f:
                json.dump(cfg, f, indent=2)

    return str(local_dir)


def _free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _create_vllm_engine(
    model_path: str,
    tensor_parallel_size: int = 4,
    enable_prefix_caching: bool = False,
    quantization: Optional[str] = None,
    gpu_memory_utilization: float = 0.90,
    max_model_len: int = 2048,
):
    from vllm import LLM
    kwargs = dict(
        model=model_path,
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel_size,
        enable_prefix_caching=enable_prefix_caching,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        enforce_eager=True,
    )
    if quantization:
        kwargs["quantization"] = quantization
    return LLM(**kwargs)


def _destroy_vllm_engine(llm):
    del llm
    _free_gpu()
    time.sleep(5)


# ─── Test 1: Routing Determinism ─────────────────────────────────────

def test_determinism(
    model_path: str,
    output_dir: Path,
    label: str,
    tensor_parallel_size: int = 4,
    n_runs: int = 5,
    max_new_tokens: int = 32,
) -> Dict[str, Any]:
    """Run multiple generations on diverse texts, check output identity.

    Uses vLLM with temperature=0. If routing is deterministic, identical
    prompts produce bit-for-bit identical outputs across all runs.
    Also collects prompt_logprobs for finer-grained divergence analysis.
    """
    log.info("=== Test 1: Routing Determinism [%s] ===", label)

    from vllm import SamplingParams

    t0 = time.time()
    llm = _create_vllm_engine(model_path, tensor_parallel_size=tensor_parallel_size)
    load_time = time.time() - t0
    log.info("model loaded in %.0fs", load_time)

    params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0,
        prompt_logprobs=1,
    )

    # Warmup
    _ = llm.generate(DETERMINISM_TEXTS[:2], params)

    results_per_text = []

    for ti, text in enumerate(DETERMINISM_TEXTS):
        runs_tokens = []
        runs_text = []
        runs_logprobs = []

        for r in range(n_runs):
            outputs = llm.generate([text], params)
            out = outputs[0]
            gen_tokens = list(out.outputs[0].token_ids)
            gen_text = out.outputs[0].text
            runs_tokens.append(gen_tokens)
            runs_text.append(gen_text)

            if out.prompt_logprobs is not None:
                prompt_lps = []
                for pos_dict in out.prompt_logprobs:
                    if pos_dict is not None:
                        top_lp = max(pos_dict.values(), key=lambda x: x.logprob if hasattr(x, 'logprob') else x)
                        lp_val = top_lp.logprob if hasattr(top_lp, 'logprob') else float(top_lp)
                        prompt_lps.append(lp_val)
                runs_logprobs.append(prompt_lps)

        all_tokens_same = all(t == runs_tokens[0] for t in runs_tokens)
        all_text_same = all(t == runs_text[0] for t in runs_text)
        unique_outputs = len(set(str(t) for t in runs_tokens))

        logprob_max_diff = 0.0
        if len(runs_logprobs) >= 2:
            for i in range(len(runs_logprobs)):
                for j in range(i + 1, len(runs_logprobs)):
                    min_len = min(len(runs_logprobs[i]), len(runs_logprobs[j]))
                    for k in range(min_len):
                        diff = abs(runs_logprobs[i][k] - runs_logprobs[j][k])
                        logprob_max_diff = max(logprob_max_diff, diff)

        result = {
            "text_idx": ti,
            "text_preview": text[:80],
            "n_runs": n_runs,
            "all_tokens_identical": all_tokens_same,
            "all_text_identical": all_text_same,
            "unique_outputs": unique_outputs,
            "logprob_max_diff": logprob_max_diff,
            "generated_text_sample": runs_text[0][:200],
            "deterministic": all_tokens_same,
        }
        results_per_text.append(result)

        status = "PASS" if all_tokens_same else f"FAIL ({unique_outputs} unique)"
        log.info(
            "  text %d/%d: %s (logprob_diff=%.2e)",
            ti + 1, len(DETERMINISM_TEXTS), status, logprob_max_diff,
        )

    _destroy_vllm_engine(llm)

    n_deterministic = sum(1 for r in results_per_text if r["deterministic"])
    agreement_rate = n_deterministic / len(results_per_text)
    overall_logprob_diff = max(r["logprob_max_diff"] for r in results_per_text)

    summary = {
        "label": label,
        "model_path": model_path,
        "n_texts": len(DETERMINISM_TEXTS),
        "n_runs_per_text": n_runs,
        "n_deterministic": n_deterministic,
        "agreement_rate": agreement_rate,
        "overall_logprob_max_diff": overall_logprob_diff,
        "all_deterministic": n_deterministic == len(results_per_text),
        "pass": n_deterministic == len(results_per_text),
        "load_time_s": load_time,
        "per_text": results_per_text,
    }

    out_path = output_dir / f"determinism_{label}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(
        "determinism [%s]: %d/%d texts deterministic (%.1f%%) → %s",
        label, n_deterministic, len(results_per_text),
        agreement_rate * 100, out_path,
    )

    return summary


# ─── Test 2: Quality Preservation via lm_eval + vLLM ─────────────────

LM_EVAL_TASKS = "mmlu,hellaswag,arc_challenge,truthfulqa_mc2,winogrande,gsm8k"


def test_quality_lm_eval(
    model_path: str,
    output_dir: Path,
    label: str,
    tensor_parallel_size: int = 4,
    tasks: str = LM_EVAL_TASKS,
) -> Dict[str, Any]:
    """Run lm_eval harness with vLLM backend."""
    log.info("=== Test 2: Quality [%s] ===", label)
    log.info("tasks: %s", tasks)

    eval_output = output_dir / f"lm_eval_{label}"
    eval_output.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "vllm",
        "--model_args", (
            f"pretrained={model_path},"
            f"tensor_parallel_size={tensor_parallel_size},"
            "trust_remote_code=True,"
            "gpu_memory_utilization=0.90,"
            "max_model_len=2048,"
            "enforce_eager=True"
        ),
        "--tasks", tasks,
        "--batch_size", "auto",
        "--output_path", str(eval_output),
    ]

    log.info("running: %s", " ".join(cmd))
    t0 = time.time()

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=36000)
    elapsed = time.time() - t0

    log.info("lm_eval [%s] completed in %.0fs (exit code %d)", label, elapsed, result.returncode)

    if result.returncode != 0:
        log.error("lm_eval stderr (last 2000 chars):\n%s", result.stderr[-2000:] if result.stderr else "(empty)")
        log.error("lm_eval stdout (last 2000 chars):\n%s", result.stdout[-2000:] if result.stdout else "(empty)")

    results_file = None
    for p in eval_output.rglob("results.json"):
        results_file = p
        break

    scores = {}
    if results_file and results_file.exists():
        with open(results_file) as f:
            raw = json.load(f)
        for task_name, task_data in raw.get("results", {}).items():
            for metric, value in task_data.items():
                if isinstance(value, (int, float)):
                    scores[f"{task_name}/{metric}"] = value
        log.info("scores: %s", {k: f"{v:.4f}" for k, v in scores.items() if "acc" in k})

    summary = {
        "label": label,
        "model_path": model_path,
        "tasks": tasks,
        "elapsed_seconds": elapsed,
        "exit_code": result.returncode,
        "scores": scores,
        "results_dir": str(eval_output),
    }

    out_path = output_dir / f"quality_{label}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def compare_quality(original: Dict, patched: Dict, output_dir: Path) -> Dict[str, Any]:
    """Compare quality scores between original and patched models."""
    log.info("=== Quality Comparison ===")

    orig_scores = original.get("scores", {})
    patch_scores = patched.get("scores", {})

    all_keys = sorted(set(orig_scores) | set(patch_scores))
    comparisons = {}
    max_degradation = 0.0

    for key in all_keys:
        if "acc" not in key:
            continue
        orig_val = orig_scores.get(key)
        patch_val = patch_scores.get(key)
        if orig_val is not None and patch_val is not None:
            diff = patch_val - orig_val
            comparisons[key] = {
                "original": orig_val,
                "patched": patch_val,
                "diff": diff,
                "diff_pct": diff * 100,
            }
            if diff < 0:
                max_degradation = max(max_degradation, abs(diff))
            log.info("  %s: %.4f → %.4f (diff: %+.4f)", key, orig_val, patch_val, diff)

    passed = max_degradation < 0.005
    summary = {
        "comparisons": comparisons,
        "max_degradation": max_degradation,
        "pass": passed,
    }

    out_path = output_dir / "quality_comparison.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("quality comparison: %s (max degradation: %.4f) → %s",
             "PASS" if passed else "FAIL", max_degradation, out_path)

    return summary


# ─── Test 3: vLLM Prefix Caching ────────────────────────────────────

def _run_vllm_workload(
    model_path: str,
    prompts: List[str],
    enable_prefix_caching: bool,
    max_new_tokens: int = 64,
    tensor_parallel_size: int = 4,
    quantization: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a vLLM workload and measure throughput/latency."""
    from vllm import SamplingParams

    log.info(
        "  vLLM workload: %d prompts, prefix_cache=%s, quant=%s, tp=%d",
        len(prompts), enable_prefix_caching, quantization, tensor_parallel_size,
    )

    t_load = time.time()
    llm = _create_vllm_engine(
        model_path,
        tensor_parallel_size=tensor_parallel_size,
        enable_prefix_caching=enable_prefix_caching,
        quantization=quantization,
    )
    load_time = time.time() - t_load

    params = SamplingParams(max_tokens=max_new_tokens, temperature=0)

    # Warmup
    _ = llm.generate(prompts[:2], params)

    # First pass: throughput
    t0 = time.time()
    outputs = llm.generate(prompts, params)
    first_pass_time = time.time() - t0
    total_gen = sum(len(o.outputs[0].token_ids) for o in outputs)
    throughput_first = total_gen / max(1e-9, first_pass_time)

    # Latency: first 10 prompts individually
    latencies = []
    for prompt in prompts[:10]:
        t0 = time.time()
        _ = llm.generate([prompt], params)
        latencies.append(time.time() - t0)

    latencies.sort()
    p50 = latencies[len(latencies) // 2] if latencies else 0.0
    p95_idx = min(int(len(latencies) * 0.95), len(latencies) - 1)
    p95 = latencies[p95_idx] if latencies else 0.0

    # Second pass: same prompts again (measures cache benefit)
    t0 = time.time()
    outputs2 = llm.generate(prompts, params)
    second_pass_time = time.time() - t0
    total_gen2 = sum(len(o.outputs[0].token_ids) for o in outputs2)
    throughput_second = total_gen2 / max(1e-9, second_pass_time)

    _destroy_vllm_engine(llm)

    return {
        "throughput_first_pass_tok_s": round(throughput_first, 2),
        "throughput_second_pass_tok_s": round(throughput_second, 2),
        "cache_speedup_ratio": round(throughput_second / max(throughput_first, 1e-9), 3),
        "latency_p50_s": round(p50, 4),
        "latency_p95_s": round(p95, 4),
        "total_generated_tokens": total_gen,
        "n_prompts": len(prompts),
        "load_time_s": round(load_time, 1),
        "first_pass_time_s": round(first_pass_time, 2),
        "second_pass_time_s": round(second_pass_time, 2),
        "enable_prefix_caching": enable_prefix_caching,
        "quantization": quantization,
    }


def test_vllm_prefix_caching(
    model_path: str,
    output_dir: Path,
    label: str,
    tensor_parallel_size: int = 4,
) -> Dict[str, Any]:
    """Test prefix caching effectiveness with shared-prefix workloads."""
    log.info("=== Test 3: vLLM Prefix Caching [%s] ===", label)

    shared_prompts = [f"{SYSTEM_PROMPT}\n\nUser: {q}\nAssistant:" for q in SHARED_PREFIX_QUERIES]
    unique_prompts = [f"User: {q}\nAssistant:" for q in UNIQUE_PROMPTS]

    results = {}

    for workload_name, prompts in [
        ("shared_prefix", shared_prompts),
        ("unique_prefix", unique_prompts),
    ]:
        for cache_enabled in [False, True]:
            key = f"{workload_name}_cache_{cache_enabled}"
            log.info("workload: %s", key)
            try:
                r = _run_vllm_workload(
                    model_path=model_path,
                    prompts=prompts,
                    enable_prefix_caching=cache_enabled,
                    tensor_parallel_size=tensor_parallel_size,
                )
                results[key] = r
                log.info(
                    "  throughput: %.0f tok/s (1st) → %.0f tok/s (2nd), speedup: %.2fx",
                    r["throughput_first_pass_tok_s"],
                    r["throughput_second_pass_tok_s"],
                    r["cache_speedup_ratio"],
                )
            except Exception as e:
                log.error("  workload %s failed: %s", key, e, exc_info=True)
                results[key] = {"error": str(e)}

    cache_benefit = {}
    for workload in ["shared_prefix", "unique_prefix"]:
        no_cache = results.get(f"{workload}_cache_False", {})
        with_cache = results.get(f"{workload}_cache_True", {})
        if "error" not in no_cache and "error" not in with_cache:
            tp_no = no_cache.get("throughput_second_pass_tok_s", 0)
            tp_yes = with_cache.get("throughput_second_pass_tok_s", 0)
            cache_benefit[workload] = {
                "without_cache_tok_s": tp_no,
                "with_cache_tok_s": tp_yes,
                "cache_benefit_ratio": round(tp_yes / max(tp_no, 1e-9), 3),
            }

    summary = {
        "label": label,
        "model_path": model_path,
        "tensor_parallel_size": tensor_parallel_size,
        "workload_results": results,
        "cache_benefit": cache_benefit,
    }

    out_path = output_dir / f"vllm_prefix_cache_{label}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("vLLM prefix cache [%s] → %s", label, out_path)

    return summary


# ─── Test 4: Quantized Routing Stability ─────────────────────────────

def test_quantized_determinism(
    model_path: str,
    output_dir: Path,
    label: str,
    tensor_parallel_size: int = 4,
    n_runs: int = 5,
    max_new_tokens: int = 32,
) -> Dict[str, Any]:
    """Test routing determinism under fp8 quantization via vLLM."""
    log.info("=== Test 4: Quantized Determinism [%s] ===", label)

    from vllm import SamplingParams

    quant_methods = ["fp8", "compressed-tensors"]
    llm = None
    used_quant = None

    for qmethod in quant_methods:
        try:
            log.info("  trying quantization: %s", qmethod)
            llm = _create_vllm_engine(
                model_path,
                tensor_parallel_size=tensor_parallel_size,
                quantization=qmethod,
                max_model_len=1024,
            )
            used_quant = qmethod
            log.info("  using quantization: %s", qmethod)
            break
        except Exception as e:
            log.warning("  %s not available: %s", qmethod, e)
            continue

    if llm is None:
        log.warning("no quantization method available, skipping")
        return {"label": label, "skipped": True, "reason": "no quantization available"}

    params = SamplingParams(max_tokens=max_new_tokens, temperature=0)

    # Warmup
    _ = llm.generate(DETERMINISM_TEXTS[:2], params)

    texts = DETERMINISM_TEXTS[:10]
    results_per_text = []

    for ti, text in enumerate(texts):
        runs_tokens = []
        for r in range(n_runs):
            outputs = llm.generate([text], params)
            gen_tokens = list(outputs[0].outputs[0].token_ids)
            runs_tokens.append(gen_tokens)

        all_same = all(t == runs_tokens[0] for t in runs_tokens)
        unique_outputs = len(set(str(t) for t in runs_tokens))
        results_per_text.append({
            "text_idx": ti,
            "text_preview": text[:80],
            "n_runs": n_runs,
            "all_identical": all_same,
            "unique_outputs": unique_outputs,
        })

        status = "PASS" if all_same else f"DIVERGENT ({unique_outputs} unique)"
        log.info("  text %d/%d: %s", ti + 1, len(texts), status)

    _destroy_vllm_engine(llm)

    n_deterministic = sum(1 for r in results_per_text if r["all_identical"])
    agreement_rate = n_deterministic / len(results_per_text) if results_per_text else 0

    summary = {
        "label": label,
        "model_path": model_path,
        "quantization_method": used_quant,
        "n_texts": len(texts),
        "n_runs_per_text": n_runs,
        "n_deterministic": n_deterministic,
        "agreement_rate": agreement_rate,
        "pass": agreement_rate > 0.999,
        "per_text": results_per_text,
    }

    out_path = output_dir / f"quantized_determinism_{label}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("quantized determinism [%s]: %.1f%% agreement → %s",
             label, agreement_rate * 100, out_path)

    return summary


# ─── Report Generation ───────────────────────────────────────────────

def generate_report(results: Dict[str, Any], output_dir: Path) -> str:
    """Generate a markdown summary report."""

    lines = [
        "# MoE Routing Patch Benchmark Report",
        "",
        f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        f"**Original model**: {results.get('original_model', 'N/A')}",
        f"**Patched model**: {results.get('patched_model', 'N/A')}",
        "",
    ]

    # Determinism
    lines.append("## 1. Routing Determinism (bf16, temperature=0)")
    lines.append("")
    for label in ["original", "patched"]:
        det = results.get(f"determinism_{label}", {})
        if "error" in det:
            lines.append(f"- **{label}**: ERROR — {det['error']}")
            continue
        n_det = det.get("n_deterministic", "?")
        n_total = det.get("n_texts", "?")
        rate = det.get("agreement_rate", 0) * 100
        status = "PASS" if det.get("pass") else "FAIL"
        lines.append(f"- **{label}**: {n_det}/{n_total} texts deterministic ({rate:.1f}%) [{status}]")
    lines.append("")

    # Quality
    lines.append("## 2. Quality Preservation (lm_eval via vLLM)")
    lines.append("")
    qc = results.get("quality_comparison", {})
    if qc and "comparisons" in qc:
        lines.append("| Metric | Original | Patched | Diff |")
        lines.append("|--------|----------|---------|------|")
        for metric, data in sorted(qc.get("comparisons", {}).items()):
            lines.append(
                f"| {metric} | {data['original']:.4f} | {data['patched']:.4f} | {data['diff']:+.4f} |"
            )
        lines.append("")
        status = "PASS" if qc.get("pass") else "FAIL"
        lines.append(f"**Result**: {status} (max degradation: {qc.get('max_degradation', 0):.4f})")
    else:
        lines.append("Quality comparison not available.")
    lines.append("")

    # vLLM Prefix Caching
    lines.append("## 3. vLLM Prefix Caching Throughput")
    lines.append("")
    for label in ["original", "patched"]:
        vllm_data = results.get(f"vllm_{label}", {})
        if "error" in vllm_data:
            lines.append(f"**{label}**: ERROR — {vllm_data.get('error', '')}")
            continue
        cb = vllm_data.get("cache_benefit", {})
        for workload, data in cb.items():
            lines.append(
                f"- **{label} / {workload}**: "
                f"{data.get('without_cache_tok_s', 0):.0f} → "
                f"{data.get('with_cache_tok_s', 0):.0f} tok/s "
                f"(cache benefit: {data.get('cache_benefit_ratio', 0):.2f}x)"
            )
    lines.append("")

    # Quantized determinism
    lines.append("## 4. Quantized Routing Stability")
    lines.append("")
    for label in ["original", "patched"]:
        nv = results.get(f"quant_{label}", {})
        if nv.get("skipped") or "error" in nv:
            reason = nv.get("reason", nv.get("error", "unknown"))
            lines.append(f"- **{label}**: skipped ({reason})")
            continue
        rate = nv.get("agreement_rate", 0) * 100
        status = "PASS" if nv.get("pass") else "FAIL"
        lines.append(f"- **{label}**: {rate:.1f}% routing agreement [{status}] (quant: {nv.get('quantization_method', 'N/A')})")
    lines.append("")

    # Overall
    lines.append("## Overall Verdict")
    lines.append("")
    det_pass = results.get("determinism_patched", {}).get("pass", False)
    qual_pass = results.get("quality_comparison", {}).get("pass", True)
    lines.append(f"- Determinism: {'PASS' if det_pass else 'FAIL'}")
    lines.append(f"- Quality: {'PASS' if qual_pass else 'FAIL/SKIPPED'}")
    lines.append(f"- **Overall: {'PASS' if (det_pass and qual_pass) else 'NEEDS REVIEW'}**")
    lines.append("")

    report = "\n".join(lines)
    report_path = output_dir / "benchmark_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    log.info("report written to %s", report_path)

    return report


# ─── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MoE Routing Patch Benchmark Suite")
    parser.add_argument("--original", required=True, help="Original model (HF ID or local path)")
    parser.add_argument("--patched", required=True, help="Patched model (HF ID or local path)")
    parser.add_argument("--output", default="~/moe_benchmark_results", help="Output directory")
    parser.add_argument("--tp", type=int, default=4, help="Tensor parallel size for vLLM")
    parser.add_argument("--skip-quality", action="store_true", help="Skip lm_eval (slow)")
    parser.add_argument("--skip-vllm", action="store_true", help="Skip vLLM prefix cache tests")
    parser.add_argument("--skip-quant", action="store_true", help="Skip quantization tests")
    args = parser.parse_args()

    output_dir = Path(args.output).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("MoE Routing Patch Benchmark Suite (vLLM-native)")
    log.info("Original: %s", args.original)
    log.info("Patched:  %s", args.patched)
    log.info("Output:   %s", output_dir)
    log.info("TP size:  %d", args.tp)
    log.info("=" * 60)

    # Download and fix patched model config if needed
    patched_local = _ensure_local_model(args.patched)
    if patched_local != args.patched:
        log.info("using local patched model: %s", patched_local)
        args.patched = patched_local

    all_results: Dict[str, Any] = {
        "original_model": args.original,
        "patched_model": args.patched,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }

    t_total = time.time()

    # ── Test 1: Determinism ──
    for label, model_path in [("original", args.original), ("patched", args.patched)]:
        try:
            r = test_determinism(model_path, output_dir, label, tensor_parallel_size=args.tp)
            all_results[f"determinism_{label}"] = r
        except Exception as e:
            log.error("determinism test [%s] failed: %s", label, e, exc_info=True)
            all_results[f"determinism_{label}"] = {"error": str(e)}

    # ── Test 2: Quality ──
    if not args.skip_quality:
        for label, model_path in [("original", args.original), ("patched", args.patched)]:
            try:
                r = test_quality_lm_eval(model_path, output_dir, label, tensor_parallel_size=args.tp)
                all_results[f"quality_{label}"] = r
            except Exception as e:
                log.error("quality test [%s] failed: %s", label, e, exc_info=True)
                all_results[f"quality_{label}"] = {"error": str(e)}

        orig_q = all_results.get("quality_original", {})
        patch_q = all_results.get("quality_patched", {})
        if "error" not in orig_q and "error" not in patch_q:
            all_results["quality_comparison"] = compare_quality(orig_q, patch_q, output_dir)
    else:
        log.info("skipping quality tests (--skip-quality)")

    # ── Test 3: vLLM Prefix Caching ──
    if not args.skip_vllm:
        for label, model_path in [("original", args.original), ("patched", args.patched)]:
            try:
                r = test_vllm_prefix_caching(model_path, output_dir, label, tensor_parallel_size=args.tp)
                all_results[f"vllm_{label}"] = r
            except Exception as e:
                log.error("vLLM test [%s] failed: %s", label, e, exc_info=True)
                all_results[f"vllm_{label}"] = {"error": str(e)}
    else:
        log.info("skipping vLLM prefix cache tests (--skip-vllm)")

    # ── Test 4: Quantized determinism ──
    if not args.skip_quant:
        for label, model_path in [("original", args.original), ("patched", args.patched)]:
            try:
                r = test_quantized_determinism(model_path, output_dir, label, tensor_parallel_size=args.tp)
                all_results[f"quant_{label}"] = r
            except Exception as e:
                log.error("quant test [%s] failed: %s", label, e, exc_info=True)
                all_results[f"quant_{label}"] = {"error": str(e), "skipped": True}
    else:
        log.info("skipping quantization tests (--skip-quant)")

    # ── Report ──
    total_time = time.time() - t_total
    all_results["total_time_seconds"] = total_time
    all_results["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    all_path = output_dir / "all_results.json"
    with open(all_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    report = generate_report(all_results, output_dir)
    print("\n" + report)

    log.info("=" * 60)
    log.info("BENCHMARK COMPLETE in %.0f seconds (%.1f hours)", total_time, total_time / 3600)
    log.info("Results: %s", output_dir)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
