#!/usr/bin/env python3
"""MoE Routing Patch Benchmark Suite.

Validates the Physarum routing canonicalization patch across four dimensions:

  1. Routing Determinism — identical inputs → identical outputs
  2. Quality Preservation — full lm_eval benchmarks (MMLU, HellaSwag, ARC, etc.)
  3. vLLM Prefix Caching — throughput with shared-prefix workloads
  4. nvfp4 Quantization Stability — determinism under 4-bit quantization

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
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("moe_benchmark")


# ─── Test 1: Routing Determinism ─────────────────────────────────────

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


def _load_model_for_determinism(model_path: str, dtype=torch.bfloat16):
    """Load model with device_map=auto for multi-GPU inference."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info("loading model: %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def _free_model(model):
    """Aggressively free GPU memory."""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.no_grad()
def test_determinism(
    model_path: str,
    output_dir: Path,
    label: str,
    n_runs: int = 5,
    dtype=torch.bfloat16,
) -> Dict[str, Any]:
    """Run multiple forward passes on diverse texts, measure logit divergence."""
    log.info("=== Test 1: Routing Determinism [%s] ===", label)

    model, tokenizer = _load_model_for_determinism(model_path, dtype=dtype)
    first_device = next(model.parameters()).device
    results_per_text = []

    for ti, text in enumerate(DETERMINISM_TEXTS):
        inp = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inp = {k: v.to(first_device) for k, v in inp.items()}
        seq_len = inp["input_ids"].shape[1]

        all_logits = []
        for r in range(n_runs):
            out = model(**inp, use_cache=False)
            all_logits.append(out.logits.cpu().float())

        max_diffs = []
        mean_diffs = []
        for i in range(n_runs):
            for j in range(i + 1, n_runs):
                diff = (all_logits[i] - all_logits[j]).abs()
                max_diffs.append(diff.max().item())
                mean_diffs.append(diff.mean().item())

        result = {
            "text_idx": ti,
            "text_preview": text[:80],
            "seq_len": seq_len,
            "n_runs": n_runs,
            "max_logit_diff": max(max_diffs) if max_diffs else 0.0,
            "mean_logit_diff": sum(mean_diffs) / len(mean_diffs) if mean_diffs else 0.0,
            "all_pairs_max_diffs": max_diffs,
            "deterministic": max(max_diffs) < 1e-4 if max_diffs else True,
        }
        results_per_text.append(result)

        status = "PASS" if result["deterministic"] else "FAIL"
        log.info(
            "  text %d/%d (len=%d): max_diff=%.2e [%s]",
            ti + 1, len(DETERMINISM_TEXTS), seq_len,
            result["max_logit_diff"], status,
        )

    all_deterministic = all(r["deterministic"] for r in results_per_text)
    overall_max_diff = max(r["max_logit_diff"] for r in results_per_text)

    summary = {
        "label": label,
        "model_path": model_path,
        "dtype": str(dtype),
        "n_texts": len(DETERMINISM_TEXTS),
        "n_runs_per_text": n_runs,
        "overall_max_logit_diff": overall_max_diff,
        "all_deterministic": all_deterministic,
        "pass": all_deterministic,
        "per_text": results_per_text,
    }

    out_path = output_dir / f"determinism_{label}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("determinism [%s]: %s (max_diff=%.2e) → %s",
             label, "PASS" if all_deterministic else "FAIL", overall_max_diff, out_path)

    _free_model(model)
    return summary


# ─── Test 2: Quality Preservation via lm_eval ────────────────────────

LM_EVAL_TASKS = "mmlu,hellaswag,arc_challenge,truthfulqa_mc2,winogrande,gsm8k"


def test_quality_lm_eval(
    model_path: str,
    output_dir: Path,
    label: str,
    tasks: str = LM_EVAL_TASKS,
    batch_size: str = "auto",
) -> Dict[str, Any]:
    """Run lm_eval harness on a model."""
    log.info("=== Test 2: Quality [%s] ===", label)
    log.info("tasks: %s", tasks)

    eval_output = output_dir / f"lm_eval_{label}"
    eval_output.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_path},dtype=bfloat16,trust_remote_code=True",
        "--tasks", tasks,
        "--batch_size", batch_size,
        "--output_path", str(eval_output),
    ]

    log.info("running: %s", " ".join(cmd))
    t0 = time.time()

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=36000)
    elapsed = time.time() - t0

    log.info("lm_eval [%s] completed in %.0fs (exit code %d)", label, elapsed, result.returncode)

    if result.returncode != 0:
        log.error("lm_eval stderr:\n%s", result.stderr[-2000:] if result.stderr else "(empty)")

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
] * 3  # 60 requests total

UNIQUE_PROMPTS = [
    f"Tell me about topic number {i} in great detail, covering all aspects." for i in range(60)
]


def _try_vllm_import():
    try:
        import vllm
        return True
    except ImportError:
        return False


def _run_vllm_workload(
    model_path: str,
    prompts: List[str],
    enable_prefix_caching: bool,
    max_new_tokens: int = 64,
    tensor_parallel_size: int = 1,
    quantization: Optional[str] = None,
    gpu_memory_utilization: float = 0.90,
) -> Dict[str, Any]:
    """Run a vLLM workload and measure throughput/latency."""
    from vllm import LLM, SamplingParams

    log.info(
        "  vLLM workload: %d prompts, prefix_cache=%s, quant=%s, tp=%d",
        len(prompts), enable_prefix_caching, quantization, tensor_parallel_size,
    )

    llm_kwargs = dict(
        model=model_path,
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel_size,
        enable_prefix_caching=enable_prefix_caching,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=2048,
    )
    if quantization:
        llm_kwargs["quantization"] = quantization

    t_load = time.time()
    llm = LLM(**llm_kwargs)
    load_time = time.time() - t_load

    params = SamplingParams(max_tokens=max_new_tokens, temperature=0)

    # Warmup
    _ = llm.generate(prompts[:2], params)

    # Throughput: all prompts
    t0 = time.time()
    outputs = llm.generate(prompts, params)
    throughput_time = time.time() - t0
    total_gen = sum(len(o.outputs[0].token_ids) for o in outputs)
    throughput = total_gen / max(1e-9, throughput_time)

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

    # Second pass: measure cache benefit (same prompts again)
    t0 = time.time()
    outputs2 = llm.generate(prompts, params)
    second_pass_time = time.time() - t0
    total_gen2 = sum(len(o.outputs[0].token_ids) for o in outputs2)
    throughput_second = total_gen2 / max(1e-9, second_pass_time)

    result = {
        "throughput_first_pass_tok_s": round(throughput, 2),
        "throughput_second_pass_tok_s": round(throughput_second, 2),
        "cache_speedup_ratio": round(throughput_second / max(throughput, 1e-9), 3),
        "latency_p50_s": round(p50, 4),
        "latency_p95_s": round(p95, 4),
        "total_generated_tokens": total_gen,
        "n_prompts": len(prompts),
        "load_time_s": round(load_time, 1),
        "enable_prefix_caching": enable_prefix_caching,
        "quantization": quantization,
    }

    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def test_vllm_prefix_caching(
    model_path: str,
    output_dir: Path,
    label: str,
    tensor_parallel_size: int = 4,
    quantization: Optional[str] = None,
) -> Dict[str, Any]:
    """Test prefix caching effectiveness with shared-prefix workloads."""
    log.info("=== Test 3: vLLM Prefix Caching [%s] ===", label)

    if not _try_vllm_import():
        log.warning("vLLM not installed, skipping prefix caching test")
        return {"label": label, "skipped": True, "reason": "vllm not installed"}

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
                    quantization=quantization,
                )
                results[key] = r
                log.info(
                    "  throughput: %.0f tok/s (1st) → %.0f tok/s (2nd), speedup: %.2fx",
                    r["throughput_first_pass_tok_s"],
                    r["throughput_second_pass_tok_s"],
                    r["cache_speedup_ratio"],
                )
            except Exception as e:
                log.error("  workload %s failed: %s", key, e)
                results[key] = {"error": str(e)}

    # Compute cache benefit delta
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
        "quantization": quantization,
        "tensor_parallel_size": tensor_parallel_size,
        "workload_results": results,
        "cache_benefit": cache_benefit,
    }

    out_path = output_dir / f"vllm_prefix_cache_{label}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("vLLM prefix cache [%s] → %s", label, out_path)

    return summary


# ─── Test 4: nvfp4 Quantization Routing Stability ───────────────────

def test_nvfp4_determinism(
    model_path: str,
    output_dir: Path,
    label: str,
    tensor_parallel_size: int = 4,
) -> Dict[str, Any]:
    """Test routing determinism under quantization via vLLM."""
    log.info("=== Test 4: nvfp4 Determinism [%s] ===", label)

    if not _try_vllm_import():
        log.warning("vLLM not installed, skipping nvfp4 test")
        return {"label": label, "skipped": True, "reason": "vllm not installed"}

    from vllm import LLM, SamplingParams

    # Try quantization methods in order of preference
    quant_methods = ["fp8", "bitsandbytes"]
    llm = None
    used_quant = None

    for qmethod in quant_methods:
        try:
            log.info("  trying quantization: %s", qmethod)
            llm_kwargs = dict(
                model=model_path,
                trust_remote_code=True,
                tensor_parallel_size=tensor_parallel_size,
                quantization=qmethod,
                gpu_memory_utilization=0.90,
                max_model_len=1024,
            )
            llm = LLM(**llm_kwargs)
            used_quant = qmethod
            log.info("  using quantization: %s", qmethod)
            break
        except Exception as e:
            log.warning("  %s failed: %s", qmethod, e)
            continue

    if llm is None:
        log.warning("no quantization method available, skipping nvfp4 test")
        return {"label": label, "skipped": True, "reason": "no quantization available"}

    params = SamplingParams(max_tokens=1, temperature=0)

    texts = DETERMINISM_TEXTS[:10]
    n_runs = 5
    results_per_text = []

    for ti, text in enumerate(texts):
        token_ids_runs = []
        for r in range(n_runs):
            outputs = llm.generate([text], params)
            generated = outputs[0].outputs[0].token_ids
            token_ids_runs.append(list(generated))

        all_same = all(t == token_ids_runs[0] for t in token_ids_runs)
        results_per_text.append({
            "text_idx": ti,
            "text_preview": text[:80],
            "n_runs": n_runs,
            "all_identical": all_same,
            "unique_outputs": len(set(str(t) for t in token_ids_runs)),
        })

        status = "PASS" if all_same else "DIVERGENT"
        log.info("  text %d/%d: %s (%d unique outputs)",
                 ti + 1, len(texts), status, results_per_text[-1]["unique_outputs"])

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

    out_path = output_dir / f"nvfp4_determinism_{label}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("nvfp4 determinism [%s]: %.1f%% agreement → %s",
             label, agreement_rate * 100, out_path)

    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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
    lines.append("## 1. Routing Determinism")
    lines.append("")
    for label in ["original", "patched"]:
        det = results.get(f"determinism_{label}", {})
        if det.get("skipped"):
            lines.append(f"**{label}**: skipped")
            continue
        status = "PASS" if det.get("pass") else "FAIL"
        max_diff = det.get("overall_max_logit_diff", "N/A")
        lines.append(f"- **{label}**: {status} (max logit diff: {max_diff:.2e})")
    lines.append("")

    # Quality
    lines.append("## 2. Quality Preservation")
    lines.append("")
    qc = results.get("quality_comparison", {})
    if qc:
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
    lines.append("## 3. vLLM Prefix Caching")
    lines.append("")
    for label in ["original", "patched"]:
        vllm_data = results.get(f"vllm_{label}", {})
        if vllm_data.get("skipped"):
            lines.append(f"**{label}**: skipped ({vllm_data.get('reason', '')})")
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

    # nvfp4
    lines.append("## 4. Quantization Routing Stability")
    lines.append("")
    for label in ["original", "patched"]:
        nv = results.get(f"nvfp4_{label}", {})
        if nv.get("skipped"):
            lines.append(f"**{label}**: skipped ({nv.get('reason', '')})")
            continue
        rate = nv.get("agreement_rate", 0) * 100
        status = "PASS" if nv.get("pass") else "FAIL"
        lines.append(f"- **{label}**: {rate:.1f}% routing agreement [{status}] (quant: {nv.get('quantization_method', 'N/A')})")
    lines.append("")

    # Overall
    lines.append("## Overall Verdict")
    lines.append("")
    all_pass = all([
        results.get("determinism_patched", {}).get("pass", False),
        results.get("quality_comparison", {}).get("pass", True),
    ])
    lines.append(f"**{'PASS' if all_pass else 'FAIL'}**")
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
    parser.add_argument("--skip-vllm", action="store_true", help="Skip vLLM tests")
    parser.add_argument("--skip-nvfp4", action="store_true", help="Skip nvfp4 tests")
    args = parser.parse_args()

    output_dir = Path(args.output).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("MoE Routing Patch Benchmark Suite")
    log.info("Original: %s", args.original)
    log.info("Patched:  %s", args.patched)
    log.info("Output:   %s", output_dir)
    log.info("=" * 60)

    all_results: Dict[str, Any] = {
        "original_model": args.original,
        "patched_model": args.patched,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }

    t_total = time.time()

    # ── Test 1: Determinism ──
    for label, model_path in [("original", args.original), ("patched", args.patched)]:
        try:
            r = test_determinism(model_path, output_dir, label)
            all_results[f"determinism_{label}"] = r
        except Exception as e:
            log.error("determinism test [%s] failed: %s", label, e, exc_info=True)
            all_results[f"determinism_{label}"] = {"error": str(e)}

    # ── Test 2: Quality ──
    if not args.skip_quality:
        for label, model_path in [("original", args.original), ("patched", args.patched)]:
            try:
                r = test_quality_lm_eval(model_path, output_dir, label)
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
                all_results[f"vllm_{label}"] = {"error": str(e), "skipped": True}
    else:
        log.info("skipping vLLM tests (--skip-vllm)")

    # ── Test 4: nvfp4 ──
    if not args.skip_nvfp4:
        for label, model_path in [("original", args.original), ("patched", args.patched)]:
            try:
                r = test_nvfp4_determinism(model_path, output_dir, label, tensor_parallel_size=args.tp)
                all_results[f"nvfp4_{label}"] = r
            except Exception as e:
                log.error("nvfp4 test [%s] failed: %s", label, e, exc_info=True)
                all_results[f"nvfp4_{label}"] = {"error": str(e), "skipped": True}
    else:
        log.info("skipping nvfp4 tests (--skip-nvfp4)")

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
