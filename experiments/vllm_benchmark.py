#!/usr/bin/env python3
"""Optional vLLM serving benchmark.

Measures throughput and decode latency for a model served through vLLM.
Requires: pip install vllm

Standalone usage:
    python experiments/vllm_benchmark.py --model-path Qwen/Qwen2-0.5B

From the harness:
    python experiments/multilayer_experiment.py --enable-vllm
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List


def is_available() -> bool:
    """Return True if vLLM is importable."""
    try:
        import vllm  # noqa: F401
        return True
    except ImportError:
        return False


def benchmark_model(
    model_path: str,
    prompts: List[str],
    max_new_tokens: int = 128,
    n_throughput_prompts: int = 128,
    n_latency_prompts: int = 20,
) -> Dict[str, Any]:
    """Benchmark a model using vLLM.

    Args:
        model_path: HuggingFace model ID or local directory.
        prompts: Real text prompts for generation.
        max_new_tokens: Max tokens to generate per prompt.
        n_throughput_prompts: Prompt count for throughput measurement.
        n_latency_prompts: Prompt count for latency measurement.

    Returns:
        Dictionary with throughput_tokens_per_sec, latency_p50_s, latency_p95_s.
    """
    from vllm import LLM, SamplingParams

    llm = LLM(model=model_path, trust_remote_code=True)
    params = SamplingParams(max_tokens=max_new_tokens, temperature=0)

    tp_prompts = prompts[:n_throughput_prompts]
    lat_prompts = prompts[:n_latency_prompts]

    # Warmup
    _ = llm.generate(tp_prompts[:4], params)

    # ── Throughput: all prompts at once ────────────────────────────────────────
    t0 = time.time()
    outputs = llm.generate(tp_prompts, params)
    dt = time.time() - t0
    total_gen = sum(len(o.outputs[0].token_ids) for o in outputs)
    throughput = total_gen / max(1e-9, dt)

    # ── Latency: single-prompt decode, collect p50/p95 ────────────────────────
    latencies: list[float] = []
    for prompt in lat_prompts:
        t0 = time.time()
        _ = llm.generate([prompt], params)
        latencies.append(time.time() - t0)

    latencies.sort()
    p50 = latencies[len(latencies) // 2] if latencies else 0.0
    p95_idx = min(int(len(latencies) * 0.95), len(latencies) - 1)
    p95 = latencies[p95_idx] if latencies else 0.0

    # Cleanup
    del llm
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return {
        "throughput_tokens_per_sec": round(throughput, 2),
        "latency_p50_s": round(p50, 4),
        "latency_p95_s": round(p95, 4),
        "n_throughput_prompts": len(tp_prompts),
        "n_latency_prompts": len(lat_prompts),
        "total_generated_tokens": total_gen,
        "max_new_tokens": max_new_tokens,
        "model_path": model_path,
    }


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="vLLM serving benchmark")
    p.add_argument("--model-path", required=True, help="HF model ID or local path")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--n-throughput-prompts", type=int, default=128)
    p.add_argument("--n-latency-prompts", type=int, default=20)
    p.add_argument("--output", default=None, help="Output JSON path")
    args = p.parse_args()

    if not is_available():
        print("vLLM not installed.  Install with: pip install vllm")
        raise SystemExit(1)

    prompts = [
        "The meaning of artificial intelligence is fundamentally about",
        "In recent developments in machine learning research, the focus has shifted",
        "The architecture of modern transformer models consists of",
    ] * (args.n_throughput_prompts // 3 + 1)
    prompts = prompts[: args.n_throughput_prompts]

    result = benchmark_model(
        args.model_path,
        prompts,
        max_new_tokens=args.max_new_tokens,
        n_throughput_prompts=args.n_throughput_prompts,
        n_latency_prompts=args.n_latency_prompts,
    )

    print(json.dumps(result, indent=2))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Written to {args.output}")


if __name__ == "__main__":
    main()
