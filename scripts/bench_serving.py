"""Serving benchmark: measures throughput and latency at increasing concurrency.

Compares baseline vs sculpted models on metrics that matter for production:
  - Prefill latency (time to first token)
  - Decode throughput (tokens/sec)
  - Peak GPU memory at each batch size
  - Max batch size before OOM

Usage:
    python scripts/bench_serving.py --baseline allenai/OLMoE-1B-7B-0924 \
        --sculpted sculpt_out_olmoe_kf90/frontier_0_production/model \
        --output eval_results/serving_bench.json
"""

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    "Explain the concept of distributed systems in simple terms.",
    "Write a Python function that implements binary search.",
    "What are the key differences between TCP and UDP?",
    "Summarize the main ideas behind transformer architectures.",
    "How does garbage collection work in modern programming languages?",
    "Describe the CAP theorem and its implications for database design.",
    "What is the difference between a process and a thread?",
    "Explain how public key cryptography works.",
    "Write a SQL query to find the top 5 customers by revenue.",
    "What are the trade-offs between microservices and monolithic architectures?",
    "How does a neural network learn through backpropagation?",
    "Explain the concept of eventual consistency in distributed databases.",
    "What is the role of attention mechanism in modern NLP models?",
    "Describe how a hash table resolves collisions.",
    "What are the benefits of using containerization for deployment?",
    "Explain the difference between supervised and unsupervised learning.",
    "How does a load balancer distribute traffic across servers?",
    "What is the purpose of a message queue in system architecture?",
    "Describe the principles of RESTful API design.",
    "How does memory management work in Rust compared to C++?",
    "What are the key features of a good caching strategy?",
    "Explain the concept of sharding in database systems.",
    "How does TLS/SSL establish a secure connection?",
    "What is the difference between horizontal and vertical scaling?",
    "Describe the map-reduce programming model.",
    "How do modern CPUs use branch prediction to improve performance?",
    "What is the role of a reverse proxy in web architecture?",
    "Explain the concept of idempotency in API design.",
    "How does a B-tree index improve database query performance?",
    "What are the main challenges in building real-time systems?",
    "Describe how consensus algorithms like Raft work.",
    "What is the difference between optimistic and pessimistic locking?",
]

MAX_NEW_TOKENS = 128
WARMUP_ITERS = 2
BENCH_ITERS = 5


def measure_model(
    model_path: str,
    batch_sizes: List[int],
    device: str = "cuda",
    trust_remote_code: bool = True,
) -> Dict:
    """Load model and benchmark at each batch size."""
    print(f"\n{'='*70}")
    print(f"Benchmarking: {model_path}")
    print(f"{'='*70}")

    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )
    model.eval()

    load_mem_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"  Model loaded: {load_mem_gb:.2f} GB GPU memory")

    results = {
        "model": model_path,
        "load_memory_gb": round(load_mem_gb, 2),
        "batch_results": [],
    }

    for bs in batch_sizes:
        torch.cuda.reset_peak_memory_stats()
        prompts = (PROMPTS * ((bs // len(PROMPTS)) + 1))[:bs]

        try:
            inputs = tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=True,
                max_length=256,
            ).to(device)
            input_len = inputs["input_ids"].shape[1]

            # Warmup
            for _ in range(WARMUP_ITERS):
                with torch.no_grad():
                    model.generate(
                        **inputs, max_new_tokens=16,
                        do_sample=False, use_cache=True,
                    )

            # Prefill benchmark (just the first forward pass)
            prefill_times = []
            for _ in range(BENCH_ITERS):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    outputs = model(**inputs)
                torch.cuda.synchronize()
                prefill_times.append(time.perf_counter() - t0)
            del outputs

            # Full generation benchmark
            gen_times = []
            total_new_tokens = []
            for _ in range(BENCH_ITERS):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    out = model.generate(
                        **inputs, max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=False, use_cache=True,
                    )
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0
                gen_times.append(elapsed)
                new_toks = out.shape[1] - input_len
                total_new_tokens.append(new_toks * bs)
                del out

            peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9

            prefill_ms = sorted(prefill_times)
            gen_sorted = sorted(gen_times)
            tps_values = [t / g for t, g in zip(total_new_tokens, gen_times)]

            batch_result = {
                "batch_size": bs,
                "prefill_ms_p50": round(prefill_ms[len(prefill_ms) // 2] * 1000, 1),
                "prefill_ms_p95": round(prefill_ms[-1] * 1000, 1),
                "generation_sec_p50": round(gen_sorted[len(gen_sorted) // 2], 2),
                "total_tokens_per_sec": round(sum(tps_values) / len(tps_values), 1),
                "peak_memory_gb": round(peak_mem_gb, 2),
            }
            results["batch_results"].append(batch_result)

            print(f"  BS={bs:3d}  prefill={batch_result['prefill_ms_p50']:7.1f}ms  "
                  f"tps={batch_result['total_tokens_per_sec']:7.1f}  "
                  f"peak={batch_result['peak_memory_gb']:.1f}GB")

        except torch.cuda.OutOfMemoryError:
            print(f"  BS={bs:3d}  OOM — max batch size is {batch_sizes[batch_sizes.index(bs) - 1] if batch_sizes.index(bs) > 0 else 0}")
            results["max_batch_size"] = batch_sizes[batch_sizes.index(bs) - 1] if batch_sizes.index(bs) > 0 else 0
            torch.cuda.empty_cache()
            break
    else:
        results["max_batch_size"] = batch_sizes[-1]

    del model
    torch.cuda.empty_cache()
    gc.collect()
    return results


def main():
    parser = argparse.ArgumentParser(description="Serving benchmark")
    parser.add_argument("--baseline", required=True, help="Baseline model path or HF id")
    parser.add_argument("--sculpted", required=True, help="Sculpted model path")
    parser.add_argument("--output", default="serving_bench.json", help="Output JSON path")
    parser.add_argument("--batch-sizes", default="1,2,4,8,16,32,64", help="Comma-separated batch sizes")
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    baseline = measure_model(args.baseline, batch_sizes)
    sculpted = measure_model(args.sculpted, batch_sizes)

    # Build comparison table
    print(f"\n{'='*70}")
    print("COMPARISON")
    print(f"{'='*70}")
    print(f"{'Metric':<30s}  {'Baseline':>12s}  {'Sculpted':>12s}  {'Delta':>10s}")
    print("-" * 70)
    print(f"{'Load memory (GB)':<30s}  {baseline['load_memory_gb']:>12.2f}  "
          f"{sculpted['load_memory_gb']:>12.2f}  "
          f"{sculpted['load_memory_gb'] - baseline['load_memory_gb']:>+10.2f}")
    print(f"{'Max batch size':<30s}  {baseline['max_batch_size']:>12d}  "
          f"{sculpted['max_batch_size']:>12d}  "
          f"{sculpted['max_batch_size'] - baseline['max_batch_size']:>+10d}")

    for bs_result_b in baseline["batch_results"]:
        bs = bs_result_b["batch_size"]
        bs_result_s = next(
            (r for r in sculpted["batch_results"] if r["batch_size"] == bs), None,
        )
        if bs_result_s is None:
            continue
        tps_b = bs_result_b["total_tokens_per_sec"]
        tps_s = bs_result_s["total_tokens_per_sec"]
        pct = ((tps_s - tps_b) / tps_b * 100) if tps_b > 0 else 0
        print(f"{'  BS=' + str(bs) + ' tok/s':<30s}  {tps_b:>12.1f}  {tps_s:>12.1f}  {pct:>+9.1f}%")

        mem_b = bs_result_b["peak_memory_gb"]
        mem_s = bs_result_s["peak_memory_gb"]
        print(f"{'  BS=' + str(bs) + ' peak mem (GB)':<30s}  {mem_b:>12.1f}  {mem_s:>12.1f}  {mem_s - mem_b:>+10.1f}")

    output = {"baseline": baseline, "sculpted": sculpted}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
