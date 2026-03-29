#!/usr/bin/env python3
"""Diagnose: does MoE routing non-determinism break vLLM prefix caching?

Sends the same prompt prefix repeatedly and measures whether vLLM
actually reuses the KV cache. If cache hit rate is high, the routing
is already deterministic enough and CacheReady is unnecessary.

Requirements:
    pip install vllm requests

Usage:
    # Terminal 1: start vLLM server
    python -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen3.5-35B-A3B \
        --quantization fp8 \
        --enable-prefix-caching \
        --max-model-len 4096 \
        --enforce-eager \
        --port 8000

    # Terminal 2: run this script
    python scripts/diagnose_moe_cache.py
"""

import argparse
import json
import time
import statistics
import requests

API_BASE = "http://localhost:8000"

SHARED_PREFIX = (
    "You are an expert software engineer. Below is a large codebase context "
    "that you must reference when answering questions.\n\n"
    "```python\n"
    "import torch\nimport torch.nn as nn\nfrom typing import Optional, Dict, List, Tuple\n\n"
    "class TransformerBlock(nn.Module):\n"
    "    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int):\n"
    "        super().__init__()\n"
    "        self.attention = nn.MultiheadAttention(hidden_size, num_heads)\n"
    "        self.feed_forward = nn.Sequential(\n"
    "            nn.Linear(hidden_size, intermediate_size),\n"
    "            nn.GELU(),\n"
    "            nn.Linear(intermediate_size, hidden_size),\n"
    "        )\n"
    "        self.norm1 = nn.LayerNorm(hidden_size)\n"
    "        self.norm2 = nn.LayerNorm(hidden_size)\n\n"
    "    def forward(self, x, mask=None):\n"
    "        residual = x\n"
    "        x = self.norm1(x)\n"
    "        x, _ = self.attention(x, x, x, attn_mask=mask)\n"
    "        x = residual + x\n"
    "        residual = x\n"
    "        x = self.norm2(x)\n"
    "        x = self.feed_forward(x)\n"
    "        return residual + x\n\n"
    "class TransformerModel(nn.Module):\n"
    "    def __init__(self, vocab_size, hidden_size, num_layers, num_heads, intermediate_size):\n"
    "        super().__init__()\n"
    "        self.embedding = nn.Embedding(vocab_size, hidden_size)\n"
    "        self.layers = nn.ModuleList([\n"
    "            TransformerBlock(hidden_size, num_heads, intermediate_size)\n"
    "            for _ in range(num_layers)\n"
    "        ])\n"
    "        self.output = nn.Linear(hidden_size, vocab_size)\n\n"
    "    def forward(self, input_ids, mask=None):\n"
    "        x = self.embedding(input_ids)\n"
    "        for layer in self.layers:\n"
    "            x = layer(x, mask)\n"
    "        return self.output(x)\n"
    "```\n\n"
)

SUFFIXES = [
    "Question: What does the forward method of TransformerBlock return?",
    "Question: How many linear layers are in the feed_forward network?",
    "Question: What normalization technique is used in this architecture?",
    "Question: What activation function is used in the feed forward network?",
    "Question: What is the purpose of the residual connection?",
    "Question: How is the attention mask used in this implementation?",
    "Question: What shape does the embedding output have?",
    "Question: How many parameters does TransformerBlock have approximately?",
    "Question: What would happen if you removed the layer normalization?",
    "Question: How does this differ from a standard GPT architecture?",
]


def wait_for_server(timeout=300):
    """Wait for vLLM server to be ready."""
    print("Waiting for vLLM server...", end="", flush=True)
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = requests.get(f"{API_BASE}/health", timeout=2)
            if r.status_code == 200:
                print(" ready!")
                return True
        except requests.ConnectionError:
            pass
        print(".", end="", flush=True)
        time.sleep(3)
    print(" TIMEOUT")
    return False


def get_metrics():
    """Fetch vLLM Prometheus metrics."""
    try:
        r = requests.get(f"{API_BASE}/metrics", timeout=5)
        return r.text
    except Exception:
        return ""


def parse_metric(metrics_text, metric_name):
    """Extract a metric value from Prometheus text format."""
    for line in metrics_text.split("\n"):
        if line.startswith(metric_name) and not line.startswith(metric_name + "_"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    return float(parts[-1])
                except ValueError:
                    pass
    return None


def send_completion(prompt, max_tokens=50):
    """Send a completion request and return timing info."""
    t0 = time.time()
    r = requests.post(
        f"{API_BASE}/v1/completions",
        json={
            "model": "Qwen/Qwen3.5-35B-A3B",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        },
        timeout=120,
    )
    elapsed = time.time() - t0
    r.raise_for_status()
    data = r.json()
    usage = data.get("usage", {})
    return {
        "elapsed_s": elapsed,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
    }


def run_diagnostic(n_rounds=3):
    """Run the prefix cache diagnostic."""
    print("\n" + "=" * 70)
    print("  MoE PREFIX CACHE DIAGNOSTIC")
    print("=" * 70)

    if not wait_for_server():
        print("ERROR: vLLM server not reachable at", API_BASE)
        return

    # Warmup: send one request to prime the model
    print("\n[1/4] Warmup...")
    send_completion(SHARED_PREFIX + SUFFIXES[0], max_tokens=10)
    print("  done")

    # Phase 1: Cold requests (different suffixes, same prefix)
    print(f"\n[2/4] Cold requests (first time seeing each suffix)...")
    cold_times = []
    for i, suffix in enumerate(SUFFIXES):
        prompt = SHARED_PREFIX + suffix
        result = send_completion(prompt, max_tokens=30)
        cold_times.append(result["elapsed_s"])
        print(f"  request {i+1}/{len(SUFFIXES)}: {result['elapsed_s']:.3f}s "
              f"({result['prompt_tokens']} prompt tokens)")

    # Get metrics after cold phase
    metrics_after_cold = get_metrics()
    cache_hits_cold = parse_metric(metrics_after_cold, "vllm:prefix_cache_hit_rate")
    
    # Phase 2: Warm requests (repeat the same prompts — prefix should be cached)
    print(f"\n[3/4] Warm requests (repeating same prompts — prefix should be cached)...")
    warm_times = []
    for round_idx in range(n_rounds):
        round_times = []
        for i, suffix in enumerate(SUFFIXES):
            prompt = SHARED_PREFIX + suffix
            result = send_completion(prompt, max_tokens=30)
            round_times.append(result["elapsed_s"])
        warm_times.extend(round_times)
        avg = statistics.mean(round_times)
        print(f"  round {round_idx+1}/{n_rounds}: avg={avg:.3f}s")

    # Get metrics after warm phase
    metrics_after_warm = get_metrics()
    cache_hits_warm = parse_metric(metrics_after_warm, "vllm:prefix_cache_hit_rate")

    # Phase 3: Identical requests (exact same prompt repeated)
    print(f"\n[4/4] Identical requests (exact same prompt 20 times)...")
    identical_prompt = SHARED_PREFIX + SUFFIXES[0]
    identical_times = []
    for i in range(20):
        result = send_completion(identical_prompt, max_tokens=30)
        identical_times.append(result["elapsed_s"])
    
    metrics_final = get_metrics()
    cache_hits_final = parse_metric(metrics_final, "vllm:prefix_cache_hit_rate")

    # Analysis
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)

    cold_avg = statistics.mean(cold_times)
    warm_avg = statistics.mean(warm_times)
    identical_avg = statistics.mean(identical_times[5:])  # skip first few
    first_identical = identical_times[0]

    print(f"\n  Cold requests (first pass):     avg {cold_avg:.3f}s")
    print(f"  Warm requests (repeated):       avg {warm_avg:.3f}s")
    print(f"  Identical requests (same x20):  avg {identical_avg:.3f}s")
    print(f"  Speedup (cold→warm):            {cold_avg/warm_avg:.2f}x")
    print(f"  Speedup (cold→identical):       {cold_avg/identical_avg:.2f}x")

    if cache_hits_cold is not None:
        print(f"\n  Cache hit rate after cold:       {cache_hits_cold:.4f}")
    if cache_hits_warm is not None:
        print(f"  Cache hit rate after warm:       {cache_hits_warm:.4f}")
    if cache_hits_final is not None:
        print(f"  Cache hit rate final:            {cache_hits_final:.4f}")

    # Verdict
    print("\n" + "-" * 70)
    speedup = cold_avg / warm_avg if warm_avg > 0 else 1.0
    if speedup >= 1.3:
        print("  VERDICT: Prefix caching IS working (>1.3x speedup on warm requests)")
        print("  → MoE routing is sufficiently deterministic")
        print("  → CacheReady patch is NOT needed for this model")
    elif speedup >= 1.1:
        print("  VERDICT: Prefix caching PARTIALLY working (1.1-1.3x speedup)")
        print("  → Some cache hits, but not optimal")
        print("  → CacheReady patch MAY help")
    else:
        print("  VERDICT: Prefix caching NOT working (<1.1x speedup)")
        print("  → MoE routing non-determinism is breaking cache reuse")
        print("  → CacheReady patch WOULD help")
    print("-" * 70)

    # Dump raw data
    print("\n  Raw timing data (seconds):")
    print(f"    Cold:      {[round(t, 3) for t in cold_times]}")
    print(f"    Identical: {[round(t, 3) for t in identical_times]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", default=API_BASE)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    API_BASE = args.api_base
    run_diagnostic(n_rounds=args.rounds)
