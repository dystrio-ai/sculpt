#!/usr/bin/env python3
"""Server-mode prefix caching benchmark.

Starts a vLLM OpenAI-compatible server and hits it with concurrent
requests sharing a system prompt prefix — the real-world scenario
where prefix caching matters.

Tests FP8 quantization + online serving + concurrent load, matching
the conditions reported by users experiencing prefix caching issues.

Usage:
    python scripts/bench_server_prefix.py \
        --model Qwen/Qwen3.5-122B-A10B-FP8 \
        --tp 4 \
        --output /ephemeral/server_bench
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [bench] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("bench")

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
]


def start_server(
    model: str,
    tp: int,
    port: int,
    prefix_caching: bool,
    quantization: str | None = None,
) -> subprocess.Popen:
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--tensor-parallel-size", str(tp),
        "--max-model-len", "2048",
        "--enforce-eager",
        "--port", str(port),
        "--disable-log-requests",
    ]
    if prefix_caching:
        cmd.append("--enable-prefix-caching")
    if quantization:
        cmd.extend(["--quantization", quantization])

    log.info("starting server: %s", " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc


def wait_for_server(port: int, timeout: int = 600) -> bool:
    """Poll until server is ready or timeout."""
    import urllib.request
    url = f"http://localhost:{port}/health"
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except Exception:
            time.sleep(5)
    return False


def kill_server(proc: subprocess.Popen):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        pass
    try:
        proc.terminate()
        proc.wait(timeout=10)
    except Exception:
        proc.kill()
    subprocess.run(
        "ps aux | grep '[v]llm.entrypoints' | awk '{print $2}' | xargs -r kill -9",
        shell=True, capture_output=True,
    )
    subprocess.run(
        "ps aux | grep '[m]ultiproc_executor' | awk '{print $2}' | xargs -r kill -9",
        shell=True, capture_output=True,
    )
    time.sleep(10)


async def send_request(
    session,
    port: int,
    query: str,
    max_tokens: int = 64,
) -> Dict[str, Any]:
    """Send one chat completion request, return timing + token count."""
    import aiohttp
    url = f"http://localhost:{port}/v1/chat/completions"
    payload = {
        "model": "default",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
    }

    t0 = time.time()
    try:
        async with session.post(url, json=payload) as resp:
            data = await resp.json()
            elapsed = time.time() - t0
            tokens = data.get("usage", {}).get("completion_tokens", 0)
            return {"tokens": tokens, "elapsed": elapsed, "ok": True}
    except Exception as e:
        return {"tokens": 0, "elapsed": time.time() - t0, "ok": False, "error": str(e)}


async def run_concurrent_requests(
    port: int,
    queries: List[str],
    concurrency: int = 8,
    max_tokens: int = 64,
) -> List[Dict[str, Any]]:
    """Send queries concurrently with limited parallelism."""
    import aiohttp

    semaphore = asyncio.Semaphore(concurrency)
    results = []

    async def _limited(q):
        async with semaphore:
            return await send_request(session, port, q, max_tokens)

    async with aiohttp.ClientSession() as session:
        tasks = [_limited(q) for q in queries]
        results = await asyncio.gather(*tasks)

    return list(results)


def measure_server(
    port: int,
    n_queries: int = 200,
    concurrency: int = 8,
    n_warmup: int = 20,
    n_rounds: int = 3,
    max_tokens: int = 64,
) -> Dict[str, Any]:
    """Run warmup + measurement rounds against a running server."""
    queries = [QUERIES[i % len(QUERIES)] for i in range(n_queries)]
    warmup_queries = [QUERIES[i % len(QUERIES)] for i in range(n_warmup)]

    log.info("  warmup: %d requests (concurrency=%d)...", n_warmup, concurrency)
    asyncio.run(run_concurrent_requests(port, warmup_queries, concurrency, max_tokens))

    round_results = []
    for r in range(n_rounds):
        t0 = time.time()
        results = asyncio.run(
            run_concurrent_requests(port, queries, concurrency, max_tokens)
        )
        elapsed = time.time() - t0
        total_tokens = sum(r["tokens"] for r in results)
        ok_count = sum(1 for r in results if r["ok"])
        tps = total_tokens / max(elapsed, 1e-9)

        per_request_latencies = [r["elapsed"] for r in results if r["ok"]]
        avg_latency = sum(per_request_latencies) / len(per_request_latencies) if per_request_latencies else 0
        p50 = sorted(per_request_latencies)[len(per_request_latencies) // 2] if per_request_latencies else 0
        p95_idx = min(int(len(per_request_latencies) * 0.95), len(per_request_latencies) - 1)
        p95 = sorted(per_request_latencies)[p95_idx] if per_request_latencies else 0

        round_results.append({
            "tps": round(tps, 1),
            "total_tokens": total_tokens,
            "elapsed": round(elapsed, 2),
            "ok": ok_count,
            "failed": n_queries - ok_count,
            "avg_latency_s": round(avg_latency, 3),
            "p50_latency_s": round(p50, 3),
            "p95_latency_s": round(p95, 3),
        })

        log.info(
            "  round %d/%d: %d tok in %.1fs = %.0f tok/s | "
            "latency p50=%.2fs p95=%.2fs | %d/%d ok",
            r + 1, n_rounds, total_tokens, elapsed, tps,
            p50, p95, ok_count, n_queries,
        )

    tps_values = [r["tps"] for r in round_results]
    import statistics
    return {
        "rounds": round_results,
        "median_tps": round(statistics.median(tps_values), 1),
        "min_tps": round(min(tps_values), 1),
        "max_tps": round(max(tps_values), 1),
        "n_queries": n_queries,
        "concurrency": concurrency,
        "n_rounds": n_rounds,
    }


def main():
    parser = argparse.ArgumentParser(description="Server-mode prefix caching benchmark")
    parser.add_argument("--model", required=True, help="Model HF ID or local path")
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--output", default="~/server_bench_results")
    parser.add_argument("--n-queries", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--n-rounds", type=int, default=3)
    parser.add_argument("--n-warmup", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument(
        "--quantization", default=None,
        help="Quantization method (e.g. fp8). None = use model's native format.",
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

    log.info("=" * 60)
    log.info("Server-mode Prefix Caching Benchmark")
    log.info("Model: %s", args.model)
    log.info("TP: %d | Queries: %d | Concurrency: %d | Rounds: %d",
             args.tp, args.n_queries, args.concurrency, args.n_rounds)
    log.info("Quantization: %s", args.quantization or "native")
    log.info("GPU: %s", gpu_info)
    log.info("=" * 60)

    all_results: Dict[str, Any] = {
        "model": args.model,
        "tp": args.tp,
        "quantization": args.quantization,
        "gpu_info": gpu_info,
        "concurrency": args.concurrency,
        "n_queries": args.n_queries,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }

    for cache_enabled in [False, True]:
        cache_label = "cache_on" if cache_enabled else "cache_off"
        log.info("=== prefix_caching=%s ===", cache_enabled)

        proc = start_server(
            model=args.model,
            tp=args.tp,
            port=args.port,
            prefix_caching=cache_enabled,
            quantization=args.quantization,
        )

        log.info("  waiting for server to be ready (up to 10 min)...")
        if not wait_for_server(args.port, timeout=600):
            log.error("  server failed to start!")
            kill_server(proc)
            all_results[cache_label] = {"error": "server failed to start"}
            continue

        log.info("  server ready!")

        try:
            result = measure_server(
                port=args.port,
                n_queries=args.n_queries,
                concurrency=args.concurrency,
                n_warmup=args.n_warmup,
                n_rounds=args.n_rounds,
                max_tokens=args.max_tokens,
            )
            result["prefix_caching"] = cache_enabled
            all_results[cache_label] = result
            log.info("[%s] median %.0f tok/s (range: %.0f–%.0f)",
                     cache_label, result["median_tps"],
                     result["min_tps"], result["max_tps"])
        except Exception as e:
            log.error("[%s] FAILED: %s", cache_label, e, exc_info=True)
            all_results[cache_label] = {"error": str(e)}

        log.info("  stopping server...")
        kill_server(proc)

    all_results["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    # Report
    lines = [
        "# Server-Mode Prefix Caching Benchmark",
        "",
        f"**Model**: {args.model}",
        f"**Quantization**: {args.quantization or 'native'}",
        f"**Hardware**: {gpu_info}",
        f"**TP**: {args.tp}",
        f"**Concurrency**: {args.concurrency}",
        f"**Queries per round**: {args.n_queries}",
        f"**Rounds**: {args.n_rounds}",
        "",
        "## Results",
        "",
        "| Prefix Caching | Throughput (tok/s) | p50 Latency | p95 Latency |",
        "|----------------|-------------------|-------------|-------------|",
    ]

    for label in ["cache_off", "cache_on"]:
        d = all_results.get(label, {})
        if "error" in d:
            lines.append(f"| {label} | ERROR: {d['error']} | — | — |")
        elif "median_tps" in d:
            rounds = d.get("rounds", [])
            mid = rounds[len(rounds) // 2] if rounds else {}
            lines.append(
                f"| {label} | {d['median_tps']:.0f} | "
                f"{mid.get('p50_latency_s', '—')}s | "
                f"{mid.get('p95_latency_s', '—')}s |"
            )

    off_tps = all_results.get("cache_off", {}).get("median_tps", 0)
    on_tps = all_results.get("cache_on", {}).get("median_tps", 0)
    if off_tps > 0 and on_tps > 0:
        effect = on_tps / off_tps
        lines.extend([
            "",
            f"**Cache effect**: {effect:.2f}x ({off_tps:.0f} → {on_tps:.0f} tok/s)",
        ])

    lines.append("")
    report = "\n".join(lines)

    report_path = output_dir / "server_bench_report.md"
    with open(report_path, "w") as f:
        f.write(report)

    raw_path = output_dir / "server_bench_results.json"
    with open(raw_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + report)
    log.info("report: %s", report_path)
    log.info("DONE")


if __name__ == "__main__":
    main()
