#!/usr/bin/env python3
"""Push sculpted model zoo to HuggingFace with generated model cards.

Usage:
    export HF_TOKEN=hf_...
    python3 scripts/push_to_hf.py --zoo-dir /data/zoo --org dystrio --dry-run
    python3 scripts/push_to_hf.py --zoo-dir /data/zoo --org dystrio
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

TIER_USE_CASES = {
    "default": "Zero-regret: quality preserved, smaller footprint",
    "production": "Practical savings with modest quality tradeoff",
    "throughput": "Maximum usable compression for speed/edge",
    "experimental": "Boundary exploration, maximum structural compression",
    "frontier": "Research only, extreme compression",
}

TIER_ORDER = ["default", "production", "throughput", "experimental", "frontier"]

MODEL_ZOO = [
    {
        "base_model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "base_display": "Mistral 7B Instruct v0.3",
        "model_short": "Mistral-7B-Instruct-v0.3",
        "arch_tag": "mistral",
        "compile_dir": "mistral-7b-instruct-f5",
        "bench_dir": "bench_mistral",
    },
    {
        "base_model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "base_display": "Llama 3.1 8B Instruct",
        "model_short": "Llama-3.1-8B-Instruct",
        "arch_tag": "llama",
        "compile_dir": "llama-3.1-8b-instruct-f4",
        "bench_dir": "bench_llama",
    },
    {
        "base_model_id": "Qwen/Qwen2.5-7B-Instruct",
        "base_display": "Qwen 2.5 7B Instruct",
        "model_short": "Qwen2.5-7B-Instruct",
        "arch_tag": "qwen",
        "compile_dir": "qwen2.5-7b-instruct-f3",
        "bench_dir": "bench_qwen",
    },
]


def _load_bench(bench_csv: Path) -> Dict[str, Dict[str, Any]]:
    """Load benchmarks.csv into {model_id: {workload: row}}."""
    results: Dict[str, Dict[str, Any]] = {}
    with open(bench_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            mid = row["model_id"]
            wl = row["workload"]
            if mid not in results:
                results[mid] = {}
            results[mid][wl] = row
    return results


def _fmt_params(n: str) -> str:
    try:
        p = int(n)
        if p >= 1e9:
            return f"{p / 1e9:.2f}B"
        return f"{p / 1e6:.0f}M"
    except (ValueError, TypeError):
        return str(n)


def _pct_change(new: float, old: float) -> str:
    if old == 0:
        return ""
    pct = (new - old) / old * 100
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.0f}%"


def _build_benchmark_table(bench: Dict[str, Dict[str, Any]], tiers: List[dict],
                           base_model_id: str) -> str:
    lines = [
        "| Model | PPL | PPL Ratio | Weights (GB) | Chat Prefill TPS | RAG TTFT p95 (ms) | Decode TPS |",
        "|-------|-----|-----------|-------------|------------------|-------------------|------------|",
    ]
    base_wiki = bench.get(base_model_id, {}).get("wikitext", {})
    base_chat = bench.get(base_model_id, {}).get("chat", {})
    base_rag = bench.get(base_model_id, {}).get("rag", {})

    def _row(label, mid):
        wiki = bench.get(mid, {}).get("wikitext", {})
        chat = bench.get(mid, {}).get("chat", {})
        rag = bench.get(mid, {}).get("rag", {})
        ppl = wiki.get("ppl_wikitext", "")
        ratio = wiki.get("ppl_ratio", chat.get("ppl_ratio", ""))
        wgb = wiki.get("weights_gb", chat.get("weights_gb", ""))
        ptps = chat.get("prefill_tokens_per_sec", "")
        ttft = rag.get("ttft_ms_p95", "")
        dtps = chat.get("decode_tokens_per_sec", "")
        return f"| {label} | {ppl} | {ratio} | {wgb} | {ptps} | {ttft} | {dtps} |"

    lines.append(_row("**Baseline**", base_model_id))
    for t in tiers:
        tier_name = t["tier_name"]
        model_path = t["model_path"]
        lines.append(_row(f"**sculpt-{tier_name}**", model_path))

    return "\n".join(lines)


def _build_tier_links(tiers: List[dict], org: str, model_short: str,
                      current_tier: str) -> str:
    lines = []
    for t in tiers:
        tn = t["tier_name"]
        hf_id = f"{org}/{model_short}-sculpt-{tn}"
        wgb = t.get("weights_gb", "")
        ratio = t.get("ppl_ratio", "")
        use_case = TIER_USE_CASES.get(tn, "")
        marker = " 👈 **this model**" if tn == current_tier else ""
        lines.append(f"| {tn} | [{hf_id}](https://huggingface.co/{hf_id}){marker} | {wgb} GB | {ratio} | {use_case} |")
    return "\n".join(lines)


def _generate_card(
    tier: dict,
    all_tiers: List[dict],
    model_info: dict,
    bench: Dict[str, Dict[str, Any]],
    org: str,
) -> str:
    tier_name = tier["tier_name"]
    model_path = tier["model_path"]
    model_short = model_info["model_short"]
    base_model_id = model_info["base_model_id"]
    hf_model_id = f"{org}/{model_short}-sculpt-{tier_name}"

    wiki = bench.get(model_path, {}).get("wikitext", {})
    chat = bench.get(model_path, {}).get("chat", {})
    rag = bench.get(model_path, {}).get("rag", {})
    base_wiki = bench.get(base_model_id, {}).get("wikitext", {})
    base_chat = bench.get(base_model_id, {}).get("chat", {})
    base_rag = bench.get(base_model_id, {}).get("rag", {})

    ppl_wikitext = wiki.get("ppl_wikitext", "")
    ppl_ratio = wiki.get("ppl_ratio", chat.get("ppl_ratio", ""))
    weights_gb = wiki.get("weights_gb", chat.get("weights_gb", ""))
    num_params = wiki.get("num_params", chat.get("num_params", ""))
    chat_prefill = chat.get("prefill_tokens_per_sec", "")
    chat_decode = chat.get("decode_tokens_per_sec", "")
    rag_ttft = rag.get("ttft_ms_p95", "")
    chat_ttft = chat.get("ttft_ms_p95", "")

    base_weights = float(base_wiki.get("weights_gb", 0) or base_chat.get("weights_gb", 0) or 0)
    cur_weights = float(weights_gb) if weights_gb else 0
    size_cut = round((1 - cur_weights / base_weights) * 100) if base_weights > 0 else 0

    base_prefill = float(base_chat.get("prefill_tokens_per_sec", 0) or 0)
    cur_prefill = float(chat_prefill) if chat_prefill else 0
    prefill_change = _pct_change(cur_prefill, base_prefill) if base_prefill else ""

    base_ttft_rag = float(base_rag.get("ttft_ms_p95", 0) or 0)
    cur_ttft_rag = float(rag_ttft) if rag_ttft else 0
    ttft_change = _pct_change(cur_ttft_rag, base_ttft_rag) if base_ttft_rag else ""

    if float(ppl_ratio or 99) < 1.0:
        speed_claim = f"quality improved ({ppl_ratio}x PPL)"
    elif float(ppl_ratio or 99) < 1.1:
        speed_claim = f"quality preserved ({ppl_ratio}x PPL)"
    else:
        speed_claim = f"{prefill_change} faster prefill"

    tier_display = tier_name.capitalize()
    benchmark_table = _build_benchmark_table(bench, all_tiers, base_model_id)
    tier_links = _build_tier_links(all_tiers, org, model_short, tier_name)

    card = f"""---
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
language:
  - en
base_model: {base_model_id}
tags:
  - dystrio
  - sculpt
  - pruned
  - compressed
  - efficient
  - dense
  - runtime-agnostic
  - no-custom-kernels
  - hf-drop-in
  - drop-in-replacement
  - smaller
  - faster
  - {model_info["arch_tag"]}
datasets:
  - wikitext
model-index:
  - name: Dystrio Sculpt ({model_info["model_short"]} {tier_display})
    results:
      - task:
          type: text-generation
        dataset:
          name: WikiText-103 (validation)
          type: wikitext
        metrics:
          - name: perplexity
            type: perplexity
            value: {ppl_wikitext}
          - name: ppl_ratio
            type: ppl_ratio
            value: {ppl_ratio}
---

# {hf_model_id}

> **{size_cut}% smaller, {speed_claim}, drop-in replacement. No custom kernels. No runtime changes.**

Dystrio Sculpt structurally compresses transformer models, producing dense models that load with standard `transformers` — no custom code, no new ops, no deployment friction.

This is the **{tier_display}** tier of [{model_info["base_display"]}](https://huggingface.co/{base_model_id}).

## Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{hf_model_id}", torch_dtype="bfloat16", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("{hf_model_id}")

inputs = tokenizer("The future of AI inference is", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Benchmark Results

All tiers compiled from [{model_info["base_display"]}](https://huggingface.co/{base_model_id}) on A100 80GB, bf16:

{benchmark_table}

### Key Metrics (this model)

| Metric | Value |
|--------|-------|
| **Weights memory** | {weights_gb} GB ({size_cut}% smaller) |
| **PPL ratio** | {ppl_ratio} |
| **Chat prefill TPS** | {chat_prefill} ({prefill_change}) |
| **RAG TTFT p95** | {rag_ttft} ms ({ttft_change}) |
| **Decode TPS** | {chat_decode} (flat) |
| **Parameters** | {_fmt_params(num_params)} |

## All Sculpt Tiers

| Tier | HuggingFace | Size | PPL Ratio | Use Case |
|------|-------------|------|-----------|----------|
{tier_links}

## What is Dystrio Sculpt?

Dystrio Sculpt compiles transformer models into smaller, faster variants. Output models:

- Are **dense** (not sparse) — standard architecture, fewer parameters
- Load with **standard HuggingFace Transformers** — no custom code needed
- Require **no custom kernels** and **no runtime changes**
- Work as a one-step compile before deployment
- Stack with quantization (AWQ, GPTQ, GGUF) for compound savings

## Compatibility

- ✅ HuggingFace Transformers
- ✅ vLLM
- ✅ TGI (Text Generation Inference)
- ✅ llama.cpp / GGUF conversion
- ✅ AWQ / GPTQ quantization
- ✅ Any framework that loads standard safetensors

## Benchmark Environment

- **GPU**: NVIDIA A100-SXM4-80GB
- **dtype**: bf16
- **Torch**: 2.10.0+cu128
- **Transformers**: 5.3.0
- **Deterministic**: True
- Single-GPU, standard HuggingFace Transformers, no custom kernels.

## Metric Definitions

- **PPL ratio**: WikiText-103 perplexity relative to baseline. <1.0 = quality improved.
- **Prefill TPS**: Tokens per second during prompt encoding (higher = faster).
- **TTFT p95**: Time to first token at 95th percentile (lower = faster).
- **Decode TPS**: Tokens per second during generation (higher = faster).
- **Weights (GB)**: Model parameter memory (deterministic, runtime-independent).

## Citation

```bibtex
@misc{{dystrio_sculpt_2026,
  title={{Dystrio Sculpt: Structural Compilation for Transformer LLMs}},
  author={{Dystrio}},
  year={{2026}},
  url={{https://huggingface.co/dystrio}}
}}
```
"""
    return card


def _discover_tiers(compile_dir: Path, bench: Dict[str, Dict[str, Any]]) -> List[dict]:
    tiers = []
    for d in sorted(compile_dir.iterdir()):
        if not d.is_dir() or not d.name.startswith("frontier_"):
            continue
        model_dir = d / "model"
        if not model_dir.exists():
            continue
        parts = d.name.split("_", 2)
        tier_name = parts[2] if len(parts) > 2 else parts[1]
        model_path = str(model_dir)

        wiki = bench.get(model_path, {}).get("wikitext", {})
        chat = bench.get(model_path, {}).get("chat", {})
        tiers.append({
            "tier_name": tier_name,
            "model_path": model_path,
            "dir": d,
            "weights_gb": wiki.get("weights_gb", chat.get("weights_gb", "")),
            "ppl_ratio": wiki.get("ppl_ratio", chat.get("ppl_ratio", "")),
        })
    tiers.sort(key=lambda t: TIER_ORDER.index(t["tier_name"]) if t["tier_name"] in TIER_ORDER else 99)
    return tiers


def main():
    parser = argparse.ArgumentParser(description="Push model zoo to HuggingFace")
    parser.add_argument("--zoo-dir", type=str, default="/data/zoo")
    parser.add_argument("--org", type=str, default="dystrio")
    parser.add_argument("--dry-run", action="store_true", help="Generate cards but don't upload")
    args = parser.parse_args()

    zoo = Path(args.zoo_dir)

    for model_info in MODEL_ZOO:
        compile_dir = zoo / model_info["compile_dir"]
        bench_csv = zoo / model_info["bench_dir"] / "benchmarks.csv"

        if not compile_dir.exists():
            print(f"SKIP {model_info['model_short']}: {compile_dir} not found")
            continue
        if not bench_csv.exists():
            print(f"SKIP {model_info['model_short']}: {bench_csv} not found")
            continue

        bench = _load_bench(bench_csv)
        tiers = _discover_tiers(compile_dir, bench)

        if not tiers:
            print(f"SKIP {model_info['model_short']}: no tiers found in {compile_dir}")
            continue

        print(f"\n{'='*60}")
        print(f"  {model_info['base_display']}: {len(tiers)} tiers")
        print(f"{'='*60}")

        for tier in tiers:
            tier_name = tier["tier_name"]
            hf_repo = f"{args.org}/{model_info['model_short']}-sculpt-{tier_name}"
            model_dir = tier["dir"] / "model"

            card = _generate_card(tier, tiers, model_info, bench, args.org)

            card_path = model_dir / "README.md"
            card_path.write_text(card)
            print(f"  wrote {card_path}")

            if args.dry_run:
                print(f"  DRY RUN: would upload {model_dir} -> {hf_repo}")
                continue

            from huggingface_hub import HfApi
            api = HfApi()

            print(f"  uploading {model_dir} -> {hf_repo} ...")
            api.create_repo(hf_repo, exist_ok=True, repo_type="model")
            api.upload_folder(
                folder_path=str(model_dir),
                repo_id=hf_repo,
                repo_type="model",
                commit_message=f"Dystrio Sculpt {tier_name} tier of {model_info['base_display']}",
            )
            print(f"  ✓ {hf_repo} uploaded")

        print()


if __name__ == "__main__":
    main()
