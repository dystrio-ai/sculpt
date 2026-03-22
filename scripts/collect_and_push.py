#!/usr/bin/env python3
"""Collect lm_eval results from frontier models, build model cards, push to HF."""

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo


TIER_META = {
    "frontier_0_default": {
        "suffix": "Sculpt-Default",
        "tier_display": "Default",
        "keep_frac": 0.950,
        "use_case": "Enterprise — maximum quality preservation",
    },
    "frontier_1_production": {
        "suffix": "Sculpt-Production",
        "tier_display": "Production",
        "keep_frac": 0.900,
        "use_case": "Enterprise — balanced quality and efficiency",
    },
    "frontier_2_throughput": {
        "suffix": "Sculpt-Throughput",
        "tier_display": "Throughput",
        "keep_frac": 0.880,
        "use_case": "Local/throughput — speed sweet spot (1.25x prefill)",
    },
    "frontier_3_experimental": {
        "suffix": "Sculpt-Experimental",
        "tier_display": "Experimental",
        "keep_frac": 0.820,
        "use_case": "Local — maximum compression (1.27x prefill)",
    },
}


def load_lm_eval_results(results_dir: Path) -> dict:
    """Parse lm_eval output directory and extract task scores."""
    scores = {}
    results_dir = Path(results_dir)

    # lm_eval saves results as JSON files in subdirectories
    for json_file in results_dir.rglob("results*.json"):
        with open(json_file) as f:
            data = json.load(f)

        if "results" not in data:
            continue

        for task_name, task_data in data["results"].items():
            # Normalize task names
            clean_name = task_name.split(",")[0].strip()

            if "acc_norm,none" in task_data:
                scores[clean_name] = round(task_data["acc_norm,none"] * 100, 1)
            elif "acc,none" in task_data:
                scores[clean_name] = round(task_data["acc,none"] * 100, 1)
            elif "exact_match,strict-match" in task_data:
                scores[clean_name] = round(task_data["exact_match,strict-match"] * 100, 1)
            elif "mc2" in task_data:
                scores[clean_name] = round(task_data["mc2"] * 100, 1)
            elif "acc_norm" in task_data:
                scores[clean_name] = round(task_data["acc_norm"] * 100, 1)
            elif "acc" in task_data:
                scores[clean_name] = round(task_data["acc"] * 100, 1)

    return scores


def build_benchmark_table(baseline: dict, tiers: dict) -> str:
    """Build a markdown benchmark table comparing all tiers."""
    tasks = ["mmlu", "hellaswag", "arc_challenge", "truthfulqa_mc2", "winogrande", "gsm8k"]
    display = {
        "mmlu": "MMLU",
        "hellaswag": "HellaSwag",
        "arc_challenge": "ARC-C",
        "truthfulqa_mc2": "TruthfulQA",
        "winogrande": "Winogrande",
        "gsm8k": "GSM8K",
    }

    header = "| Model | " + " | ".join(display.get(t, t) for t in tasks) + " |"
    sep = "|---|" + "|".join("---:" for _ in tasks) + "|"
    rows = [header, sep]

    # Baseline row
    vals = [str(baseline.get(t, "—")) for t in tasks]
    rows.append(f"| **Qwen3.5-9B (baseline)** | " + " | ".join(vals) + " |")

    # Tier rows
    for tier_name, tier_scores in tiers.items():
        meta = TIER_META[tier_name]
        kf = meta["keep_frac"]
        vals = []
        for t in tasks:
            score = tier_scores.get(t, "—")
            base_score = baseline.get(t)
            if isinstance(score, (int, float)) and isinstance(base_score, (int, float)):
                diff = score - base_score
                arrow = "↑" if diff > 0 else "↓" if diff < 0 else "="
                vals.append(f"{score} ({arrow}{abs(diff):.1f})")
            else:
                vals.append(str(score))
        rows.append(f"| **Sculpt {meta['tier_display']}** (kf={kf}) | " + " | ".join(vals) + " |")

    return "\n".join(rows)


def build_model_card(
    tier_name: str,
    base_model: str,
    org: str,
    baseline_scores: dict,
    tier_scores: dict,
    all_tiers_scores: dict,
    benchmark_table: str,
) -> str:
    """Build a complete model card for one tier."""
    meta = TIER_META[tier_name]
    model_short = base_model.split("/")[-1]
    hf_model_id = f"{org}/{model_short}-{meta['suffix']}"
    size_cut = round((1 - meta["keep_frac"]) * 100)

    tier_links = []
    for tn, tm in TIER_META.items():
        tn_id = f"{org}/{model_short}-{tm['suffix']}"
        tier_links.append(
            f"| {tm['tier_display']} | [{tn_id}](https://huggingface.co/{tn_id}) | "
            f"kf={tm['keep_frac']} | {tm['use_case']} |"
        )
    tier_links_table = "\n".join(tier_links)

    scores_section = ""
    for task, score in sorted(tier_scores.items()):
        base = baseline_scores.get(task)
        if isinstance(base, (int, float)):
            diff = score - base
            scores_section += f"| {task} | {score} | {base} | {diff:+.1f} |\n"
        else:
            scores_section += f"| {task} | {score} | — | — |\n"

    card = f"""---
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
language:
  - en
base_model: {base_model}
tags:
  - dystrio
  - sculpt
  - pruned
  - compressed
  - efficient
  - dense
  - drop-in-replacement
  - qwen3.5
datasets:
  - wikitext
  - cais/mmlu
  - teknium/OpenHermes-2.5
---

# {model_short}-{meta['suffix']}

> **{size_cut}% FFN compression with live teacher distillation. Drop-in replacement — no custom kernels, no runtime changes.**

Dystrio Sculpt structurally compresses transformer FFN layers, producing dense models that load with standard `transformers`.

This is the **{meta['tier_display']}** tier of [{model_short}](https://huggingface.co/{base_model}).

**Use case:** {meta['use_case']}

## Benchmark Results (lm_eval)

{benchmark_table}

### This Model vs Baseline

| Benchmark | {meta['tier_display']} | Baseline | Delta |
|---|---:|---:|---:|
{scores_section}

## Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "{hf_model_id}",
    torch_dtype="bfloat16",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("{hf_model_id}")

inputs = tokenizer("The future of AI inference is", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## All Sculpt Tiers

| Tier | HuggingFace | Config | Use Case |
|------|-------------|--------|----------|
{tier_links_table}

## Technical Details

- **Method:** Structural FFN pruning with Physarum-inspired block selection + live teacher distillation (alpha=0.5)
- **Keep fraction:** {meta['keep_frac']} ({size_cut}% of FFN neurons removed)
- **Repair:** 8-stage cosine-LR fine-tuning with best-checkpoint restore
- **Training data:** general_v2 mixture (WikiText, OpenHermes 2.5, MMLU, HellaSwag, GSM8K, OpenOrca)
- **Hardware:** 1x NVIDIA H200 141GB
- **Output:** Standard dense transformer — loads with any HuggingFace-compatible framework

## Compatibility

- HuggingFace Transformers
- vLLM
- TGI (Text Generation Inference)
- llama.cpp / GGUF conversion
- AWQ / GPTQ quantization
- Any framework that loads standard safetensors

## Citation

```bibtex
@misc{{dystrio_sculpt_2026,
  title={{Dystrio Sculpt: Structural Compilation for Transformer LLMs}},
  author={{Dystrio}},
  year={{2026}},
  url={{https://huggingface.co/{org}}}
}}
```
"""
    return card


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sculpt-dir", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--org", default="dystrio")
    args = parser.parse_args()

    sculpt_dir = Path(args.sculpt_dir)
    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)

    # Load baseline results
    print("Loading baseline lm_eval results...")
    baseline_scores = load_lm_eval_results(sculpt_dir / "baseline_lm_eval")
    print(f"  Baseline: {baseline_scores}")

    # Load tier results
    all_tiers_scores = {}
    for tier_name in TIER_META:
        tier_dir = sculpt_dir / tier_name / "lm_eval_results"
        if tier_dir.exists():
            scores = load_lm_eval_results(tier_dir)
            all_tiers_scores[tier_name] = scores
            print(f"  {tier_name}: {scores}")
        else:
            print(f"  WARNING: {tier_dir} not found, skipping")

    # Build benchmark table
    benchmark_table = build_benchmark_table(baseline_scores, all_tiers_scores)
    print("\n" + benchmark_table + "\n")

    # Build model cards and push each tier
    model_short = args.base_model.split("/")[-1]

    for tier_name, tier_scores in all_tiers_scores.items():
        meta = TIER_META[tier_name]
        repo_id = f"{args.org}/{model_short}-{meta['suffix']}"
        model_dir = sculpt_dir / tier_name / "model"

        print(f"\n>> Pushing {tier_name} -> {repo_id}")

        # Build model card
        card = build_model_card(
            tier_name=tier_name,
            base_model=args.base_model,
            org=args.org,
            baseline_scores=baseline_scores,
            tier_scores=tier_scores,
            all_tiers_scores=all_tiers_scores,
            benchmark_table=benchmark_table,
        )

        # Write model card to model dir
        card_path = model_dir / "README.md"
        card_path.write_text(card)
        print(f"   Model card written to {card_path}")

        # Create repo and upload
        try:
            create_repo(repo_id, token=token, private=True, exist_ok=True)
        except Exception as e:
            print(f"   Repo creation note: {e}")

        api.upload_folder(
            folder_path=str(model_dir),
            repo_id=repo_id,
            commit_message=f"Dystrio Sculpt {meta['tier_display']} tier (kf={meta['keep_frac']})",
        )
        print(f"   Pushed: https://huggingface.co/{repo_id}")

    # Save summary
    summary = {
        "base_model": args.base_model,
        "baseline": baseline_scores,
        "tiers": {k: {"scores": v, "meta": TIER_META[k]} for k, v in all_tiers_scores.items()},
        "benchmark_table": benchmark_table,
    }
    summary_path = sculpt_dir / "lm_eval_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
