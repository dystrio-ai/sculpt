#!/usr/bin/env python3
"""Sculpt + Quantize comparison: proves structural pruning stacks with AWQ.

Quantizes both the baseline and sculpted-conservative models to AWQ 4-bit,
then evaluates PPL on WikiText-103 for all four variants:
  1. Baseline bf16
  2. Baseline AWQ-4bit
  3. Sculpt-conservative bf16
  4. Sculpt-conservative AWQ-4bit

Usage:
  python3 scripts/quantize_compare.py \
    --baseline mistralai/Mistral-7B-Instruct-v0.3 \
    --sculpted /data/zoo/mistral-7b-instruct/frontier_0_conservative/model \
    --outdir /data/zoo/quant_compare_mistral
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


def eval_ppl_wikitext103(model, tokenizer, device="cuda", max_tokens=40_000):
    """Evaluate perplexity on WikiText-103 validation set."""
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
    texts = [t for t in ds["text"] if len(t.strip()) > 50][:300]

    total_loss = 0.0
    total_tokens = 0
    max_len = 256

    model.eval()
    with torch.no_grad():
        for text in texts:
            if total_tokens >= max_tokens:
                break
            enc = tokenizer(text, return_tensors="pt", truncation=True,
                            max_length=max_len).to(device)
            if enc.input_ids.shape[1] < 2:
                continue
            out = model(**enc, labels=enc.input_ids)
            n = enc.input_ids.shape[1] - 1
            total_loss += out.loss.item() * n
            total_tokens += n

    import math
    return math.exp(total_loss / total_tokens) if total_tokens > 0 else float("inf")


def get_model_size_gb(model_path):
    """Get total size of model files in GB."""
    p = Path(model_path)
    total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return round(total / (1024 ** 3), 3)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def quantize_awq(model_id, quant_output_dir, device="cuda"):
    """Quantize a model to AWQ 4-bit."""
    from awq import AutoAWQForCausalLM

    print(f"  Loading {model_id} for AWQ quantization...")
    model = AutoAWQForCausalLM.from_pretrained(
        model_id, device_map=device, torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    quant_config = {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM",
    }

    print(f"  Quantizing (this takes ~10-15 min)...")
    t0 = time.time()
    model.quantize(tokenizer, quant_config=quant_config)
    elapsed = time.time() - t0
    print(f"  Quantization done in {elapsed:.0f}s")

    print(f"  Saving to {quant_output_dir}")
    model.save_quantized(quant_output_dir)
    tokenizer.save_pretrained(quant_output_dir)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return elapsed


def load_and_eval(model_id, label, device="cuda", is_awq=False):
    """Load a model variant and evaluate PPL."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {label}")
    print(f"  Model: {model_id}")
    print(f"{'='*60}")

    if is_awq:
        from awq import AutoAWQForCausalLM
        model = AutoAWQForCausalLM.from_pretrained(
            model_id, device_map=device, torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).model
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, trust_remote_code=True,
            ignore_mismatched_sizes=True,
        ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    n_params = count_params(model)
    size_gb = get_model_size_gb(model_id)

    print(f"  Params: {n_params:,}")
    print(f"  Size on disk: {size_gb} GB")
    print(f"  Evaluating PPL on WikiText-103...")

    t0 = time.time()
    ppl = eval_ppl_wikitext103(model, tokenizer, device)
    elapsed = time.time() - t0

    print(f"  PPL: {ppl:.4f} ({elapsed:.0f}s)")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "label": label,
        "model_id": str(model_id),
        "params": n_params,
        "size_gb": size_gb,
        "ppl_w103": round(ppl, 4),
        "eval_time_s": round(elapsed, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, help="HF model ID for baseline")
    parser.add_argument("--sculpted", required=True, help="Path to sculpted conservative model")
    parser.add_argument("--outdir", required=True, help="Output directory for quantized models and results")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    baseline_awq_dir = outdir / "baseline_awq"
    sculpted_awq_dir = outdir / "sculpted_awq"

    results = []

    # Step 1: Quantize baseline
    if not baseline_awq_dir.exists():
        print("\n" + "=" * 60)
        print("STEP 1: Quantize baseline to AWQ 4-bit")
        print("=" * 60)
        quantize_awq(args.baseline, str(baseline_awq_dir), args.device)
    else:
        print(f"Baseline AWQ already exists at {baseline_awq_dir}, skipping.")

    # Step 2: Quantize sculpted
    if not sculpted_awq_dir.exists():
        print("\n" + "=" * 60)
        print("STEP 2: Quantize sculpted conservative to AWQ 4-bit")
        print("=" * 60)
        quantize_awq(args.sculpted, str(sculpted_awq_dir), args.device)
    else:
        print(f"Sculpted AWQ already exists at {sculpted_awq_dir}, skipping.")

    # Step 3: Evaluate all four variants
    print("\n" + "=" * 60)
    print("STEP 3: Evaluate all variants")
    print("=" * 60)

    results.append(load_and_eval(args.baseline, "Baseline bf16", args.device))
    results.append(load_and_eval(str(baseline_awq_dir), "Baseline AWQ-4bit", args.device, is_awq=True))
    results.append(load_and_eval(args.sculpted, "Sculpt-conservative bf16", args.device))
    results.append(load_and_eval(str(sculpted_awq_dir), "Sculpt-conservative AWQ-4bit", args.device, is_awq=True))

    # Step 4: Print comparison table
    print("\n")
    print("=" * 80)
    print("RESULTS: Sculpt + Quantize Stacking Comparison")
    print("=" * 80)
    print(f"{'Variant':<30} {'Params':>12} {'Size (GB)':>10} {'PPL':>10} {'PPL ratio':>10}")
    print("-" * 80)

    baseline_ppl = results[0]["ppl_w103"]
    for r in results:
        ratio = r["ppl_w103"] / baseline_ppl
        print(f"{r['label']:<30} {r['params']:>12,} {r['size_gb']:>10} {r['ppl_w103']:>10.2f} {ratio:>10.4f}")

    print("-" * 80)
    print()

    baseline_awq_ppl = results[1]["ppl_w103"]
    sculpt_awq_ppl = results[3]["ppl_w103"]
    baseline_awq_size = results[1]["size_gb"]
    sculpt_awq_size = results[3]["size_gb"]

    ppl_advantage = (baseline_awq_ppl - sculpt_awq_ppl) / baseline_awq_ppl * 100
    size_advantage = (baseline_awq_size - sculpt_awq_size) / baseline_awq_size * 100

    print(f"Sculpt+AWQ vs AWQ-only:")
    print(f"  PPL advantage:  {ppl_advantage:+.1f}% ({'better' if ppl_advantage > 0 else 'worse'})")
    print(f"  Size advantage: {size_advantage:+.1f}% ({'smaller' if size_advantage > 0 else 'larger'})")
    print()

    results_path = outdir / "quant_comparison.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Full results saved to {results_path}")


if __name__ == "__main__":
    main()
