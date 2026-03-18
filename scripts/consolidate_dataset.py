#!/usr/bin/env python3
"""Consolidate ALL data sources into a single efficiency_dataset.jsonl.

This is the one-time migration that merges:
  1. First sculpt round (data/downstream_benchmarks.jsonl)
  2. Zoo A/B test sculpt summaries + downstream evals (data/zoo_ab_consolidated.jsonl)
  3. Corrected baselines from lm-eval on A100

Into a single unified record per (model, variant, tier) with the full
build_record schema: fingerprint + optimization + environment + outcomes + trace.

After this runs, efficiency_dataset.jsonl is the single source of truth.
"""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dystrio_sculpt.efficiency_dataset import build_record, LOCAL_CACHE

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Corrected baselines from lm-eval (acc_norm for HellaSwag)
CORRECTED_BASELINES = {
    "google/gemma-2-2b-it": {
        "arc_challenge_acc_norm": 0.5290, "hellaswag_acc_norm": 0.7262,
        "mmlu_acc": 0.5691, "truthfulqa_mc2_acc": 0.5322,
    },
    "mistralai/Mistral-7B-Instruct-v0.3": {
        "arc_challenge_acc_norm": 0.6015, "hellaswag_acc_norm": 0.8328,
        "mmlu_acc": 0.5975, "truthfulqa_mc2_acc": 0.5939,
    },
    "meta-llama/Llama-3.2-3B-Instruct": {
        "arc_challenge_acc_norm": 0.4633, "hellaswag_acc_norm": 0.7163,
        "mmlu_acc": 0.6223, "truthfulqa_mc2_acc": 0.5138,
    },
    "meta-llama/Llama-3.1-8B-Instruct": {
        "arc_challenge_acc_norm": 0.5555, "hellaswag_acc_norm": 0.7955,
        "mmlu_acc": 0.6844, "truthfulqa_mc2_acc": 0.5456,
    },
    "Qwen/Qwen2.5-3B-Instruct": {
        "arc_challenge_acc_norm": 0.4821, "hellaswag_acc_norm": 0.7495,
        "mmlu_acc": 0.6545, "truthfulqa_mc2_acc": 0.5874,
    },
}


def parse_variant(variant_str: str):
    """Parse 'Qwen_Qwen2.5-3B-Instruct_distill' -> (model_id, distill_bool)."""
    parts = variant_str.rsplit("_", 1)
    if len(parts) == 2 and parts[1] in ("distill", "nodistill"):
        model_id = parts[0].replace("_", "/", 1)
        return model_id, parts[1] == "distill"
    return variant_str.replace("_", "/", 1), False


def safe_float(v, default=None):
    try:
        return float(v) if v not in (None, "", "None") else default
    except (ValueError, TypeError):
        return default


def ingest_baselines():
    """Create baseline records from corrected lm-eval results."""
    records = []
    for model_id, scores in CORRECTED_BASELINES.items():
        records.append(build_record(
            model_id=model_id,
            model_family=model_id.split("/")[-1],
            method="baseline",
            tier="baseline",
            keep_frac=1.0,
            record_type="baseline",
            eval_engine="lm-eval",
            eval_date="2026-03-18",
            gpu_name="A100-80GB",
            dtype="bf16",
            **scores,
        ))
    return records


def ingest_zoo_ab():
    """Merge zoo A/B sculpt summaries + downstream evals into unified records."""
    zoo_path = DATA_DIR / "zoo_ab_consolidated.jsonl"
    if not zoo_path.exists():
        return []

    raw = []
    with open(zoo_path) as f:
        for line in f:
            raw.append(json.loads(line))

    sculpts = {(r["variant"], r["name"]): r for r in raw if r.get("type") == "sculpt_summary"}
    evals = {(r["variant"], r["tier"]): r for r in raw if r.get("type") == "downstream_eval"}

    records = []
    for key, s in sculpts.items():
        variant_str, tier = key
        model_id, distill = parse_variant(variant_str)
        e = evals.get(key, {})

        records.append(build_record(
            model_id=model_id,
            model_family=model_id.split("/")[-1],
            method="sculpt+distill" if distill else "sculpt",
            tier=tier,
            record_type="zoo_ab",

            # Optimization
            keep_frac=safe_float(s.get("keep_frac")),
            distillation_enabled=distill,
            distill_alpha=0.5 if distill else 0.0,
            calib_dataset="wikitext",
            calib_config="wikitext-2-raw-v1",
            risk_score=safe_float(s.get("risk_score")),
            compile_time_s=safe_float(s.get("compile_time_s")),

            # Environment
            gpu_name="A100-80GB",
            dtype="bf16",

            # Outcomes — sculpt metrics
            ppl_w103=safe_float(s.get("ppl_w103")),
            ppl_ratio=safe_float(s.get("ppl_ratio")),
            prefill_speedup=safe_float(s.get("prefill_speedup")),
            decode_speedup=safe_float(s.get("decode_speedup")),
            e2e_speedup_chat=safe_float(s.get("e2e_speedup_chat")),
            e2e_speedup_rag=safe_float(s.get("e2e_speedup_rag")),
            e2e_speedup_batch=safe_float(s.get("e2e_speedup_batch")),
            prefill_ms_p95=safe_float(s.get("prefill_ms_p95")),
            decode_ms_per_tok_p95=safe_float(s.get("decode_ms_per_tok_p95")),
            baseline_prefill_ms_p95=safe_float(s.get("baseline_prefill_ms_p95")),
            baseline_decode_ms_per_tok_p95=safe_float(s.get("baseline_decode_ms_per_tok_p95")),
            weights_gb=safe_float(s.get("weights_gb")),
            weights_memory_reduction_pct=safe_float(s.get("weights_memory_reduction_pct")),
            peak_compile_alloc_gb=safe_float(s.get("peak_compile_alloc_gb")),
            peak_bench_alloc_gb=safe_float(s.get("peak_bench_alloc_gb")),
            steady_state_alloc_gb=safe_float(s.get("steady_state_alloc_gb")),
            num_params_compressed=int(safe_float(s.get("num_params"), 0)),

            # Outcomes — downstream benchmarks (from lm-eval)
            arc_challenge_acc_norm=safe_float(e.get("arc_norm") or e.get("arc_challenge_acc_norm")),
            hellaswag_acc_norm=safe_float(e.get("hellaswag_norm") or e.get("hellaswag_acc_norm")),
            mmlu_acc=safe_float(e.get("mmlu_acc")),
            truthfulqa_mc2_acc=safe_float(e.get("truthfulqa_mc2") or e.get("truthfulqa_mc2_acc")),
            eval_engine="lm-eval" if e else None,
            eval_date="2026-03-18" if e else None,
        ))
    return records


def ingest_first_round():
    """Ingest first sculpt round (pre-zoo) from downstream_benchmarks.jsonl."""
    bench_path = DATA_DIR / "downstream_benchmarks.jsonl"
    if not bench_path.exists():
        return []

    records = []
    with open(bench_path) as f:
        for line in f:
            r = json.loads(line)

            # Fix HellaSwag: first round stored acc, not acc_norm
            hs_val = safe_float(r.get("hellaswag_acc_norm"))
            hs_note = None
            if hs_val and hs_val < 0.6:
                hs_note = "raw_acc_not_acc_norm"

            records.append(build_record(
                model_id=r.get("model_id", "?"),
                model_family=r.get("model_family"),
                method=r.get("method", "sculpt"),
                tier=r.get("tier", "unknown"),
                record_type="first_round",
                keep_frac=safe_float(r.get("keep_frac")),
                hidden_size=r.get("hidden_size"),
                intermediate_size=r.get("intermediate_size"),
                gpu_name="A100-80GB",
                dtype="bf16",
                arc_challenge_acc_norm=safe_float(r.get("arc_challenge_acc_norm")),
                hellaswag_acc_norm=hs_val,
                mmlu_acc=safe_float(r.get("mmlu_acc")),
                truthfulqa_mc2_acc=safe_float(r.get("truthfulqa_mc2_acc")),
                eval_engine="lm-eval",
                eval_date="2026-03-15",
                failure_reason=hs_note,
            ))
    return records


def main():
    all_records = []

    print("Ingesting corrected baselines...")
    baselines = ingest_baselines()
    all_records.extend(baselines)
    print(f"  {len(baselines)} baseline records")

    print("Ingesting zoo A/B (sculpt + downstream merged)...")
    zoo = ingest_zoo_ab()
    all_records.extend(zoo)
    print(f"  {len(zoo)} zoo A/B records")

    print("Ingesting first sculpt round...")
    first = ingest_first_round()
    all_records.extend(first)
    print(f"  {len(first)} first-round records")

    print(f"\nTotal: {len(all_records)} records")

    # Write
    output = DATA_DIR / "efficiency_dataset.jsonl"
    with open(output, "w") as f:
        for r in all_records:
            f.write(json.dumps(r) + "\n")
    print(f"Written to {output}")

    # Summary
    from collections import Counter
    types = Counter(r["record_type"] for r in all_records)
    print("\nBreakdown:")
    for t, c in types.most_common():
        print(f"  {t}: {c}")

    has_both = sum(
        1 for r in all_records
        if r.get("ppl_ratio") and r.get("arc_challenge_acc_norm")
    )
    print(f"\nRecords with BOTH sculpt metrics + downstream benchmarks: {has_both}")


if __name__ == "__main__":
    main()
