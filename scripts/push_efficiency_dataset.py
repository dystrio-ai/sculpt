#!/usr/bin/env python3
"""Build and push full degradation curve (SAFE + UNSAFE) to efficiency dataset.

Combines:
  - summary.csv (4 SAFE frontier points with full metrics)
  - Log-extracted UNSAFE probe data (kf=0.750, 0.700, 0.660)
  - lm_eval results (for SAFE points + baseline)
  - run_metadata.json (environment info)
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dystrio_sculpt.efficiency_dataset import build_record, push_to_hub


BASELINE_PROBE = {
    "accuracy": 0.7084,
    "mmlu": 0.7400,
    "hellaswag": 0.7250,
    "arc": 0.6286,
}

UNSAFE_POINTS = [
    {
        "keep_frac": 0.750,
        "probe_accuracy": 0.5805,
        "probe_mmlu": 0.6200,
        "probe_hellaswag": 0.6250,
        "probe_arc": 0.4571,
        "status": "UNSAFE",
    },
    {
        "keep_frac": 0.700,
        "probe_accuracy": 0.5956,
        "probe_mmlu": 0.6100,
        "probe_hellaswag": 0.6625,
        "probe_arc": 0.5000,
        "status": "UNSAFE",
    },
    {
        "keep_frac": 0.660,
        "probe_accuracy": 0.5537,
        "probe_mmlu": 0.5600,
        "probe_hellaswag": 0.6375,
        "probe_arc": 0.4571,
        "status": "UNSAFE",
    },
]

SAFE_PROBES = {
    0.950: {"accuracy": 0.6914, "mmlu": 0.7400, "hellaswag": 0.7000, "arc": 0.5857},
    0.900: {"accuracy": 0.6719, "mmlu": 0.7000, "hellaswag": 0.6875, "arc": 0.6000},
    0.880: {"accuracy": 0.6591, "mmlu": 0.6700, "hellaswag": 0.7250, "arc": 0.5714},
    0.820: {"accuracy": 0.6304, "mmlu": 0.6600, "hellaswag": 0.6875, "arc": 0.5143},
}


def load_lm_eval(results_dir):
    scores = {}
    for f in Path(results_dir).rglob("results*.json"):
        data = json.load(open(f))
        for task, vals in data.get("results", {}).items():
            name = task.split(",")[0].strip()
            for key in ("acc_norm,none", "acc,none", "exact_match,strict-match", "mc2"):
                if key in vals:
                    scores[name] = vals[key]
                    break
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sculpt-dir", required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--repo-id", default="dystrio/efficiency-dataset")
    args = parser.parse_args()

    sculpt_dir = Path(args.sculpt_dir)
    meta = json.load(open(sculpt_dir / "run_metadata.json"))

    env = {
        "gpu_name": meta.get("gpu_name"),
        "torch_version": meta.get("torch_version"),
        "transformers_version": meta.get("transformers_version"),
        "dtype": meta.get("dtype", "bf16"),
        "git_sha": meta.get("git_commit"),
    }

    records = []

    # ── Baseline record ──
    baseline_lm = load_lm_eval(sculpt_dir / "baseline_lm_eval")
    records.append(build_record(
        model_id=args.base_model,
        record_type="baseline",
        tier="baseline",
        keep_frac=1.0,
        arc_challenge_acc_norm=baseline_lm.get("arc_challenge"),
        hellaswag_acc_norm=baseline_lm.get("hellaswag"),
        mmlu_acc=baseline_lm.get("mmlu"),
        truthfulqa_mc2_acc=baseline_lm.get("truthfulqa_mc2"),
        winogrande_acc=baseline_lm.get("winogrande"),
        gsm8k_acc=baseline_lm.get("gsm8k"),
        downstream_probe_accuracy=BASELINE_PROBE["accuracy"],
        downstream_probe_mmlu=BASELINE_PROBE["mmlu"],
        downstream_probe_hellaswag=BASELINE_PROBE["hellaswag"],
        downstream_probe_arc=BASELINE_PROBE["arc"],
        eval_engine="lm-eval",
        eval_date="2026-03-22",
        notes="Qwen3.5-9B baseline, full lm_eval suite",
        **env,
    ))
    print(f"  baseline: lm_eval loaded ({len(baseline_lm)} tasks)")

    # ── SAFE frontier points (from summary.csv + lm_eval) ──
    tier_map = {
        "frontier_0_default": "default",
        "frontier_1_production": "production",
        "frontier_2_throughput": "throughput",
        "frontier_3_experimental": "experimental",
    }

    with open(sculpt_dir / "summary.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tier_dir = row["name"]
            kf = float(row["keep_frac"])
            probe = SAFE_PROBES.get(kf, {})

            lm_dir = sculpt_dir / tier_dir / "lm_eval_results"
            lm_scores = load_lm_eval(lm_dir) if lm_dir.exists() else {}

            records.append(build_record(
                model_id=args.base_model,
                tier=tier_map.get(tier_dir, tier_dir),
                keep_frac=kf,
                ppl_w103=float(row.get("ppl_w103", 0)) or None,
                ppl_ratio=float(row.get("ppl_ratio", 0)) or None,
                prefill_speedup=float(row.get("prefill_speedup", 0)) or None,
                decode_speedup=float(row.get("decode_speedup", 0)) or None,
                compile_time_s=float(row.get("compile_time_s", 0)) or None,
                weights_gb=float(row.get("weights_gb", 0)) or None,
                num_params_compressed=int(row.get("num_params", 0)) or None,
                prefill_ms_p95=float(row.get("prefill_ms_p95", 0)) or None,
                decode_ms_per_tok_p95=float(row.get("decode_ms_per_tok_p95", 0)) or None,
                baseline_prefill_ms_p95=float(row.get("baseline_prefill_ms_p95", 0)) or None,
                baseline_decode_ms_per_tok_p95=float(row.get("baseline_decode_ms_per_tok_p95", 0)) or None,
                risk_score=float(row.get("risk_score", 0)) or None,
                weights_memory_reduction_pct=float(row.get("weights_memory_reduction_pct", 0)) or None,
                e2e_speedup_chat=float(row.get("e2e_speedup_chat", 0)) or None,
                e2e_speedup_rag=float(row.get("e2e_speedup_rag", 0)) or None,
                e2e_speedup_batch=float(row.get("e2e_speedup_batch", 0)) or None,
                peak_compile_alloc_gb=float(row.get("peak_compile_alloc_gb", 0)) or None,
                peak_bench_alloc_gb=float(row.get("peak_bench_alloc_gb", 0)) or None,
                steady_state_alloc_gb=float(row.get("steady_state_alloc_gb", 0)) or None,
                downstream_probe_accuracy=probe.get("accuracy"),
                downstream_probe_mmlu=probe.get("mmlu"),
                downstream_probe_hellaswag=probe.get("hellaswag"),
                downstream_probe_arc=probe.get("arc"),
                baseline_downstream_probe_accuracy=BASELINE_PROBE["accuracy"],
                arc_challenge_acc_norm=lm_scores.get("arc_challenge"),
                hellaswag_acc_norm=lm_scores.get("hellaswag"),
                mmlu_acc=lm_scores.get("mmlu"),
                truthfulqa_mc2_acc=lm_scores.get("truthfulqa_mc2"),
                winogrande_acc=lm_scores.get("winogrande"),
                gsm8k_acc=lm_scores.get("gsm8k"),
                eval_engine="lm-eval",
                eval_date="2026-03-22",
                distillation_enabled=True,
                distill_alpha=0.5,
                notes=f"SAFE — kf={kf}, live teacher distillation on H200",
                **env,
            ))
            print(f"  {tier_dir} (kf={kf}): SAFE, lm_eval={len(lm_scores)} tasks")

    # ── UNSAFE points (probe data only, no model saved) ──
    for pt in UNSAFE_POINTS:
        kf = pt["keep_frac"]
        retention = pt["probe_accuracy"] / BASELINE_PROBE["accuracy"] * 100
        records.append(build_record(
            model_id=args.base_model,
            tier=f"unsafe_kf{kf:.3f}",
            keep_frac=kf,
            downstream_probe_accuracy=pt["probe_accuracy"],
            downstream_probe_mmlu=pt["probe_mmlu"],
            downstream_probe_hellaswag=pt["probe_hellaswag"],
            downstream_probe_arc=pt["probe_arc"],
            baseline_downstream_probe_accuracy=BASELINE_PROBE["accuracy"],
            distillation_enabled=True,
            distill_alpha=0.5,
            notes=f"UNSAFE — kf={kf}, {retention:.1f}% retention, probe only (model not saved)",
            **env,
        ))
        print(f"  kf={kf}: UNSAFE ({retention:.1f}% retention), probe data only")

    print(f"\nTotal records: {len(records)} (1 baseline + 4 SAFE + 3 UNSAFE)")

    # ── Push to HF ──
    print(f"\nPushing to {args.repo_id}...")
    url = push_to_hub(records, repo_id=args.repo_id)
    print(f"Done: {url}")


if __name__ == "__main__":
    main()
