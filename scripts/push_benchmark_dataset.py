"""Backfill all existing data into the Dystrio Efficiency Dataset.

Sources:
  1. data/downstream_benchmarks.jsonl  — first sculpt round (26 records)
  2. data/zoo_ab_consolidated.jsonl    — zoo A/B test (35 sculpt + 34 evals)
  3. data/eval_results.json            — older eval data (same as #1, different format)

Merges sculpt metrics + downstream evals into canonical records and pushes
to the master HF dataset.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dystrio_sculpt.efficiency_dataset import build_record, append_local, push_to_hub, LOCAL_CACHE

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _parse_variant(variant: str) -> tuple[str, str]:
    if variant.endswith("_distill"):
        return variant.rsplit("_distill", 1)[0].replace("_", "/", 1), "sculpt+distill"
    elif variant.endswith("_nodistill"):
        return variant.rsplit("_nodistill", 1)[0].replace("_", "/", 1), "sculpt"
    elif variant.endswith("_baseline"):
        return variant.rsplit("_baseline", 1)[0].replace("_", "/", 1), "baseline"
    return variant.replace("_", "/", 1), "unknown"


def backfill_first_round():
    """Ingest data/downstream_benchmarks.jsonl (first sculpt round)."""
    path = DATA_DIR / "downstream_benchmarks.jsonl"
    if not path.exists():
        print(f"  SKIP: {path} not found")
        return 0

    count = 0
    with open(path) as f:
        for line in f:
            row = json.loads(line.strip())
            record = build_record(
                record_type="sculpt_run",
                model_id=row.get("model_id", ""),
                model_family=row.get("model_family"),
                method="baseline" if row.get("tier") == "baseline" else "sculpt",
                tier=row.get("tier"),
                intermediate_size=row.get("intermediate_size"),
                hidden_size=row.get("hidden_size"),
                keep_frac=row.get("mlp_kept_pct", 100.0) / 100.0,
                arc_challenge_acc_norm=row.get("arc_challenge"),
                hellaswag_acc_norm=row.get("hellaswag"),
                mmlu_acc=row.get("mmlu"),
                truthfulqa_mc2_acc=row.get("truthfulqa_mc2"),
                eval_engine=row.get("eval_engine", "lm-eval"),
                eval_date=row.get("eval_date"),
                gpu_name=row.get("eval_gpu", "A100-80GB"),
                dtype="bf16",
                timestamp=f"{row.get('eval_date', '2026-03-15')}T00:00:00+00:00",
            )
            append_local(record)
            count += 1
    return count


def backfill_zoo_ab():
    """Ingest data/zoo_ab_consolidated.jsonl (A/B distill test)."""
    path = DATA_DIR / "zoo_ab_consolidated.jsonl"
    if not path.exists():
        print(f"  SKIP: {path} not found")
        return 0

    sculpt = {}
    evals = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rec_type = rec.pop("type", "")
            variant = rec.get("variant", "")
            if rec_type == "sculpt_summary":
                sculpt[(variant, rec.get("name", ""))] = rec
            elif rec_type == "downstream_eval":
                evals[(variant, rec.get("tier", ""))] = rec

    all_keys = set(sculpt.keys()) | set(evals.keys())
    count = 0
    for key in sorted(all_keys):
        variant, tier = key
        model_id, method = _parse_variant(variant)
        s = sculpt.get(key, {})
        e = evals.get(key, {})

        def _f(v):
            try:
                return float(v)
            except (ValueError, TypeError):
                return None

        record = build_record(
            record_type="ab_test_run",
            model_id=model_id,
            method=method,
            tier=tier,
            keep_frac=_f(s.get("keep_frac")),
            ppl_w103=_f(s.get("ppl_w103")),
            ppl_ratio=_f(s.get("ppl_ratio")),
            prefill_speedup=_f(s.get("prefill_speedup")),
            decode_speedup=_f(s.get("decode_speedup")),
            risk_score=_f(s.get("risk_score")),
            compile_time_s=_f(s.get("compile_time_s")),
            e2e_speedup_chat=_f(s.get("e2e_speedup_chat")),
            e2e_speedup_rag=_f(s.get("e2e_speedup_rag")),
            e2e_speedup_batch=_f(s.get("e2e_speedup_batch")),
            prefill_ms_p95=_f(s.get("prefill_ms_p95")),
            decode_ms_per_tok_p95=_f(s.get("decode_ms_per_tok_p95")),
            num_params_compressed=int(_f(s.get("num_params")) or 0) or None,
            weights_gb=_f(s.get("weights_gb")),
            peak_compile_alloc_gb=_f(s.get("peak_compile_alloc_gb")),
            peak_bench_alloc_gb=_f(s.get("peak_bench_alloc_gb")),
            steady_state_alloc_gb=_f(s.get("steady_state_alloc_gb")),
            baseline_prefill_ms_p95=_f(s.get("baseline_prefill_ms_p95")),
            baseline_decode_ms_per_tok_p95=_f(s.get("baseline_decode_ms_per_tok_p95")),
            weights_memory_reduction_pct=_f(s.get("weights_memory_reduction_pct")),
            distillation_enabled="distill" in method,
            distill_alpha=0.5 if "distill" in method else 0.0,
            arc_challenge_acc_norm=e.get("arc_norm"),
            hellaswag_acc_norm=e.get("hellaswag_norm"),
            mmlu_acc=e.get("mmlu_acc"),
            truthfulqa_mc2_acc=e.get("truthfulqa_mc2"),
            eval_engine="lm-eval",
            gpu_name="A100-80GB",
            dtype="bf16",
            timestamp="2026-03-18T00:00:00+00:00",
        )
        append_local(record)
        count += 1
    return count


def main():
    # Clear existing cache to avoid duplicates
    if LOCAL_CACHE.exists():
        LOCAL_CACHE.unlink()
        print(f"Cleared {LOCAL_CACHE}")

    print("Backfilling first sculpt round...")
    n1 = backfill_first_round()
    print(f"  {n1} records")

    print("Backfilling zoo A/B test...")
    n2 = backfill_zoo_ab()
    print(f"  {n2} records")

    total = n1 + n2
    print(f"\nTotal: {total} records in {LOCAL_CACHE}")

    print("\nPushing to HuggingFace...")
    try:
        url = push_to_hub(private=True)
        print(f"Done: {url}")
    except Exception as exc:
        print(f"Push failed: {exc}")
        print(f"Records saved locally at {LOCAL_CACHE}")


if __name__ == "__main__":
    main()
