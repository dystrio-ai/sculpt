#!/usr/bin/env python3
"""One-shot: pull perf data from efficiency dataset, update model cards on HF.

Usage:
    export HF_TOKEN="hf_..."
    python3 scripts/update_model_cards_perf.py
"""

import json
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
# hf_hub_download used in update_card to fetch README

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

TOKEN = os.environ.get("HF_TOKEN")
ORG = "dystrio"
BASE_MODEL = "Qwen/Qwen3.5-9B"
MODEL_SHORT = "Qwen3.5-9B"
DATASET_REPO = "dystrio/efficiency-dataset"

TIERS = {
    "default": {"suffix": "Sculpt-Default", "keep_frac": 0.95},
    "production": {"suffix": "Sculpt-Production", "keep_frac": 0.90},
    "throughput": {"suffix": "Sculpt-Throughput", "keep_frac": 0.88},
    "experimental": {"suffix": "Sculpt-Experimental", "keep_frac": 0.82},
}


def load_efficiency_data():
    """Pull efficiency dataset and extract per-tier perf metrics."""
    from datasets import load_dataset

    ds = load_dataset(DATASET_REPO, split="optimization_runs", token=TOKEN)
    records = [dict(r) for r in ds]

    tier_perf = {}
    baseline_perf = {}

    for rec in records:
        if rec.get("model_id") != BASE_MODEL:
            continue
        tier = rec.get("tier", "")
        kf = rec.get("keep_frac")

        if tier == "baseline" or kf == 1.0:
            baseline_perf = {
                "prefill_tps": rec.get("prefill_tps"),
                "decode_tps": rec.get("decode_tps"),
                "weights_gb": rec.get("weights_gb"),
                "steady_state_alloc_gb": rec.get("steady_state_alloc_gb"),
                "prefill_ms_p95": rec.get("prefill_ms_p95"),
                "decode_ms_per_tok_p95": rec.get("decode_ms_per_tok_p95"),
            }
            continue

        for tier_name, tier_info in TIERS.items():
            tier_aliases = [tier_name, f"frontier_{list(TIERS.keys()).index(tier_name)}_{tier_name}"]
            if tier in tier_aliases or (kf and abs(kf - tier_info["keep_frac"]) < 0.005):
                candidate = {
                    "prefill_tps": rec.get("prefill_tps"),
                    "decode_tps": rec.get("decode_tps"),
                    "prefill_speedup": rec.get("prefill_speedup"),
                    "decode_speedup": rec.get("decode_speedup"),
                    "weights_gb": rec.get("weights_gb"),
                    "baseline_weights_gb": rec.get("baseline_weights_gb") or baseline_perf.get("weights_gb"),
                    "weights_memory_reduction_pct": rec.get("weights_memory_reduction_pct"),
                    "steady_state_alloc_gb": rec.get("steady_state_alloc_gb"),
                    "baseline_steady_state_alloc_gb": rec.get("baseline_steady_state_alloc_gb") or baseline_perf.get("steady_state_alloc_gb"),
                    "steady_state_memory_reduction_pct": rec.get("steady_state_memory_reduction_pct"),
                    "num_params": rec.get("num_params_compressed"),
                    "baseline_prefill_tps": rec.get("baseline_prefill_tps") or baseline_perf.get("prefill_tps"),
                    "baseline_decode_tps": rec.get("baseline_decode_tps") or baseline_perf.get("decode_tps"),
                }
                if tier_name not in tier_perf:
                    tier_perf[tier_name] = candidate
                else:
                    for k, v in candidate.items():
                        if v and not tier_perf[tier_name].get(k):
                            tier_perf[tier_name][k] = v
                break

    # Derive baseline TPS from tier data if baseline record is empty
    if not baseline_perf.get("prefill_tps"):
        for tp in tier_perf.values():
            pt = tp.get("prefill_tps")
            ps = tp.get("prefill_speedup")
            if pt and ps and ps > 0:
                baseline_perf["prefill_tps"] = pt / ps
                break
    if not baseline_perf.get("decode_tps"):
        for tp in tier_perf.values():
            dt = tp.get("decode_tps")
            ds = tp.get("decode_speedup")
            if dt and ds and ds > 0:
                baseline_perf["decode_tps"] = dt / ds
                break
    # Backfill baseline TPS into tier data
    for tp in tier_perf.values():
        if not tp.get("baseline_prefill_tps"):
            tp["baseline_prefill_tps"] = baseline_perf.get("prefill_tps")
        if not tp.get("baseline_decode_tps"):
            tp["baseline_decode_tps"] = baseline_perf.get("decode_tps")
        if not tp.get("baseline_weights_gb"):
            tp["baseline_weights_gb"] = baseline_perf.get("weights_gb")
        if not tp.get("baseline_steady_state_alloc_gb"):
            tp["baseline_steady_state_alloc_gb"] = baseline_perf.get("steady_state_alloc_gb")

    return tier_perf, baseline_perf


def build_perf_section(perf):
    """Build markdown performance table from perf dict."""
    rows = []

    w = perf.get("weights_gb")
    bw = perf.get("baseline_weights_gb")
    wr = perf.get("weights_memory_reduction_pct")
    if w and bw:
        change = f"-{wr:.1f}%" if wr else "—"
        rows.append(f"| Model size | {w:.1f} GB | {bw:.1f} GB | {change} |")

    np_c = perf.get("num_params")
    if np_c:
        rows.append(f"| Parameters | {np_c:,} | — | — |")

    pt = perf.get("prefill_tps")
    bpt = perf.get("baseline_prefill_tps")
    ps = perf.get("prefill_speedup")
    if pt and bpt:
        pct = f"{(ps - 1) * 100:+.0f}%" if ps else "—"
        rows.append(f"| Prefill throughput | {pt:,.0f} tok/s | {bpt:,.0f} tok/s | {pct} |")

    dt = perf.get("decode_tps")
    bdt = perf.get("baseline_decode_tps")
    ds = perf.get("decode_speedup")
    if dt and bdt:
        dpct = f"{(ds - 1) * 100:+.0f}%" if ds else "—"
        rows.append(f"| Decode throughput | {dt:,.0f} tok/s | {bdt:,.0f} tok/s | {dpct} |")

    ss = perf.get("steady_state_alloc_gb")
    bss = perf.get("baseline_steady_state_alloc_gb")
    ssr = perf.get("steady_state_memory_reduction_pct")
    if ss and bss:
        change = f"{ssr:+.1f}%" if ssr else "—"
        rows.append(f"| VRAM (steady state) | {ss:.1f} GB | {bss:.1f} GB | {change} |")

    if not rows:
        return None

    header = "## Performance\n\n| Metric | Sculpt | Baseline | Change |\n|--------|-------:|----------|--------|\n"
    footer = "\n\n> KV-cache footprint is unchanged — Sculpt only compresses FFN layers, not attention.\n"
    return header + "\n".join(rows) + footer


def update_card(api, repo_id, perf_section):
    """Download existing model card, insert perf section, re-upload."""
    try:
        readme_path = hf_hub_download(repo_id=repo_id, filename="README.md", token=TOKEN)
        with open(readme_path) as f:
            card = f.read()
    except Exception as e:
        print(f"  ERROR downloading README: {e}")
        return False

    if "## Performance" in card:
        print(f"  already has Performance section, replacing...")
        import re
        card = re.sub(
            r"## Performance\n.*?(?=\n## )",
            perf_section + "\n",
            card,
            flags=re.DOTALL,
        )
    elif "## Quick Start" in card:
        card = card.replace("## Quick Start", perf_section + "\n## Quick Start")
    else:
        print(f"  WARNING: couldn't find insertion point")
        return False

    api.upload_file(
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        commit_message="Add performance metrics to model card",
        token=TOKEN,
    )
    return True


def main():
    if not TOKEN:
        print("ERROR: set HF_TOKEN")
        sys.exit(1)

    api = HfApi(token=TOKEN)

    print("Pulling efficiency dataset...")
    tier_perf, baseline_perf = load_efficiency_data()
    print(f"  Found data for tiers: {list(tier_perf.keys())}")
    print(f"  Baseline: {baseline_perf}")

    for tier_name, tier_info in TIERS.items():
        repo_id = f"{ORG}/{MODEL_SHORT}-{tier_info['suffix']}"
        perf = tier_perf.get(tier_name)

        if not perf:
            print(f"\n{repo_id}: no perf data found, skipping")
            continue

        perf_section = build_perf_section(perf)
        if not perf_section:
            print(f"\n{repo_id}: perf data empty, skipping")
            continue

        print(f"\n{repo_id}:")
        print(perf_section)

        ok = update_card(api, repo_id, perf_section)
        print(f"  {'UPDATED' if ok else 'FAILED'}")


if __name__ == "__main__":
    main()
