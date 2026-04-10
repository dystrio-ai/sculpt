#!/usr/bin/env python3
"""
Collect all data needed for the Physarum technical blog post.

Produces a single JSON file with four sections:
  1. physarum_internals — what the selector measures (covariance, conductance, diversity)
  2. selector_comparison — structural vs magnitude vs sensitivity vs random (PPL at matched keep_fracs)
  3. risk_scoring — per-layer risk breakdown and bracket selection
  4. component_ablation — diversity penalty on/off, distillation on/off

Usage:
  python scripts/collect_blog_data.py \
    --model-id mistralai/Mistral-7B-v0.1 \
    --outdir blog_data \
    --keep-fracs 0.90,0.85,0.80,0.75 \
    --phases all

  # Run just the fast phases (no full pipeline runs):
  python scripts/collect_blog_data.py --model-id <model> --phases internals,risk

  # Run just the selector comparison (slower, needs GPU time):
  python scripts/collect_blog_data.py --model-id <model> --phases selectors

  # Run everything including distillation ablation (slowest):
  python scripts/collect_blog_data.py --model-id <model> --phases all
"""

import argparse
import copy
import gc
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _add_src_to_path():
    root = Path(__file__).resolve().parent.parent
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


_add_src_to_path()

from dystrio_sculpt._calibrate import (
    collect_block_geometry_swiglu,
    collect_block_operator_sensitivity_swiglu,
)
from dystrio_sculpt._data import load_text_sets
from dystrio_sculpt.risk import layer_risk_score, model_risk_score, risk_aware_keep_candidates
from dystrio_sculpt.selectors.structural import (
    build_graph_from_cov,
    physarum_conductance,
    select_blocks_structural,
)
from dystrio_sculpt.selectors.baselines import select_blocks_sensitivity, select_blocks_random
from dystrio_sculpt._compile import select_blocks_magnitude  # noqa: used in magnitude path


def timestamp():
    return time.strftime("%H:%M:%S")


# ---------------------------------------------------------------------------
# Phase 1: Physarum internals — what the selector actually measures
# ---------------------------------------------------------------------------

def collect_physarum_internals(model, tokenizer, prescan_cache, device, sample_layers=None):
    """Dump intermediate artifacts from the Physarum selector for sample layers."""
    print(f"\n[{timestamp()}] === Phase 1: Physarum internals ===")

    if sample_layers is None:
        n_layers = len(prescan_cache)
        sample_layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
        sample_layers = sorted(set(sample_layers))

    results = {}

    for layer_idx in sample_layers:
        cache = prescan_cache[layer_idx]
        D = cache["D"]
        block_sens = cache.get("block_sensitivity")
        block_energy = cache.get("block_energy")
        n_blocks = D.shape[0] // cache.get("feature_multiplier", 3)

        D_np = D.numpy() if isinstance(D, torch.Tensor) else D
        u, v, w = build_graph_from_cov(D_np, k=20)
        n_nodes = D_np.shape[0]
        rng = np.random.RandomState(42)
        k_edge = physarum_conductance(u, v, w, n_nodes, n_iters=200, mu=1.5, rng=rng)

        # Run structural selector WITH diversity penalty
        sel_div, idx_div, art_div = select_blocks_structural(
            D, keep_frac=0.80, block_size=1,
            diversity_lambda=0.2, n_physarum_iters=200,
            block_sensitivity=block_sens, block_energy=block_energy,
        )

        # Run structural selector WITHOUT diversity penalty
        sel_nodiv, idx_nodiv, art_nodiv = select_blocks_structural(
            D, keep_frac=0.80, block_size=1,
            diversity_lambda=0.0, n_physarum_iters=200,
            block_sensitivity=block_sens, block_energy=block_energy,
        )

        raw_scores = art_div["block_scores"].numpy().tolist()
        adj_norm = art_div["block_adj_norm"]
        adj_stats = {}
        if adj_norm is not None:
            adj_np = adj_norm.numpy() if isinstance(adj_norm, torch.Tensor) else adj_norm
            adj_stats = {
                "mean_coupling": float(np.mean(adj_np)),
                "max_coupling": float(np.max(adj_np)),
                "std_coupling": float(np.std(adj_np)),
                "sparsity": float(np.mean(adj_np == 0)),
            }

        rerank_count = sum(1 for a, b in zip(sel_div, sel_nodiv) if a != b)
        dropped_by_diversity = [b for b in sel_nodiv if b not in set(sel_div)]
        added_by_diversity = [b for b in sel_div if b not in set(sel_nodiv)]

        results[str(layer_idx)] = {
            "n_blocks": n_blocks,
            "n_graph_edges": len(u),
            "conductance_stats": {
                "mean": float(np.mean(k_edge)),
                "std": float(np.std(k_edge)),
                "max": float(np.max(k_edge)),
                "min": float(np.min(k_edge)),
                "median": float(np.median(k_edge)),
            },
            "block_score_stats": {
                "mean": float(np.mean(raw_scores)),
                "std": float(np.std(raw_scores)),
                "max": float(np.max(raw_scores)),
                "min": float(np.min(raw_scores)),
            },
            "adj_coupling_stats": adj_stats,
            "diversity_reranking": {
                "blocks_reranked": rerank_count,
                "blocks_dropped_by_diversity": dropped_by_diversity[:10],
                "blocks_added_by_diversity": added_by_diversity[:10],
                "jaccard_similarity": len(set(sel_div) & set(sel_nodiv)) / max(1, len(set(sel_div) | set(sel_nodiv))),
            },
            "top_10_blocks_with_diversity": sel_div[:10],
            "top_10_blocks_without_diversity": sel_nodiv[:10],
        }

        print(f"  Layer {layer_idx}: {n_blocks} blocks, {len(u)} edges, "
              f"{rerank_count} blocks reranked by diversity penalty")

    return results


# ---------------------------------------------------------------------------
# Phase 2: Selector comparison — PPL at matched keep_fracs
# ---------------------------------------------------------------------------

def _eval_ppl(model, tokenizer, texts, device, max_len=2048):
    """Fast WikiText-2 perplexity evaluation."""
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    with torch.no_grad():
        for text in texts:
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len).to(device)
            if enc.input_ids.shape[1] < 2:
                continue
            out = model(**enc, labels=enc.input_ids)
            n_tok = enc.input_ids.shape[1] - 1
            total_nll += out.loss.item() * n_tok
            total_tokens += n_tok
    return math.exp(total_nll / max(1, total_tokens))


def _compress_model_with_selector(model_id, keep_frac, selector, device, dtype,
                                  texts_cal, texts_eval, prescan_cache,
                                  distill=False, diversity_lambda=0.2,
                                  outdir=None):
    """Run a single sculpt pass and return PPL + metadata."""
    from dystrio_sculpt.architectures import get_adapter
    from dystrio_sculpt.engine import compress_model

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    adapter = get_adapter(model)
    n_layers = adapter.num_layers(model)

    layer_results = {}
    for li in range(n_layers):
        cache = prescan_cache[li]
        D = cache["D"]
        block_sens = cache.get("block_sensitivity")
        block_energy = cache.get("block_energy")
        fm = cache.get("feature_multiplier", 3)
        n_blocks = D.shape[0] // fm

        if selector == "structural":
            sel, idx, art = select_blocks_structural(
                D, keep_frac=keep_frac, block_size=1,
                diversity_lambda=diversity_lambda,
                block_sensitivity=block_sens, block_energy=block_energy,
            )
        elif selector == "magnitude":
            energy = block_energy if block_energy is not None else torch.ones(n_blocks)
            keep_n = max(1, int(math.ceil(keep_frac * n_blocks)))
            ranked = sorted(range(n_blocks), key=lambda b: float(energy[b]), reverse=True)
            sel = sorted(ranked[:keep_n])
        elif selector == "sensitivity":
            sel, idx, _ = select_blocks_sensitivity(
                block_sens, keep_frac=keep_frac, block_size=1,
                block_energy=block_energy,
            )
        elif selector == "random":
            sel, idx, _ = select_blocks_random(
                n_blocks, keep_frac=keep_frac, block_size=1,
            )
        else:
            raise ValueError(f"Unknown selector: {selector}")

        block_size = adapter.block_size(model, li)
        kept_idx = []
        for b in sorted(sel):
            start = b * block_size
            end = min(start + block_size, adapter.intermediate_size(model, li))
            kept_idx.extend(range(start, end))
        kept_idx = torch.tensor(sorted(kept_idx), dtype=torch.long)

        adapter.compress_layer(model, li, kept_idx, dtype=dtype, device=device)

    model.eval()
    ppl = _eval_ppl(model, tokenizer, texts_eval, device)

    result = {
        "selector": selector,
        "keep_frac": keep_frac,
        "diversity_lambda": diversity_lambda,
        "distill": distill,
        "ppl": ppl,
    }

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return result


def collect_selector_comparison(model_id, keep_fracs, device, dtype,
                                texts_cal, texts_eval, prescan_cache, baseline_ppl):
    """Run all selectors at matched keep_fracs and measure PPL."""
    print(f"\n[{timestamp()}] === Phase 2: Selector comparison ===")

    selectors = ["structural", "magnitude", "sensitivity", "random"]
    results = []

    for kf in keep_fracs:
        for sel in selectors:
            print(f"  [{timestamp()}] {sel} @ kf={kf} ...", end=" ", flush=True)
            t0 = time.time()
            try:
                r = _compress_model_with_selector(
                    model_id, kf, sel, device, dtype,
                    texts_cal, texts_eval, prescan_cache,
                    distill=False, diversity_lambda=0.2,
                )
                r["ppl_ratio"] = r["ppl"] / baseline_ppl
                r["elapsed_s"] = time.time() - t0
                r["status"] = "ok"
                print(f"PPL={r['ppl']:.2f} ({r['ppl_ratio']:.3f}x) [{r['elapsed_s']:.0f}s]")
            except Exception as e:
                r = {
                    "selector": sel, "keep_frac": kf, "ppl": None,
                    "ppl_ratio": None, "status": f"error: {e}",
                    "elapsed_s": time.time() - t0,
                }
                print(f"FAILED: {e}")
            results.append(r)

    return results


# ---------------------------------------------------------------------------
# Phase 3: Risk scoring — per-layer breakdown
# ---------------------------------------------------------------------------

def collect_risk_scoring(prescan_cache):
    """Dump per-layer risk scores, components, and bracket selection."""
    print(f"\n[{timestamp()}] === Phase 3: Risk scoring ===")

    risk_result = model_risk_score(prescan_cache)
    aggregate = risk_result["aggregate"]
    bracket = risk_aware_keep_candidates(aggregate)

    per_layer = {}
    for key, val in risk_result.items():
        if key == "aggregate":
            continue
        per_layer[str(key)] = {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                               for k, v in val.items()}

    risk_values = [v["risk_score"] for v in per_layer.values()]

    result = {
        "aggregate_risk": float(aggregate),
        "search_bracket": bracket,
        "bracket_label": (
            "aggressive" if aggregate <= 0.35
            else "conservative" if aggregate >= 0.65
            else "standard"
        ),
        "per_layer": per_layer,
        "risk_distribution": {
            "mean": float(np.mean(risk_values)),
            "std": float(np.std(risk_values)),
            "min": float(np.min(risk_values)),
            "max": float(np.max(risk_values)),
            "low_risk_layers": sum(1 for r in risk_values if r <= 0.35),
            "medium_risk_layers": sum(1 for r in risk_values if 0.35 < r < 0.65),
            "high_risk_layers": sum(1 for r in risk_values if r >= 0.65),
        },
        "formula": {
            "description": "risk = 0.45*sigmoid((sens-0.5)/0.3) + 0.35*clamp01((top10-0.3)/0.4) + 0.20*(1-clamp01((rank_ratio-0.1)/0.5))",
            "weights": {"sensitivity": 0.45, "coupling": 0.35, "rank": 0.20},
        },
    }

    print(f"  Aggregate risk: {aggregate:.3f} → {result['bracket_label']} bracket")
    print(f"  Bracket: {bracket}")
    print(f"  Layer risk range: {np.min(risk_values):.3f} – {np.max(risk_values):.3f}")

    return result


# ---------------------------------------------------------------------------
# Phase 4: Component ablation
# ---------------------------------------------------------------------------

def collect_component_ablation(model_id, device, dtype, texts_cal, texts_eval,
                               prescan_cache, baseline_ppl, ablation_kf=0.85):
    """Ablation: diversity on/off, distillation on/off."""
    print(f"\n[{timestamp()}] === Phase 4: Component ablation @ kf={ablation_kf} ===")

    configs = [
        {"label": "structural (full)", "selector": "structural", "diversity_lambda": 0.2},
        {"label": "structural (no diversity)", "selector": "structural", "diversity_lambda": 0.0},
        {"label": "magnitude", "selector": "magnitude", "diversity_lambda": 0.0},
        {"label": "random", "selector": "random", "diversity_lambda": 0.0},
    ]

    results = []
    for cfg in configs:
        print(f"  [{timestamp()}] {cfg['label']} ...", end=" ", flush=True)
        t0 = time.time()
        try:
            r = _compress_model_with_selector(
                model_id, ablation_kf, cfg["selector"], device, dtype,
                texts_cal, texts_eval, prescan_cache,
                distill=False, diversity_lambda=cfg["diversity_lambda"],
            )
            r["label"] = cfg["label"]
            r["ppl_ratio"] = r["ppl"] / baseline_ppl
            r["elapsed_s"] = time.time() - t0
            print(f"PPL={r['ppl']:.2f} ({r['ppl_ratio']:.3f}x) [{r['elapsed_s']:.0f}s]")
        except Exception as e:
            r = {"label": cfg["label"], "ppl": None, "ppl_ratio": None,
                 "status": f"error: {e}", "elapsed_s": time.time() - t0}
            print(f"FAILED: {e}")
        results.append(r)

    return {"keep_frac": ablation_kf, "baseline_ppl": baseline_ppl, "runs": results}


# ---------------------------------------------------------------------------
# Prescan
# ---------------------------------------------------------------------------

def run_prescan(model, tokenizer, texts_cal, device, max_len=2048):
    """Run prescan and return cache dict keyed by layer index."""
    from dystrio_sculpt.architectures import get_adapter
    print(f"\n[{timestamp()}] Running prescan ...")

    adapter = get_adapter(model)
    n_layers = adapter.num_layers(model)
    prescan_cache = {}

    for li in range(n_layers):
        t0 = time.time()
        geom = collect_block_geometry_swiglu(model, tokenizer, texts_cal, li,
                                             device=device, max_len=max_len)
        sens = collect_block_operator_sensitivity_swiglu(model, tokenizer, texts_cal, li,
                                                         device=device, max_len=max_len)
        prescan_cache[li] = {
            "D": geom["D"].cpu(),
            "block_energy": geom.get("block_energy", sens.get("block_energy")),
            "block_sensitivity": sens["block_sensitivity"].cpu(),
            "feature_multiplier": geom.get("feature_multiplier", 3),
        }
        if prescan_cache[li]["block_energy"] is not None:
            prescan_cache[li]["block_energy"] = prescan_cache[li]["block_energy"].cpu()
        elapsed = time.time() - t0
        if li == 0 or (li + 1) % 8 == 0 or li == n_layers - 1:
            print(f"  Layer {li}/{n_layers-1} [{elapsed:.1f}s]")

    return prescan_cache


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Collect data for Physarum blog post")
    parser.add_argument("--model-id", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--outdir", default="blog_data")
    parser.add_argument("--keep-fracs", default="0.90,0.85,0.80,0.75",
                        help="Comma-separated keep fracs for selector comparison")
    parser.add_argument("--ablation-kf", type=float, default=0.85,
                        help="Single keep_frac for component ablation")
    parser.add_argument("--phases", default="all",
                        help="Comma-separated: internals,selectors,risk,ablation,all")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--eval-samples", type=int, default=100,
                        help="Number of eval text samples for PPL measurement")
    parser.add_argument("--cal-samples", type=int, default=128,
                        help="Number of calibration samples for prescan")
    args = parser.parse_args()

    keep_fracs = [float(x) for x in args.keep_fracs.split(",")]
    phases = set(args.phases.split(","))
    if "all" in phases:
        phases = {"internals", "selectors", "risk", "ablation"}

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    os.makedirs(args.outdir, exist_ok=True)
    output = {
        "model_id": args.model_id,
        "device": args.device,
        "dtype": args.dtype,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    print(f"[{timestamp()}] Loading model: {args.model_id}")
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=dtype).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[{timestamp()}] Loading calibration + eval texts ...")
    text_sets = load_text_sets(
        n_cal=args.cal_samples,
        n_train=max(args.cal_samples * 2, 256),
        n_eval=args.eval_samples,
        mixture_workload="general_v2",
    )
    texts_cal = text_sets["cal"]
    texts_eval = text_sets["eval_w103"]

    # Baseline PPL
    print(f"[{timestamp()}] Measuring baseline PPL ...")
    baseline_ppl = _eval_ppl(model, tokenizer, texts_eval, args.device)
    output["baseline_ppl"] = baseline_ppl
    print(f"  Baseline PPL: {baseline_ppl:.2f}")

    # Prescan (needed for all phases)
    prescan_cache = run_prescan(model, tokenizer, texts_cal, args.device)

    # Phase 1: Physarum internals
    if "internals" in phases:
        output["physarum_internals"] = collect_physarum_internals(
            model, tokenizer, prescan_cache, args.device,
        )

    # Phase 3: Risk scoring (fast, no model reload needed)
    if "risk" in phases:
        output["risk_scoring"] = collect_risk_scoring(prescan_cache)

    # Free the base model before compression phases
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Phase 2: Selector comparison
    if "selectors" in phases:
        output["selector_comparison"] = collect_selector_comparison(
            args.model_id, keep_fracs, args.device, dtype,
            texts_cal, texts_eval, prescan_cache, baseline_ppl,
        )

    # Phase 4: Component ablation
    if "ablation" in phases:
        output["component_ablation"] = collect_component_ablation(
            args.model_id, args.device, dtype,
            texts_cal, texts_eval, prescan_cache, baseline_ppl,
            ablation_kf=args.ablation_kf,
        )

    # Write output
    outpath = os.path.join(args.outdir, "blog_data.json")
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n[{timestamp()}] === Done! Output: {outpath} ===")
    print(f"  Phases completed: {', '.join(sorted(phases))}")
    print(f"  Baseline PPL: {baseline_ppl:.2f}")

    # Print summary tables for quick review
    if "selector_comparison" in output:
        print("\n--- Selector Comparison ---")
        print(f"{'Selector':<15} {'keep_frac':<10} {'PPL':<10} {'PPL ratio':<10}")
        for r in output["selector_comparison"]:
            ppl_str = f"{r['ppl']:.2f}" if r.get("ppl") else "FAIL"
            ratio_str = f"{r['ppl_ratio']:.3f}x" if r.get("ppl_ratio") else "—"
            print(f"{r['selector']:<15} {r['keep_frac']:<10} {ppl_str:<10} {ratio_str:<10}")

    if "component_ablation" in output:
        print("\n--- Component Ablation ---")
        print(f"{'Config':<30} {'PPL':<10} {'PPL ratio':<10}")
        for r in output["component_ablation"]["runs"]:
            ppl_str = f"{r['ppl']:.2f}" if r.get("ppl") else "FAIL"
            ratio_str = f"{r['ppl_ratio']:.3f}x" if r.get("ppl_ratio") else "—"
            print(f"{r['label']:<30} {ppl_str:<10} {ratio_str:<10}")


if __name__ == "__main__":
    main()
