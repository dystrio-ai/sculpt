#!/usr/bin/env python3
"""Multi-layer experiment harness for pnn_compiler.

Runs a deterministic experiment matrix:
  Phase 0: baseline (no compression)
  Phase 1: layer-set sweep at keep_frac=0.50
  Phase 2: keep_frac sweep on best layer-set from Phase 1
  Phase 3: repair_steps sweep on best (layer-set, keep_frac) from Phase 2

Features:
  - OOD evaluation (OpenWebText / C4)
  - Ablation baselines (no-repair, random-blocks + repair)
  - Optional vLLM serving benchmark
  - Gradient accumulation support
  - Repair curve logging

Usage:
    python experiments/multilayer_experiment.py
    python experiments/multilayer_experiment.py --smoke --phases 0,1 --skip-ablations
    python experiments/multilayer_experiment.py --phases 0 --outdir runs_baseline_only
    python experiments/multilayer_experiment.py --grad-accum-steps 4
    python experiments/multilayer_experiment.py --enable-vllm
    python experiments/multilayer_experiment.py --strike-gold --skip-ablations --outdir runs_gold
    python experiments/multilayer_experiment.py --strike-gold --selector structural --skip-ablations --outdir runs_structural
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import logging
import math
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pnn_compiler.config import resolve_dtype
from pnn_compiler.data import load_text_sets, load_ood_texts
from pnn_compiler.model import load_model_and_tokenizer
from pnn_compiler.calibrate import collect_ffn_importance_swiglu, collect_block_geometry_swiglu
from pnn_compiler.compile import select_blocks, compress_mlp_layer_swiglu_inplace
from pnn_compiler.structural import select_blocks_structural
from pnn_compiler.repair import repair_layers
from pnn_compiler.eval import eval_perplexity
from pnn_compiler.bench import (
    bench_prefill_tokens_per_sec, bench_decode_tokens_per_sec,
    bench_prefill_tps, bench_decode_tps,
)

log = logging.getLogger("multilayer_experiment")

# ── Experiment constants (spec-locked) ─────────────────────────────────────────

MODEL_ID = "Qwen/Qwen2-0.5B"
DTYPE = "bf16"
MAX_LEN = 256
BLOCK_SIZE = 128
SEED = 0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    DTYPE = "fp32"

N_TEXTS_CAL = 400
N_TEXTS_TRAIN = 2500
N_TEXTS_EVAL = 300
MAX_EVAL_TOKENS = 40000

REPAIR_LR = 3e-4
REPAIR_WARMUP = 100
REPAIR_WEIGHT_DECAY = 0.01

PREFILL_BATCH = 32
PREFILL_SEQ_LEN = 256
PREFILL_WARMUP_ITERS = 20
PREFILL_ITERS = 80

DECODE_TOKENS = 128
DECODE_WARMUP_ITERS = 3
DECODE_ITERS = 10

NUM_LAYERS = 24  # Qwen2-0.5B

RUNS_DIR = Path(__file__).resolve().parents[1] / "runs"


# ── Helpers ────────────────────────────────────────────────────────────────────


def _setup_determinism():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def _run_dir_name(run_id: int, layer_desc: str, keep_frac: float, repair_steps: int) -> str:
    return f"run_{run_id}_{layer_desc}_keep{keep_frac:.2f}_steps{repair_steps}"


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def _load_fresh_model():
    dtype = resolve_dtype(DTYPE)
    return load_model_and_tokenizer(MODEL_ID, DEVICE, dtype)


def _collect_metrics(
    model, tok,
    eval_w2: List[str], eval_w103: List[str],
    bench_texts: List[str], decode_text: str,
    ood_texts: Optional[List[str]] = None,
    max_eval_tokens_ood: int = 5000,
) -> Dict[str, float]:
    ppl_w2 = eval_perplexity(model, tok, eval_w2, MAX_LEN, DEVICE, MAX_EVAL_TOKENS)
    ppl_w103 = eval_perplexity(model, tok, eval_w103, MAX_LEN, DEVICE, MAX_EVAL_TOKENS)
    prefill = bench_prefill_tps(
        model, tok, bench_texts, PREFILL_SEQ_LEN, DEVICE,
        PREFILL_WARMUP_ITERS, PREFILL_ITERS,
    )
    decode = bench_decode_tps(
        model, tok, decode_text, MAX_LEN, DEVICE,
        DECODE_TOKENS, DECODE_WARMUP_ITERS, DECODE_ITERS,
    )
    metrics: Dict[str, float] = {
        "ppl_w2_test": ppl_w2,
        "ppl_w103_valid": ppl_w103,
        "prefill_tokens_per_sec": prefill,
        "decode_tokens_per_sec": decode,
    }
    if ood_texts:
        metrics["ppl_ood"] = eval_perplexity(
            model, tok, ood_texts, MAX_LEN, DEVICE, max_eval_tokens_ood,
        )
    return metrics


def _select_random_blocks(
    ffn_dim: int, block_size: int, n_keep_blocks: int, seed: int,
) -> torch.Tensor:
    """Select random FFN blocks deterministically.

    Uses seed = SEED + layer_idx for reproducibility.
    Returns a kept_idx tensor compatible with compress_mlp_layer_swiglu_inplace.
    """
    n_blocks = math.ceil(ffn_dim / block_size)
    rng = random.Random(seed)
    kept = sorted(rng.sample(range(n_blocks), min(n_keep_blocks, n_blocks)))
    idx: list[int] = []
    for b in kept:
        lo = b * block_size
        hi = min(ffn_dim, (b + 1) * block_size)
        idx.extend(range(lo, hi))
    return torch.tensor(idx, dtype=torch.long)


def _time_to_recover_90pct(
    curve_points: list[dict[str, Any]],
    post_repair_ppl_w103: float,
) -> Optional[float]:
    """First curve opt_step where ppl_w103 is within 10% of final post-repair."""
    if not curve_points:
        return None
    threshold = post_repair_ppl_w103 * 1.10
    for pt in curve_points:
        if pt.get("ppl_w103_valid", float("inf")) <= threshold:
            return float(pt["opt_step"])
    return None


def _safe_round(v: Any, digits: int = 4) -> Any:
    return round(v, digits) if v is not None else None


def _safe_div(a: Any, b: float) -> Optional[float]:
    if a is None:
        return None
    return a / max(1e-9, b)


# ── Selector dispatch ─────────────────────────────────────────────────────────


def _select_for_layer(
    model, tok, li: int, texts_cal, selector: str,
    importance_vectors: Dict[int, torch.Tensor],
    kept_indices: Dict[int, torch.Tensor],
    structural_artifacts: Dict[int, Dict[str, Any]],
    keep_frac: float,
) -> Tuple[List[int], torch.Tensor]:
    """Run calibration + block selection for one layer using the chosen selector."""
    if selector == "structural":
        geom = collect_block_geometry_swiglu(
            model, tok, li, texts_cal, MAX_LEN, DEVICE,
            block_size=BLOCK_SIZE, max_tokens=30_000,
        )
        kept_blocks, kept_idx, arts = select_blocks_structural(
            geom["D"], keep_frac, BLOCK_SIZE, topk_edges=20,
            block_energy=geom.get("block_energy"),
            feature_multiplier=geom.get("feature_multiplier", 3),
        )
        importance_vectors[li] = geom["D"].cpu()
        kept_indices[li] = kept_idx.cpu()
        structural_artifacts[li] = {
            "D": geom["D"].cpu(),
            "block_energy": geom["block_energy"].cpu() if geom.get("block_energy") is not None else None,
            "edges": arts["edges"].cpu(),
            "k_edge": arts["k_edge"].cpu(),
            "block_scores": arts["block_scores"].cpu(),
            "diagnostics": arts["diagnostics"],
        }
        return kept_blocks, kept_idx.to(DEVICE)
    else:
        imp = collect_ffn_importance_swiglu(
            model, tok, li, texts_cal, MAX_LEN, DEVICE,
        )
        kept_blocks, kept_idx = select_blocks(imp, BLOCK_SIZE, keep_frac)
        importance_vectors[li] = imp.cpu()
        kept_indices[li] = kept_idx.cpu()
        return kept_blocks, kept_idx


# ── Sanity checks ─────────────────────────────────────────────────────────────


def _assert_physical_slicing(
    model, layers: List[int], original_ffn_dims: Dict[int, int],
) -> None:
    for li in layers:
        mlp = model.model.layers[li].mlp
        new_ffn = mlp.gate_proj.out_features
        orig_ffn = original_ffn_dims[li]

        assert new_ffn < orig_ffn, (
            f"Layer {li}: FFN dim not reduced ({new_ffn} >= {orig_ffn})"
        )
        hidden = mlp.gate_proj.in_features
        assert mlp.gate_proj.weight.shape == (new_ffn, hidden)
        assert mlp.up_proj.weight.shape == (new_ffn, hidden)
        assert mlp.down_proj.weight.shape == (hidden, new_ffn)
        assert type(mlp.gate_proj) is torch.nn.Linear
        assert type(mlp.up_proj) is torch.nn.Linear
        assert type(mlp.down_proj) is torch.nn.Linear


def _assert_repair_freeze(model, layers: List[int]) -> None:
    target_ids = set()
    for li in layers:
        for p in model.model.layers[li].mlp.parameters():
            target_ids.add(id(p))

    for name, p in model.named_parameters():
        if id(p) in target_ids:
            assert p.requires_grad, f"Target MLP param {name} should be trainable"
        else:
            assert not p.requires_grad, f"Non-target param {name} should be frozen"


def _assert_no_masking(model, layers: List[int], original_ffn_dims: Dict[int, int]) -> None:
    for li in layers:
        mlp = model.model.layers[li].mlp
        assert mlp.gate_proj.out_features < original_ffn_dims[li], (
            f"Layer {li}: gate_proj.out_features unchanged — masking suspected"
        )


# ── Random-blocks ablation variant ────────────────────────────────────────────


def _run_random_variant(
    layers: List[int],
    compile_info: Dict[int, Dict[str, Any]],
    texts: Dict[str, List[str]],
    repair_steps: int,
    grad_accum_steps: int,
    ood_texts: Optional[List[str]],
    max_eval_tokens_ood: int,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Run random-blocks + repair ablation.  Returns (metrics_pre, metrics_post)."""
    log.info("  ── Random-blocks variant ──")
    _setup_determinism()
    model, tok = _load_fresh_model()
    dtype = resolve_dtype(DTYPE)

    original_ffn_dims = {li: info["original_ffn"] for li, info in compile_info.items()}

    for li in layers:
        n_keep = compile_info[li]["kept_blocks"]
        random_idx = _select_random_blocks(
            compile_info[li]["original_ffn"], BLOCK_SIZE, n_keep, SEED + li,
        )
        compress_mlp_layer_swiglu_inplace(model, li, random_idx.to(DEVICE), dtype, DEVICE)

    _assert_physical_slicing(model, layers, original_ffn_dims)
    _assert_no_masking(model, layers, original_ffn_dims)
    log.info("    [OK] Random variant: physical slicing verified")

    for li in layers:
        expected = compile_info[li]["ffn_kept"]
        actual = model.model.layers[li].mlp.gate_proj.out_features
        assert actual == expected, (
            f"Layer {li}: random variant kept {actual} neurons != main {expected}"
        )

    eval_w2 = texts["eval_w2"]
    eval_w103 = texts["eval_w103"]
    bench_texts = eval_w2[:PREFILL_BATCH]
    decode_text = eval_w2[0]

    metrics_pre = _collect_metrics(
        model, tok, eval_w2, eval_w103, bench_texts, decode_text,
        ood_texts, max_eval_tokens_ood,
    )
    log.info(
        f"    Pre-repair:  PPL w2={metrics_pre['ppl_w2_test']:.2f}  "
        f"w103={metrics_pre['ppl_w103_valid']:.2f}"
    )

    if repair_steps > 0:
        repair_layers(
            model=model, tokenizer=tok, texts_train=texts["train"],
            layers=layers, steps=repair_steps, lr=REPAIR_LR,
            warmup=REPAIR_WARMUP, weight_decay=REPAIR_WEIGHT_DECAY,
            max_len=MAX_LEN, device=DEVICE, log_every=500,
            grad_accum_steps=grad_accum_steps,
        )

    metrics_post = _collect_metrics(
        model, tok, eval_w2, eval_w103, bench_texts, decode_text,
        ood_texts, max_eval_tokens_ood,
    )
    log.info(
        f"    Post-repair: PPL w2={metrics_post['ppl_w2_test']:.2f}  "
        f"w103={metrics_post['ppl_w103_valid']:.2f}"
    )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics_pre, metrics_post


# ── Optional vLLM benchmark helper ────────────────────────────────────────────


def _try_vllm_benchmark(
    model_path: str, prompts: List[str], rdir: Path,
) -> Optional[Dict[str, Any]]:
    """Run vLLM serving benchmark if the library is available."""
    bench_path = Path(__file__).resolve().parent / "vllm_benchmark.py"
    if not bench_path.exists():
        log.warning("  vllm_benchmark.py not found — skipping")
        return None

    try:
        spec = importlib.util.spec_from_file_location("vllm_benchmark", str(bench_path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    except Exception as exc:
        log.warning("  Failed to load vllm_benchmark module: %s", exc)
        return None

    if not mod.is_available():
        log.info("  vLLM not installed — skipping serving benchmark")
        return None

    try:
        result = mod.benchmark_model(model_path, prompts)
        _write_json(rdir / "vllm_metrics.json", result)
        log.info(f"  vLLM metrics written to: {rdir / 'vllm_metrics.json'}")
        return result
    except Exception as exc:
        log.warning("  vLLM benchmark failed: %s", exc)
        return None


# ── Run execution ─────────────────────────────────────────────────────────────


def run_baseline(
    texts: Dict[str, List[str]],
    ood_texts: Optional[List[str]] = None,
    max_eval_tokens_ood: int = 5000,
    enable_vllm: bool = False,
    vllm_prompts: Optional[List[str]] = None,
) -> Dict[str, Any]:
    log.info("=" * 70)
    log.info("Run 0: BASELINE (no compression)")
    log.info("=" * 70)

    _setup_determinism()
    model, tok = _load_fresh_model()

    n_layers = len(model.model.layers)
    assert n_layers == NUM_LAYERS, f"Expected {NUM_LAYERS} layers, got {n_layers}"
    log.info(f"  Model loaded: {n_layers} layers")

    eval_w2 = texts["eval_w2"]
    eval_w103 = texts["eval_w103"]
    bench_texts = eval_w2[:PREFILL_BATCH]
    decode_text = eval_w2[0]

    metrics = _collect_metrics(
        model, tok, eval_w2, eval_w103, bench_texts, decode_text,
        ood_texts, max_eval_tokens_ood,
    )

    rdir = RUNS_DIR / _run_dir_name(0, "baseline", 1.00, 0)
    _write_json(rdir / "config.json", {
        "run_id": 0, "phase": 0, "layer_desc": "baseline",
        "layers": [], "keep_frac": 1.0, "repair_steps": 0,
        "model_id": MODEL_ID, "dtype": DTYPE, "seed": SEED,
        "block_size": BLOCK_SIZE, "max_len": MAX_LEN,
        "grad_accum_steps": 1,
        "max_eval_tokens_ood": max_eval_tokens_ood,
        "compile_wall_time_seconds": 0.0,
    })
    _write_json(rdir / "compile_report.json", {})
    _write_json(rdir / "metrics_pre.json", metrics)
    _write_json(rdir / "metrics_post.json", metrics)

    ppl_ood = metrics.get("ppl_ood")
    log.info(
        f"  PPL w2={metrics['ppl_w2_test']:.2f}  w103={metrics['ppl_w103_valid']:.2f}"
        + (f"  ood={ppl_ood:.2f}" if ppl_ood is not None else "")
    )
    log.info(
        f"  prefill={metrics['prefill_tokens_per_sec']:.0f} tok/s  "
        f"decode={metrics['decode_tokens_per_sec']:.0f} tok/s"
    )

    vllm_metrics: Optional[Dict[str, Any]] = None
    if enable_vllm and vllm_prompts:
        vllm_metrics = _try_vllm_benchmark(MODEL_ID, vllm_prompts, rdir)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "run_id": 0,
        "phase": 0,
        "layer_desc": "baseline",
        "layers": [],
        "keep_frac": 1.0,
        "repair_steps": 0,
        "actual_repair_steps": 0,
        "selector_name": "swiglu_mag",
        "grad_accum_steps": 1,
        "early_stopped": False,
        "guardrail_failed": False,
        "staged": False,
        "stages_completed": 0,
        "ppl_w2_test_pre": metrics["ppl_w2_test"],
        "ppl_w103_valid_pre": metrics["ppl_w103_valid"],
        "ppl_w2_test_post": metrics["ppl_w2_test"],
        "ppl_w103_valid_post": metrics["ppl_w103_valid"],
        "ppl_ood_pre": ppl_ood,
        "ppl_ood_post": ppl_ood,
        "prefill_tokens_per_sec_pre": metrics["prefill_tokens_per_sec"],
        "prefill_tokens_per_sec_post": metrics["prefill_tokens_per_sec"],
        "decode_tokens_per_sec_pre": metrics["decode_tokens_per_sec"],
        "decode_tokens_per_sec_post": metrics["decode_tokens_per_sec"],
        "repair_wall_time_seconds": 0.0,
        "compile_wall_time_seconds": 0.0,
        "time_to_recover_90pct": None,
        # Ablation variants (N/A for baseline)
        "ppl_w103_post_no_repair": metrics["ppl_w103_valid"],
        "ppl_w103_post_random": None,
        "ppl_ood_post_no_repair": ppl_ood,
        "ppl_ood_post_random": None,
        "prefill_tps_no_repair": metrics["prefill_tokens_per_sec"],
        "prefill_tps_random": None,
        "decode_tps_no_repair": metrics["decode_tokens_per_sec"],
        "decode_tps_random": None,
        "vllm_metrics": vllm_metrics,
    }


def run_compressed(
    run_id: int,
    phase: int,
    layer_desc: str,
    layers: List[int],
    keep_frac: float,
    repair_steps: int,
    texts: Dict[str, List[str]],
    selector: str = "swiglu_mag",
    grad_accum_steps: int = 1,
    curve_every: int = 250,
    curve_eval_texts: int = 50,
    curve_max_eval_tokens: int = 5000,
    ood_texts: Optional[List[str]] = None,
    max_eval_tokens_ood: int = 5000,
    skip_ablations: bool = False,
    enable_vllm: bool = False,
    vllm_prompts: Optional[List[str]] = None,
    save_artifacts: bool = False,
    ppl_guardrail: float = 0,
    early_stop_patience: int = 0,
    staged: bool = False,
    stage_size: int = 6,
    stage_repair_steps: int = 500,
    stage_guardrail: float = 200.0,
) -> Dict[str, Any]:
    log.info("=" * 70)
    log.info(
        f"Run {run_id}: phase={phase}  {layer_desc}  "
        f"keep_frac={keep_frac}  repair_steps={repair_steps}"
    )
    log.info("=" * 70)

    _setup_determinism()
    model, tok = _load_fresh_model()
    dtype = resolve_dtype(DTYPE)

    eval_w2 = texts["eval_w2"]
    eval_w103 = texts["eval_w103"]
    bench_texts = eval_w2[:PREFILL_BATCH]
    decode_text = eval_w2[0]

    rdir = RUNS_DIR / _run_dir_name(run_id, layer_desc, keep_frac, repair_steps)

    original_ffn_dims: Dict[int, int] = {}
    for li in layers:
        original_ffn_dims[li] = model.model.layers[li].mlp.gate_proj.out_features

    # ── Calibrate + compile + repair ─────────────────────────────────────────
    compile_report: Dict[str, Any] = {}
    compile_info: Dict[int, Dict[str, Any]] = {}
    importance_vectors: Dict[int, torch.Tensor] = {}
    kept_indices: Dict[int, torch.Tensor] = {}
    structural_artifacts: Dict[int, Dict[str, Any]] = {}
    stages_completed = 0

    if staged:
        # ── STAGED: compress + repair in chunks ──────────────────────────────
        metrics_pre = _collect_metrics(
            model, tok, eval_w2, eval_w103, bench_texts, decode_text,
            ood_texts, max_eval_tokens_ood,
        )
        log.info(
            f"  Pre-compression: PPL w2={metrics_pre['ppl_w2_test']:.2f}  "
            f"w103={metrics_pre['ppl_w103_valid']:.2f}  "
            f"prefill={metrics_pre['prefill_tokens_per_sec']:.0f}  "
            f"decode={metrics_pre['decode_tokens_per_sec']:.0f}"
        )

        chunks = [layers[i:i + stage_size] for i in range(0, len(layers), stage_size)]
        log.info(
            f"  Staged compression: {len(chunks)} stages of up to "
            f"{stage_size} layers each"
        )

        compile_wall = 0.0
        repair_wall = 0.0
        total_steps = 0
        any_early_stopped = False
        guardrail_failed = False
        all_curve_points: list[dict[str, Any]] = []

        curve_w2 = eval_w2[:curve_eval_texts]
        curve_w103 = eval_w103[:curve_eval_texts]

        def curve_fn(opt_step: int) -> Dict[str, float]:
            return {
                "ppl_w2_test": eval_perplexity(
                    model, tok, curve_w2, MAX_LEN, DEVICE, curve_max_eval_tokens,
                ),
                "ppl_w103_valid": eval_perplexity(
                    model, tok, curve_w103, MAX_LEN, DEVICE, curve_max_eval_tokens,
                ),
            }

        for si, chunk in enumerate(chunks):
            log.info(f"  ── Stage {si + 1}/{len(chunks)}: layers {chunk} ──")

            ct0 = time.time()
            for li in chunk:
                kept_blocks, kept_idx = _select_for_layer(
                    model, tok, li, texts["cal"], selector,
                    importance_vectors, kept_indices, structural_artifacts,
                    keep_frac,
                )
                rep = compress_mlp_layer_swiglu_inplace(
                    model, li, kept_idx, dtype, DEVICE,
                )
                info = {
                    "kept_blocks": len(kept_blocks),
                    "original_ffn": original_ffn_dims[li],
                    "ffn_kept": rep["ffn_kept"],
                }
                compile_info[li] = info
                compile_report[str(li)] = {**info, **rep}
            compile_wall += time.time() - ct0

            _assert_physical_slicing(model, chunk, original_ffn_dims)
            _assert_no_masking(model, chunk, original_ffn_dims)
            log.info(f"    [OK] Stage {si + 1} physical slicing verified")

            stage_ppl = eval_perplexity(
                model, tok, curve_w103, MAX_LEN, DEVICE, curve_max_eval_tokens,
            )
            log.info(f"    Post-compile ppl_w103={stage_ppl:.2f}")

            if stage_guardrail > 0 and stage_ppl > stage_guardrail:
                log.warning(
                    f"    [STAGE GUARDRAIL] ppl_w103={stage_ppl:.2f} > "
                    f"{stage_guardrail} — aborting remaining stages"
                )
                guardrail_failed = True
                break

            warmup_stage = min(REPAIR_WARMUP, stage_repair_steps // 5)
            log.info(
                f"    Repairing layers {chunk}  steps={stage_repair_steps}  "
                f"warmup={warmup_stage}  grad_accum={grad_accum_steps}"
            )

            rt0 = time.time()
            sr = repair_layers(
                model=model, tokenizer=tok, texts_train=texts["train"],
                layers=chunk, steps=stage_repair_steps, lr=REPAIR_LR,
                warmup=warmup_stage, weight_decay=REPAIR_WEIGHT_DECAY,
                max_len=MAX_LEN, device=DEVICE, log_every=200,
                grad_accum_steps=grad_accum_steps,
                curve_fn=curve_fn, curve_every=curve_every,
                early_stop_patience=early_stop_patience,
            )
            repair_wall += time.time() - rt0

            stage_steps = int(sr["steps"])
            total_steps += stage_steps
            if sr.get("early_stopped"):
                any_early_stopped = True

            for pt in sr.get("curve", []):
                pt["stage"] = si
            all_curve_points.extend(sr.get("curve", []))

            stages_completed += 1
            post_ppl = eval_perplexity(
                model, tok, curve_w103, MAX_LEN, DEVICE, curve_max_eval_tokens,
            )
            log.info(
                f"    Post-repair ppl_w103={post_ppl:.2f}  "
                f"(steps={stage_steps}, total={total_steps})"
            )

        repair_result: Dict[str, Any] = {
            "steps": float(total_steps),
            "microsteps": 0.0,
            "curve": all_curve_points,
            "early_stopped": any_early_stopped,
        }
        actual_repair_steps = total_steps
        log.info(
            f"  Staged compression complete: {stages_completed}/{len(chunks)} "
            f"stages, {actual_repair_steps} total repair steps"
        )

    else:
        # ── NON-STAGED: calibrate + compile all at once (original path) ──────
        compile_t0 = time.time()
        for li in layers:
            kept_blocks, kept_idx = _select_for_layer(
                model, tok, li, texts["cal"], selector,
                importance_vectors, kept_indices, structural_artifacts,
                keep_frac,
            )
            rep = compress_mlp_layer_swiglu_inplace(model, li, kept_idx, dtype, DEVICE)
            info = {
                "kept_blocks": len(kept_blocks),
                "original_ffn": original_ffn_dims[li],
                "ffn_kept": rep["ffn_kept"],
            }
            compile_info[li] = info
            compile_report[str(li)] = {**info, **rep}
        compile_wall = time.time() - compile_t0

        _assert_physical_slicing(model, layers, original_ffn_dims)
        _assert_no_masking(model, layers, original_ffn_dims)
        log.info("  [OK] Physical slicing verified (shapes reduced, no masking)")

        metrics_pre = _collect_metrics(
            model, tok, eval_w2, eval_w103, bench_texts, decode_text,
            ood_texts, max_eval_tokens_ood,
        )
        log.info(
            f"  Pre-repair:  PPL w2={metrics_pre['ppl_w2_test']:.2f}  "
            f"w103={metrics_pre['ppl_w103_valid']:.2f}  "
            f"prefill={metrics_pre['prefill_tokens_per_sec']:.0f}  "
            f"decode={metrics_pre['decode_tokens_per_sec']:.0f}"
            + (f"  ood={metrics_pre['ppl_ood']:.2f}" if "ppl_ood" in metrics_pre else "")
        )

        guardrail_failed = False
        if ppl_guardrail > 0 and metrics_pre["ppl_w103_valid"] > ppl_guardrail:
            log.warning(
                f"  [GUARDRAIL] ppl_w103={metrics_pre['ppl_w103_valid']:.2f} > "
                f"threshold {ppl_guardrail} — skipping repair, marking as failed"
            )
            guardrail_failed = True

        repair_wall = 0.0
        repair_result = {
            "steps": 0.0, "microsteps": 0.0, "curve": [], "early_stopped": False,
        }

        if repair_steps > 0 and not guardrail_failed:
            curve_w2 = eval_w2[:curve_eval_texts]
            curve_w103 = eval_w103[:curve_eval_texts]

            def curve_fn(opt_step: int) -> Dict[str, float]:
                return {
                    "ppl_w2_test": eval_perplexity(
                        model, tok, curve_w2, MAX_LEN, DEVICE, curve_max_eval_tokens,
                    ),
                    "ppl_w103_valid": eval_perplexity(
                        model, tok, curve_w103, MAX_LEN, DEVICE, curve_max_eval_tokens,
                    ),
                }

            log.info(
                f"  Repair uses grad_accum_steps={grad_accum_steps} "
                f"(optimizer steps = {repair_steps}, "
                f"microsteps = {repair_steps * grad_accum_steps})"
            )
            if early_stop_patience > 0:
                log.info(
                    f"  Early-stop enabled: patience={early_stop_patience} "
                    f"checkpoints on ppl_w103_valid"
                )

            repair_t0 = time.time()
            repair_result = repair_layers(
                model=model,
                tokenizer=tok,
                texts_train=texts["train"],
                layers=layers,
                steps=repair_steps,
                lr=REPAIR_LR,
                warmup=REPAIR_WARMUP,
                weight_decay=REPAIR_WEIGHT_DECAY,
                max_len=MAX_LEN,
                device=DEVICE,
                log_every=500,
                grad_accum_steps=grad_accum_steps,
                curve_fn=curve_fn,
                curve_every=curve_every,
                early_stop_patience=early_stop_patience,
            )
            repair_wall = time.time() - repair_t0

            _assert_repair_freeze(model, layers)
            log.info("  [OK] Repair freeze discipline verified")
            if repair_result.get("early_stopped"):
                log.info(
                    f"  Repair early-stopped at step {int(repair_result['steps'])} "
                    f"(of {repair_steps} planned)"
                )

        actual_repair_steps = int(repair_result.get("steps", 0))

    # ── Save artifacts (shared) ───────────────────────────────────────────────
    if save_artifacts:
        artifact_paths: Dict[str, str] = {}
        for li in layers:
            if li not in importance_vectors:
                continue
            rdir.mkdir(parents=True, exist_ok=True)
            imp_path = rdir / f"importance_layer_{li}.pt"
            idx_path = rdir / f"kept_indices_layer_{li}.pt"
            torch.save(importance_vectors[li], imp_path)
            torch.save(kept_indices[li], idx_path)
            artifact_paths[f"importance_layer_{li}"] = str(imp_path.name)
            artifact_paths[f"kept_indices_layer_{li}"] = str(idx_path.name)

            if li in structural_artifacts:
                sa = structural_artifacts[li]
                for key, fname in [
                    ("D", f"block_cov_layer_{li}.pt"),
                    ("edges", f"coupling_edges_layer_{li}.pt"),
                    ("k_edge", f"coupling_k_layer_{li}.pt"),
                    ("block_scores", f"block_scores_layer_{li}.pt"),
                ]:
                    p = rdir / fname
                    torch.save(sa[key], p)
                    artifact_paths[fname.replace(".pt", "")] = fname

        n_saved = sum(1 for k in artifact_paths if k.startswith("importance_layer_"))
        log.info(f"  Saved artifacts for {n_saved} layers (selector={selector})")

    # ── Post metrics (shared) ─────────────────────────────────────────────────
    if guardrail_failed and not staged:
        metrics_post = metrics_pre
    else:
        metrics_post = _collect_metrics(
            model, tok, eval_w2, eval_w103, bench_texts, decode_text,
            ood_texts, max_eval_tokens_ood,
        )
    log.info(
        f"  Post-repair: PPL w2={metrics_post['ppl_w2_test']:.2f}  "
        f"w103={metrics_post['ppl_w103_valid']:.2f}  "
        f"prefill={metrics_post['prefill_tokens_per_sec']:.0f}  "
        f"decode={metrics_post['decode_tokens_per_sec']:.0f}"
        + (f"  ood={metrics_post['ppl_ood']:.2f}" if "ppl_ood" in metrics_post else "")
    )

    actual_repair_steps = int(actual_repair_steps)

    # ── Save model for vLLM before deleting ───────────────────────────────────
    saved_model_dir: Optional[Path] = None
    if enable_vllm:
        saved_model_dir = rdir / "_model_saved"
        try:
            model.save_pretrained(str(saved_model_dir))
            tok.save_pretrained(str(saved_model_dir))
        except Exception as exc:
            log.warning("  Model save for vLLM failed: %s", exc)
            saved_model_dir = None

    # ── Write main-variant artifacts ──────────────────────────────────────────
    config_data: Dict[str, Any] = {
        "run_id": run_id, "phase": phase, "layer_desc": layer_desc,
        "layers": layers, "keep_frac": keep_frac, "repair_steps": repair_steps,
        "actual_repair_steps": actual_repair_steps,
        "selector_name": selector,
        "model_id": MODEL_ID, "dtype": DTYPE, "seed": SEED,
        "block_size": BLOCK_SIZE, "max_len": MAX_LEN,
        "grad_accum_steps": grad_accum_steps,
        "curve_every": curve_every,
        "curve_eval_texts": curve_eval_texts,
        "curve_max_eval_tokens": curve_max_eval_tokens,
        "skip_ablations": skip_ablations,
        "max_eval_tokens_ood": max_eval_tokens_ood,
        "compile_wall_time_seconds": round(compile_wall, 2),
        "early_stopped": repair_result.get("early_stopped", False),
        "guardrail_failed": guardrail_failed,
        "staged": staged,
        "stage_size": stage_size if staged else 0,
        "stage_repair_steps": stage_repair_steps if staged else 0,
        "stage_guardrail": stage_guardrail if staged else 0,
        "stages_completed": stages_completed,
    }
    if save_artifacts:
        config_data["artifact_files"] = artifact_paths
    _write_json(rdir / "config.json", config_data)
    _write_json(rdir / "compile_report.json", compile_report)
    _write_json(rdir / "metrics_pre.json", metrics_pre)
    _write_json(rdir / "metrics_post.json", metrics_post)
    _write_json(rdir / "variant_main" / "metrics_pre.json", metrics_pre)
    _write_json(rdir / "variant_main" / "metrics_post.json", metrics_post)

    curve_points = repair_result.get("curve", [])
    _write_json(rdir / "repair_curve.json", {
        "curve_every": curve_every,
        "curve_eval_texts": curve_eval_texts,
        "points": curve_points,
    })
    log.info(f"  Repair curve written to: {rdir / 'repair_curve.json'}")

    # No-repair variant = pre-repair metrics (same compile, no repair)
    _write_json(rdir / "variant_no_repair" / "metrics_post.json", metrics_pre)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Optional vLLM benchmark ───────────────────────────────────────────────
    vllm_metrics: Optional[Dict[str, Any]] = None
    if enable_vllm and vllm_prompts and saved_model_dir and saved_model_dir.exists():
        vllm_metrics = _try_vllm_benchmark(
            str(saved_model_dir), vllm_prompts, rdir,
        )
        shutil.rmtree(saved_model_dir, ignore_errors=True)
    elif saved_model_dir and saved_model_dir.exists():
        shutil.rmtree(saved_model_dir, ignore_errors=True)

    # ── Random-blocks ablation variant ────────────────────────────────────────
    random_metrics_pre: Optional[Dict[str, float]] = None
    random_metrics_post: Optional[Dict[str, float]] = None

    if not skip_ablations:
        random_metrics_pre, random_metrics_post = _run_random_variant(
            layers=layers,
            compile_info=compile_info,
            texts=texts,
            repair_steps=repair_steps,
            grad_accum_steps=grad_accum_steps,
            ood_texts=ood_texts,
            max_eval_tokens_ood=max_eval_tokens_ood,
        )
        _write_json(
            rdir / "variant_random_blocks" / "metrics_pre.json", random_metrics_pre,
        )
        _write_json(
            rdir / "variant_random_blocks" / "metrics_post.json", random_metrics_post,
        )
        log.info("  Ablation artifacts written to variant_no_repair/ and variant_random_blocks/")

    # ── Build result dict ─────────────────────────────────────────────────────
    recover_step = _time_to_recover_90pct(
        curve_points, metrics_post["ppl_w103_valid"],
    )

    # Aggregate geometry diagnostics across layers (mean)
    geom_diags: Dict[str, Optional[float]] = {
        "eff_rank95_D": None, "gini_k": None, "top10_edge_mass": None,
    }
    if structural_artifacts:
        keys = ["eff_rank95_D", "gini_k", "top10_edge_mass"]
        for k in keys:
            vals = [sa["diagnostics"][k] for sa in structural_artifacts.values()
                    if "diagnostics" in sa and k in sa["diagnostics"]]
            if vals:
                geom_diags[k] = round(sum(vals) / len(vals), 4)

    return {
        "run_id": run_id,
        "phase": phase,
        "layer_desc": layer_desc,
        "layers": layers,
        "keep_frac": keep_frac,
        "repair_steps": repair_steps,
        "actual_repair_steps": actual_repair_steps,
        "selector_name": selector,
        "grad_accum_steps": grad_accum_steps,
        "early_stopped": repair_result.get("early_stopped", False),
        "guardrail_failed": guardrail_failed,
        "staged": staged,
        "stages_completed": stages_completed,
        # Main variant metrics
        "ppl_w2_test_pre": metrics_pre["ppl_w2_test"],
        "ppl_w103_valid_pre": metrics_pre["ppl_w103_valid"],
        "ppl_w2_test_post": metrics_post["ppl_w2_test"],
        "ppl_w103_valid_post": metrics_post["ppl_w103_valid"],
        "ppl_ood_pre": metrics_pre.get("ppl_ood"),
        "ppl_ood_post": metrics_post.get("ppl_ood"),
        "prefill_tokens_per_sec_pre": metrics_pre["prefill_tokens_per_sec"],
        "prefill_tokens_per_sec_post": metrics_post["prefill_tokens_per_sec"],
        "decode_tokens_per_sec_pre": metrics_pre["decode_tokens_per_sec"],
        "decode_tokens_per_sec_post": metrics_post["decode_tokens_per_sec"],
        "repair_wall_time_seconds": repair_wall,
        "compile_wall_time_seconds": compile_wall,
        "time_to_recover_90pct": recover_step,
        # No-repair variant (= pre-repair metrics from main)
        "ppl_w103_post_no_repair": metrics_pre["ppl_w103_valid"],
        "ppl_ood_post_no_repair": metrics_pre.get("ppl_ood"),
        "prefill_tps_no_repair": metrics_pre["prefill_tokens_per_sec"],
        "decode_tps_no_repair": metrics_pre["decode_tokens_per_sec"],
        # Random-blocks variant
        "ppl_w103_post_random": (
            random_metrics_post["ppl_w103_valid"] if random_metrics_post else None
        ),
        "ppl_ood_post_random": (
            random_metrics_post.get("ppl_ood") if random_metrics_post else None
        ),
        "prefill_tps_random": (
            random_metrics_post["prefill_tokens_per_sec"] if random_metrics_post else None
        ),
        "decode_tps_random": (
            random_metrics_post["decode_tokens_per_sec"] if random_metrics_post else None
        ),
        "vllm_metrics": vllm_metrics,
        # Geometry diagnostics (structural selector)
        **geom_diags,
    }


# ── Summary ────────────────────────────────────────────────────────────────────


def write_summary(
    all_results: List[Dict[str, Any]], baseline: Dict[str, Any],
) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    baseline_prefill = baseline["prefill_tokens_per_sec_post"]
    baseline_decode = baseline["decode_tokens_per_sec_post"]
    baseline_ppl_ood = baseline.get("ppl_ood_post")

    rows: List[Dict[str, Any]] = []
    for r in all_results:
        rows.append({
            "run_id": r["run_id"],
            "layer_desc": r["layer_desc"],
            "keep_frac": r["keep_frac"],
            "repair_steps": r["repair_steps"],
            "grad_accum_steps": r.get("grad_accum_steps", 1),
            # Main variant (backwards compat)
            "ppl_w2_test_post": _safe_round(r["ppl_w2_test_post"]),
            "ppl_w103_valid_post": _safe_round(r["ppl_w103_valid_post"]),
            "prefill_speedup_vs_baseline": _safe_round(
                r["prefill_tokens_per_sec_post"] / max(1e-9, baseline_prefill),
            ),
            "decode_speedup_vs_baseline": _safe_round(
                r["decode_tokens_per_sec_post"] / max(1e-9, baseline_decode),
            ),
            "repair_wall_time_seconds": round(r["repair_wall_time_seconds"], 2),
            "time_to_recover_90pct": r.get("time_to_recover_90pct"),
            # OOD
            "ppl_ood_post": _safe_round(r.get("ppl_ood_post")),
            "ppl_ood_delta_vs_baseline": _safe_round(
                (r["ppl_ood_post"] - baseline_ppl_ood)
                if r.get("ppl_ood_post") is not None and baseline_ppl_ood is not None
                else None,
            ),
            # Variant: main (redundant with above, but spec-required)
            "ppl_w103_post_main": _safe_round(r["ppl_w103_valid_post"]),
            "prefill_speedup_main": _safe_round(
                r["prefill_tokens_per_sec_post"] / max(1e-9, baseline_prefill),
            ),
            "decode_speedup_main": _safe_round(
                r["decode_tokens_per_sec_post"] / max(1e-9, baseline_decode),
            ),
            "ppl_ood_post_main": _safe_round(r.get("ppl_ood_post")),
            # Variant: no-repair
            "ppl_w103_post_no_repair": _safe_round(r.get("ppl_w103_post_no_repair")),
            "prefill_speedup_no_repair": _safe_round(
                _safe_div(r.get("prefill_tps_no_repair"), baseline_prefill),
            ),
            "decode_speedup_no_repair": _safe_round(
                _safe_div(r.get("decode_tps_no_repair"), baseline_decode),
            ),
            "ppl_ood_post_no_repair": _safe_round(r.get("ppl_ood_post_no_repair")),
            # Variant: random-blocks
            "ppl_w103_post_random": _safe_round(r.get("ppl_w103_post_random")),
            "prefill_speedup_random": _safe_round(
                _safe_div(r.get("prefill_tps_random"), baseline_prefill),
            ),
            "decode_speedup_random": _safe_round(
                _safe_div(r.get("decode_tps_random"), baseline_decode),
            ),
            "ppl_ood_post_random": _safe_round(r.get("ppl_ood_post_random")),
            # Strike-gold columns (appended for compat)
            "selector_name": r.get("selector_name", "swiglu_mag"),
            "actual_repair_steps": r.get("actual_repair_steps", r["repair_steps"]),
            "early_stopped": r.get("early_stopped", False),
            "guardrail_failed": r.get("guardrail_failed", False),
            "prefill_tps_pre": _safe_round(r.get("prefill_tokens_per_sec_pre")),
            "prefill_tps_post": _safe_round(r.get("prefill_tokens_per_sec_post")),
            "decode_tps_pre": _safe_round(r.get("decode_tokens_per_sec_pre")),
            "decode_tps_post": _safe_round(r.get("decode_tokens_per_sec_post")),
            "prefill_speedup_pre_vs_base": _safe_round(
                _safe_div(r.get("prefill_tokens_per_sec_pre"), baseline_prefill),
            ),
            "prefill_speedup_post_vs_base": _safe_round(
                _safe_div(r.get("prefill_tokens_per_sec_post"), baseline_prefill),
            ),
            "decode_speedup_pre_vs_base": _safe_round(
                _safe_div(r.get("decode_tokens_per_sec_pre"), baseline_decode),
            ),
            "decode_speedup_post_vs_base": _safe_round(
                _safe_div(r.get("decode_tokens_per_sec_post"), baseline_decode),
            ),
            "staged": r.get("staged", False),
            "stages_completed": r.get("stages_completed", 0),
            # Geometry diagnostics (structural selector)
            "eff_rank95_D": _safe_round(r.get("eff_rank95_D")),
            "gini_k": _safe_round(r.get("gini_k")),
            "top10_edge_mass": _safe_round(r.get("top10_edge_mass")),
        })

    fieldnames = list(rows[0].keys())
    with open(RUNS_DIR / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    _write_json(RUNS_DIR / "summary.json", rows)

    log.info(f"Summary written to {RUNS_DIR / 'summary.csv'}")
    log.info(f"Summary written to {RUNS_DIR / 'summary.json'}")


# ── Strike-gold mode ──────────────────────────────────────────────────────────


def _run_strike_gold(
    args: argparse.Namespace,
    texts: Dict[str, List[str]],
    ood_texts: Optional[List[str]],
) -> None:
    """Focused run-plan designed to surface real speedups."""

    layer_sets = [
        ("even12", list(range(0, NUM_LAYERS, 2))),
        ("all24", list(range(NUM_LAYERS))),
    ]
    keep_fracs = [0.50, 0.35]
    grad_accums = [1, 4, 8]
    repair_steps = args.gold_repair_steps

    total_runs = 1 + len(layer_sets) * len(keep_fracs) * len(grad_accums)

    log.info("")
    log.info("=" * 70)
    log.info("[STRIKE-GOLD] Focused run-plan active")
    log.info(f"  Selector:       {args.selector}")
    log.info(f"  Layer sets:     {[d for d, _ in layer_sets]}")
    log.info(f"  Keep fracs:     {keep_fracs}")
    log.info(f"  Grad accum:     {grad_accums}")
    log.info(f"  Repair steps:   {repair_steps} (max)")
    log.info(f"  PPL guardrail:  {args.gold_ppl_guardrail}")
    log.info(f"  Early-stop:     patience={args.gold_early_stop_patience}")
    if args.staged:
        log.info(
            f"  Staged:         stage_size={args.stage_size}  "
            f"stage_repair_steps={args.stage_repair_steps}  "
            f"stage_guardrail={args.stage_guardrail}"
        )
    log.info(f"  Total runs:     {total_runs} (1 baseline + {total_runs - 1} compressed)")
    log.info("=" * 70)

    vllm_prompts = texts["eval_w2"][:128] if args.enable_vllm else None

    all_results: List[Dict[str, Any]] = []
    run_id = 0

    # ── Baseline ──────────────────────────────────────────────────────────────
    log.info("")
    log.info("STRIKE-GOLD: BASELINE")
    baseline = run_baseline(
        texts,
        ood_texts=ood_texts,
        max_eval_tokens_ood=args.max_eval_tokens_ood,
        enable_vllm=args.enable_vllm,
        vllm_prompts=vllm_prompts,
    )
    all_results.append(baseline)
    run_id += 1

    # ── Compressed runs grid ──────────────────────────────────────────────────
    for layer_desc, layers in layer_sets:
        for kf in keep_fracs:
            for ga in grad_accums:
                log.info("")
                log.info(
                    f"STRIKE-GOLD: {layer_desc} keep_frac={kf} "
                    f"grad_accum={ga}"
                )
                r = run_compressed(
                    run_id=run_id,
                    phase=99,  # strike-gold sentinel
                    layer_desc=layer_desc,
                    layers=layers,
                    keep_frac=kf,
                    repair_steps=repair_steps,
                    texts=texts,
                    selector=args.selector,
                    grad_accum_steps=ga,
                    curve_every=args.curve_every,
                    curve_eval_texts=args.curve_eval_texts,
                    curve_max_eval_tokens=args.curve_max_eval_tokens,
                    ood_texts=ood_texts,
                    max_eval_tokens_ood=args.max_eval_tokens_ood,
                    skip_ablations=args.skip_ablations,
                    enable_vllm=args.enable_vllm,
                    vllm_prompts=vllm_prompts,
                    save_artifacts=True,
                    ppl_guardrail=args.gold_ppl_guardrail,
                    early_stop_patience=args.gold_early_stop_patience,
                    staged=args.staged,
                    stage_size=args.stage_size,
                    stage_repair_steps=args.stage_repair_steps,
                    stage_guardrail=args.stage_guardrail,
                )
                all_results.append(r)
                run_id += 1

    write_summary(all_results, baseline)

    log.info("")
    log.info("=" * 70)
    log.info(f"STRIKE-GOLD COMPLETE: {len(all_results)} runs finished")
    log.info("=" * 70)


# ── Main ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-layer compression experiment harness for pnn_compiler.",
    )
    p.add_argument(
        "--grad-accum-steps", type=int, default=1,
        help="Gradient accumulation steps per optimizer update (default: 1).",
    )
    p.add_argument(
        "--curve-every", type=int, default=250,
        help="Evaluate repair curve every N optimizer steps (default: 250).",
    )
    p.add_argument(
        "--curve-eval-texts", type=int, default=50,
        help="Number of eval texts for repair curve checkpoints (default: 50).",
    )
    p.add_argument(
        "--curve-max-eval-tokens", type=int, default=5000,
        help="Max tokens budget per curve eval (default: 5000).",
    )
    p.add_argument(
        "--n-ood-texts", type=int, default=100,
        help="Number of OOD texts to load for evaluation (default: 100).",
    )
    p.add_argument(
        "--max-eval-tokens-ood", type=int, default=5000,
        help="Max tokens budget for OOD perplexity eval (default: 5000).",
    )
    p.add_argument(
        "--skip-ablations", action="store_true", default=False,
        help="Skip ablation baselines (no-repair, random-blocks) to save time.",
    )
    p.add_argument(
        "--enable-vllm", action="store_true", default=False,
        help="Run optional vLLM serving benchmark (requires vLLM installed).",
    )
    p.add_argument(
        "--smoke", action="store_true", default=False,
        help="Shrink text counts, eval budgets, and bench iters for a fast sanity check "
             "(~5 min on GPU). Does NOT change file defaults — overrides are in-memory only.",
    )
    p.add_argument(
        "--phases", type=str, default="0,1,2,3",
        help="Comma-separated phase numbers to run (default: 0,1,2,3). "
             "Phase 2 requires 1; phase 3 requires 2; phases >0 require 0 for summary.",
    )
    p.add_argument(
        "--outdir", type=str, default="runs",
        help="Output directory name relative to pnn_compiler/ (default: runs).",
    )
    # Selector
    p.add_argument(
        "--selector", type=str, default="swiglu_mag",
        choices=["swiglu_mag", "structural"],
        help="Block selection method: swiglu_mag (magnitude-based, default) "
             "or structural (coupling-geometry + Physarum conductance).",
    )
    # Strike-gold mode flags
    p.add_argument(
        "--strike-gold", action="store_true", default=False,
        help="Run focused strike-gold experiment plan instead of the phase matrix. "
             "Sweeps even12/all24 x keep_frac{0.50,0.35} x grad_accum{1,4,8} with "
             "early stopping and PPL guardrails. Saves importance/kept-indices artifacts.",
    )
    p.add_argument(
        "--gold-repair-steps", type=int, default=1000,
        help="Max repair optimizer steps per run in strike-gold mode (default: 1000).",
    )
    p.add_argument(
        "--gold-ppl-guardrail", type=float, default=200.0,
        help="Skip repair if pre-repair ppl_w103 exceeds this threshold (default: 200).",
    )
    p.add_argument(
        "--gold-early-stop-patience", type=int, default=3,
        help="Stop repair if ppl_w103 hasn't improved for N consecutive curve "
             "checkpoints (default: 3). 0 disables early stopping.",
    )
    # Staged compression flags
    p.add_argument(
        "--staged", action="store_true", default=False,
        help="Compress layers in stages (subset at a time with repair between) "
             "to avoid catastrophic PPL blowup when compressing many layers at once.",
    )
    p.add_argument(
        "--stage-size", type=int, default=6,
        help="Number of layers to compress per stage (default: 6).",
    )
    p.add_argument(
        "--stage-repair-steps", type=int, default=500,
        help="Max repair optimizer steps per stage (default: 500).",
    )
    p.add_argument(
        "--stage-guardrail", type=float, default=200.0,
        help="Abort staging if post-compile ppl_w103 exceeds this per-stage "
             "threshold (default: 200).",
    )
    return p.parse_args()


def main() -> None:
    global N_TEXTS_CAL, N_TEXTS_TRAIN, N_TEXTS_EVAL, MAX_EVAL_TOKENS
    global PREFILL_WARMUP_ITERS, PREFILL_ITERS, DECODE_WARMUP_ITERS, DECODE_ITERS
    global RUNS_DIR

    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── --smoke: shrink runtime constants in-memory ───────────────────────────
    if args.smoke:
        N_TEXTS_CAL = 5
        N_TEXTS_TRAIN = 10
        N_TEXTS_EVAL = 5
        MAX_EVAL_TOKENS = 500
        PREFILL_WARMUP_ITERS = 2
        PREFILL_ITERS = 5
        DECODE_WARMUP_ITERS = 1
        DECODE_ITERS = 2

    # ── --outdir: override output directory ───────────────────────────────────
    RUNS_DIR = Path(__file__).resolve().parents[1] / args.outdir

    # ── --phases: parse and validate dependencies ─────────────────────────────
    phases: set = set()
    for tok in args.phases.split(","):
        tok = tok.strip()
        if tok:
            phases.add(int(tok))

    if phases - {0} and 0 not in phases:
        raise SystemExit(
            "ERROR: phases >0 require phase 0 (baseline needed for summary speedup ratios)."
        )
    if 2 in phases and 1 not in phases:
        raise SystemExit(
            "ERROR: phase 2 requires phase 1 (layer-set sweep selects winner for "
            "keep_frac sweep)."
        )
    if 3 in phases and 2 not in phases:
        raise SystemExit(
            "ERROR: phase 3 requires phase 2 (keep_frac sweep selects winner for "
            "repair_steps sweep)."
        )

    # ── Config echo ───────────────────────────────────────────────────────────
    log.info("Multi-layer experiment harness starting")
    log.info(f"Device={DEVICE}  dtype={DTYPE}  seed={SEED}")
    log.info(f"Model={MODEL_ID}  block_size={BLOCK_SIZE}  max_len={MAX_LEN}")
    log.info(f"phases={sorted(phases)}  outdir={RUNS_DIR}")
    if args.smoke:
        log.info(
            f"[SMOKE MODE] N_TEXTS_CAL={N_TEXTS_CAL}  N_TEXTS_TRAIN={N_TEXTS_TRAIN}  "
            f"N_TEXTS_EVAL={N_TEXTS_EVAL}  MAX_EVAL_TOKENS={MAX_EVAL_TOKENS}"
        )
        log.info(
            f"[SMOKE MODE] PREFILL_WARMUP={PREFILL_WARMUP_ITERS}  "
            f"PREFILL_ITERS={PREFILL_ITERS}  "
            f"DECODE_WARMUP={DECODE_WARMUP_ITERS}  DECODE_ITERS={DECODE_ITERS}"
        )
    log.info(
        f"grad_accum_steps={args.grad_accum_steps}  "
        f"curve_every={args.curve_every}  "
        f"curve_eval_texts={args.curve_eval_texts}  "
        f"curve_max_eval_tokens={args.curve_max_eval_tokens}"
    )
    log.info(
        f"selector={args.selector}  "
        f"skip_ablations={args.skip_ablations}  "
        f"enable_vllm={args.enable_vllm}  "
        f"n_ood_texts={args.n_ood_texts}  "
        f"max_eval_tokens_ood={args.max_eval_tokens_ood}"
    )
    if args.strike_gold:
        log.info(
            f"[STRIKE-GOLD] repair_steps={args.gold_repair_steps}  "
            f"ppl_guardrail={args.gold_ppl_guardrail}  "
            f"early_stop_patience={args.gold_early_stop_patience}"
        )
    if args.staged:
        log.info(
            f"[STAGED] stage_size={args.stage_size}  "
            f"stage_repair_steps={args.stage_repair_steps}  "
            f"stage_guardrail={args.stage_guardrail}"
        )

    # ── Load data once ────────────────────────────────────────────────────────
    log.info("Loading datasets...")
    texts = load_text_sets(N_TEXTS_CAL, N_TEXTS_TRAIN, N_TEXTS_EVAL)
    log.info(
        f"  cal={len(texts['cal'])}  train={len(texts['train'])}  "
        f"eval_w2={len(texts['eval_w2'])}  eval_w103={len(texts['eval_w103'])}"
    )

    # OOD texts (graceful if unavailable)
    ood_texts: Optional[List[str]] = None
    if args.n_ood_texts > 0:
        try:
            ood_texts = load_ood_texts(args.n_ood_texts) or None
        except Exception as exc:
            log.warning("OOD dataset loading failed: %s", exc)
            ood_texts = None
    if ood_texts:
        log.info(f"  OOD texts loaded: {len(ood_texts)}")
    else:
        log.info("  OOD texts unavailable — OOD eval will be skipped")

    # ── Strike-gold mode: run focused plan and return ────────────────────────
    if args.strike_gold:
        _run_strike_gold(args, texts, ood_texts)
        return

    # Prompts for vLLM benchmark
    vllm_prompts = texts["eval_w2"][:128] if args.enable_vllm else None

    all_results: List[Dict[str, Any]] = []
    run_id = 0

    compressed_kwargs: Dict[str, Any] = {
        "selector": args.selector,
        "grad_accum_steps": args.grad_accum_steps,
        "curve_every": args.curve_every,
        "curve_eval_texts": args.curve_eval_texts,
        "curve_max_eval_tokens": args.curve_max_eval_tokens,
        "ood_texts": ood_texts,
        "max_eval_tokens_ood": args.max_eval_tokens_ood,
        "skip_ablations": args.skip_ablations,
        "enable_vllm": args.enable_vllm,
        "vllm_prompts": vllm_prompts,
    }

    # ── Phase 0: Baseline ─────────────────────────────────────────────────────
    baseline: Optional[Dict[str, Any]] = None

    if 0 in phases:
        log.info("")
        log.info("PHASE 0: BASELINE")
        baseline = run_baseline(
            texts,
            ood_texts=ood_texts,
            max_eval_tokens_ood=args.max_eval_tokens_ood,
            enable_vllm=args.enable_vllm,
            vllm_prompts=vllm_prompts,
        )
        all_results.append(baseline)
        run_id += 1

    # ── Phase 1: Layer-set sweep (keep_frac=0.50, repair_steps=2000) ──────────
    best_p1: Optional[Dict[str, Any]] = None

    if 1 in phases:
        log.info("")
        log.info("PHASE 1: LAYER-SET SWEEP (keep_frac=0.50)")

        layer_configs = [
            ("6layers", [0, 3, 6, 9, 12, 15]),
            ("even12", list(range(0, NUM_LAYERS, 2))),
            ("all24", list(range(NUM_LAYERS))),
        ]

        phase1_results: List[Dict[str, Any]] = []
        for desc, layers in layer_configs:
            r = run_compressed(
                run_id=run_id, phase=1, layer_desc=desc, layers=layers,
                keep_frac=0.50, repair_steps=2000, texts=texts,
                **compressed_kwargs,
            )
            phase1_results.append(r)
            all_results.append(r)
            run_id += 1

        best_p1 = min(phase1_results, key=lambda r: r["ppl_w2_test_post"])
        log.info(
            f"Phase 1 winner: {best_p1['layer_desc']}  "
            f"PPL w2={best_p1['ppl_w2_test_post']:.2f}"
        )

    # ── Phase 2: keep_frac sweep on best layer-set ────────────────────────────
    best_p2: Optional[Dict[str, Any]] = None

    if 2 in phases:
        assert best_p1 is not None  # guaranteed by dependency validation above
        log.info("")
        log.info(f"PHASE 2: KEEP_FRAC SWEEP (layers={best_p1['layer_desc']})")

        keep_fracs = [0.70, 0.50, 0.35, 0.25]
        phase2_results: List[Dict[str, Any]] = []
        for kf in keep_fracs:
            r = run_compressed(
                run_id=run_id, phase=2, layer_desc=best_p1["layer_desc"],
                layers=best_p1["layers"], keep_frac=kf, repair_steps=2000,
                texts=texts, **compressed_kwargs,
            )
            phase2_results.append(r)
            all_results.append(r)
            run_id += 1

        best_p2 = min(phase2_results, key=lambda r: r["ppl_w2_test_post"])
        log.info(
            f"Phase 2 winner: keep_frac={best_p2['keep_frac']}  "
            f"PPL w2={best_p2['ppl_w2_test_post']:.2f}"
        )

    # ── Phase 3: repair_steps sweep on best (layer-set + keep_frac) ───────────
    if 3 in phases:
        assert best_p2 is not None  # guaranteed by dependency validation above
        log.info("")
        log.info(
            f"PHASE 3: REPAIR_STEPS SWEEP "
            f"(layers={best_p2['layer_desc']}, keep_frac={best_p2['keep_frac']})"
        )

        step_values = [250, 500, 1000, 2000]
        for steps in step_values:
            r = run_compressed(
                run_id=run_id, phase=3, layer_desc=best_p2["layer_desc"],
                layers=best_p2["layers"], keep_frac=best_p2["keep_frac"],
                repair_steps=steps, texts=texts, **compressed_kwargs,
            )
            all_results.append(r)
            run_id += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    if baseline is not None and all_results:
        write_summary(all_results, baseline)

    log.info("")
    log.info("=" * 70)
    log.info(f"COMPLETE: {len(all_results)} runs finished")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
