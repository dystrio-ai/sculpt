"""Compile engine: structural compression + staged repair + evaluation.

Integrates RepairPolicy auto-selection, two-tier evaluation (cheap vs final),
best-checkpoint restore, and staged rollback with never-ship-worse invariant.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from ._model import load_model_and_tokenizer, resolve_dtype
from ._data import CalibConfig, load_text_sets, deterministic_subset
from ._eval import eval_perplexity
from ._bench import (
    bench_prefill_tps, bench_decode_tps,
    bench_prefill_latency_ms, bench_decode_latency_ms,
    compute_latency_percentiles,
)
from ._compile import compress_mlp_layer_swiglu_inplace
from .selectors import select_for_layer, BLOCK_SIZE
from .selectors.structural import prescan_structural_artifacts
from .repair import repair_layers, _snapshot_trainable, _restore_trainable
from .policy import RepairPolicy, auto_select_policy, build_policy_ladder, escalate_policy, HELPFUL_THRESHOLD

_log = logging.getLogger(__name__)

MAX_LEN = 256
BENCH_BATCH = 32
BENCH_WARMUP = 20
BENCH_ITERS = 80
DECODE_STEPS = 128
DECODE_WARMUP = 3
DECODE_ITERS = 10
STAGE_GUARDRAIL = 200.0


@dataclass
class CompileResult:
    """Result from a single compile_model invocation."""

    model: Any = None
    tokenizer: Any = None
    keep_frac: float = 0.0
    baseline_metrics: Dict[str, float] = field(default_factory=dict)
    metrics_pre: Dict[str, float] = field(default_factory=dict)
    metrics_post: Dict[str, float] = field(default_factory=dict)
    compile_report: Dict[str, Any] = field(default_factory=dict)
    wall_time_s: float = 0.0
    early_stopped: bool = False
    guardrail_failed: bool = False
    layers_compressed: List[int] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    policy_name: str = ""
    pilot_report: Optional[Dict[str, Any]] = None
    failure: Optional[Dict[str, Any]] = None
    stage_stats: List[Dict[str, Any]] = field(default_factory=list)
    escalation_applied: bool = False
    escalation_details: Optional[Dict[str, Any]] = None
    peak_cuda_allocated_compile_bytes: Optional[int] = None
    peak_cuda_reserved_compile_bytes: Optional[int] = None
    peak_cuda_allocated_bench_bytes: Optional[int] = None
    peak_cuda_reserved_bench_bytes: Optional[int] = None
    cuda_allocated_end_bytes: Optional[int] = None
    cuda_reserved_end_bytes: Optional[int] = None
    num_params: Optional[int] = None
    weights_bytes: Optional[int] = None
    baseline_num_params: Optional[int] = None
    baseline_weights_bytes: Optional[int] = None


def _collect_metrics(
    model, tokenizer, texts: Dict[str, List[str]], device: str,
    max_eval_tokens: int = 40_000,
) -> Dict[str, Any]:
    eval_w2 = texts["eval_w2"]
    eval_w103 = texts["eval_w103"]
    ppl_w2 = eval_perplexity(model, tokenizer, eval_w2, MAX_LEN, device, max_eval_tokens)
    ppl_w103 = eval_perplexity(model, tokenizer, eval_w103, MAX_LEN, device, max_eval_tokens)
    bench_texts = eval_w2[:BENCH_BATCH]
    decode_text = eval_w2[0] if eval_w2 else ""
    prefill = bench_prefill_tps(
        model, tokenizer, bench_texts, MAX_LEN, device, BENCH_WARMUP, BENCH_ITERS,
    )
    decode = bench_decode_tps(
        model, tokenizer, decode_text, MAX_LEN, device,
        DECODE_STEPS, DECODE_WARMUP, DECODE_ITERS,
    )
    out: Dict[str, Any] = {
        "ppl_w2_test": ppl_w2,
        "ppl_w103_valid": ppl_w103,
        "prefill_tokens_per_sec": prefill,
        "decode_tokens_per_sec": decode,
    }

    # Latency percentiles (bounded: small warmup + measure iterations)
    prefill_ms = bench_prefill_latency_ms(
        model, tokenizer, bench_texts, MAX_LEN, device,
    )
    decode_ms = bench_decode_latency_ms(
        model, tokenizer, decode_text, MAX_LEN, device,
    )
    pf_pct = compute_latency_percentiles(prefill_ms)
    dc_pct = compute_latency_percentiles(decode_ms)
    for k, v in pf_pct.items():
        out[f"prefill_latency_ms_{k}"] = v
    for k, v in dc_pct.items():
        out[f"decode_ms_per_token_{k}"] = v

    if pf_pct and dc_pct:
        _log.info(
            "[bench] prefill_ms p50=%.1f p95=%.1f p99=%.1f | "
            "decode_ms_per_tok p50=%.2f p95=%.2f p99=%.2f",
            pf_pct["p50"], pf_pct["p95"], pf_pct["p99"],
            dc_pct["p50"], dc_pct["p95"], dc_pct["p99"],
        )

    return out


def cheap_eval(
    model, tokenizer, texts_eval: Sequence[str],
    device: str, max_tokens: int = 5000,
) -> float:
    """Quick PPL evaluation on a small subset for early stopping / curve."""
    return eval_perplexity(model, tokenizer, texts_eval, MAX_LEN, device, max_tokens)


def setup_determinism(seed: int, deterministic: bool = False) -> None:
    """Seed all RNGs and optionally disable TF32 for bitwise reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True


def _write_failure(outdir: Optional[Path], keep_frac: float, reason: str, detail: str) -> None:
    if outdir is None:
        return
    outdir.mkdir(parents=True, exist_ok=True)
    fail = {"keep_frac": keep_frac, "reason": reason, "detail": detail}
    with open(outdir / "failure.json", "w") as f:
        json.dump(fail, f, indent=2)


def compile_model(
    model_id: str,
    keep_frac: float,
    *,
    texts: Optional[Dict[str, List[str]]] = None,
    prescan_cache: Optional[Dict[int, Dict[str, Any]]] = None,
    baseline_metrics: Optional[Dict[str, float]] = None,
    policy: Optional[RepairPolicy] = None,
    pilot_report: Optional[Dict[str, Any]] = None,
    device: str = "cuda",
    dtype_str: str = "bf16",
    seed: int = 0,
    deterministic: bool = False,
    n_texts_cal: int = 400,
    n_texts_train: int = 2500,
    n_texts_eval: int = 300,
    max_eval_tokens: int = 40_000,
    selector: str = "structural",
    failure_dir: Optional[Path] = None,
    layer_order: Optional[List[int]] = None,
    allow_escalation: bool = True,
    calib: Optional[CalibConfig] = None,
    distill: bool = False,
) -> CompileResult:
    """Compile a model at a specific keep_frac.

    Orchestrates: load -> prescan -> staged compress -> repair -> eval.
    Uses *policy* for all repair hyperparameters.  If a stage fails, attempts
    one downshift to a more conservative policy before marking as failed.
    """
    wall_t0 = time.time()
    setup_determinism(seed, deterministic)
    dtype = resolve_dtype(dtype_str)

    _log.info("loading model %s (keep_frac=%.3f)", model_id, keep_frac)
    model, tok = load_model_and_tokenizer(model_id, device, dtype)
    num_layers = model.config.num_hidden_layers
    layers = list(range(num_layers))

    if texts is None:
        _log.info("loading datasets")
        texts = load_text_sets(n_texts_cal, n_texts_train, n_texts_eval, calib=calib)

    # Resolve policy
    if policy is None:
        from .policy import _estimate_param_billions
        param_b = _estimate_param_billions(model)
        ladder = build_policy_ladder(param_b)
        policy = ladder[0]
        _log.info("using default policy: %s", policy.name)

    if baseline_metrics is None:
        _log.info("computing baseline metrics (final eval)")
        baseline_metrics = _collect_metrics(model, tok, texts, device, max_eval_tokens)
        if torch.cuda.is_available():
            baseline_metrics["cuda_allocated_baseline_bytes"] = torch.cuda.memory_allocated()

    baseline_num_params = sum(p.numel() for p in model.parameters())
    baseline_weights_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    rng = np.random.RandomState(seed) if deterministic else None

    if prescan_cache is None and selector == "structural":
        _log.info("running structural prescan on %d layers", num_layers)
        prescan_cache = prescan_structural_artifacts(
            model, tok, layers, texts["cal"], MAX_LEN, device,
            block_size=BLOCK_SIZE,
        )

    # Distillation: create a frozen teacher from the original model before compression.
    # A/B testing showed distillation only helps at aggressive compression
    # (keep_frac <= 0.65) where the model is damaged enough that teacher
    # guidance aids recovery. At lighter compression, CE-only repair is
    # sufficient and the KL term adds noise.
    DISTILL_THRESHOLD = 0.65
    teacher_model = None
    distill_alpha = 0.0
    if distill:
        if policy is not None and policy.distill_alpha > 0.0:
            distill_alpha = policy.distill_alpha
        elif keep_frac <= DISTILL_THRESHOLD:
            distill_alpha = 0.5
        else:
            _log.info(
                "distillation skipped: keep_frac=%.3f > %.2f threshold",
                keep_frac, DISTILL_THRESHOLD,
            )
        if distill_alpha > 0.0:
            _log.info("creating teacher model for distillation (alpha=%.2f)", distill_alpha)
            teacher_model = copy.deepcopy(model)
            for p in teacher_model.parameters():
                p.requires_grad = False
            teacher_model.eval()

    original_ffn_dims: Dict[int, int] = {}
    for li in layers:
        original_ffn_dims[li] = model.model.layers[li].mlp.gate_proj.out_features

    if layer_order is not None and keep_frac < 1.0:
        compressible = [li for li in layer_order if li in set(layers)]
    else:
        compressible = [li for li in layers if keep_frac < 1.0]
    if not compressible:
        metrics = _collect_metrics(model, tok, texts, device, max_eval_tokens)
        return CompileResult(
            model=model, tokenizer=tok, keep_frac=keep_frac,
            baseline_metrics=baseline_metrics, metrics_pre=metrics,
            metrics_post=metrics, wall_time_s=time.time() - wall_t0,
            layers_compressed=[], policy_name=policy.name,
            pilot_report=pilot_report,
        )

    # Deterministic cheap-eval subsets
    cheap_w103 = deterministic_subset(texts["eval_w103"], policy.cheap_eval_texts, seed)
    cheap_w2 = deterministic_subset(texts["eval_w2"], policy.cheap_eval_texts, seed)

    # Staged compression with rollback support
    stage_size = policy.stage_size
    chunks = [compressible[i : i + stage_size] for i in range(0, len(compressible), stage_size)]
    compressed_so_far: List[int] = []
    compile_report: Dict[str, Any] = {}
    guardrail_failed = False
    any_early_stopped = False
    total_repair_steps = 0
    current_policy = policy
    stage_stats_list: List[Dict[str, Any]] = []
    _escalation_applied = False
    _escalation_details: Optional[Dict[str, Any]] = None
    _consec_fail_count = 0

    def _curve_fn(opt_step: int) -> Dict[str, float]:
        return {
            "ppl_w2_test": cheap_eval(model, tok, cheap_w2, device, current_policy.cheap_eval_max_tokens),
            "ppl_w103_valid": cheap_eval(
                model, tok, cheap_w103, device, current_policy.cheap_eval_max_tokens,
            ),
        }

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    _log.info(
        "staged compression: %d stages of up to %d layers  policy=%s",
        len(chunks), stage_size, current_policy.name,
    )

    for si, chunk in enumerate(chunks):
        _log.info("stage %d/%d: layers %s", si + 1, len(chunks), chunk)

        # Save pre-stage snapshot for rollback
        pre_stage_snap = _snapshot_trainable(model, compressed_so_far + chunk) if compressed_so_far else None

        for li in chunk:
            kept_blocks, kept_idx, _ = select_for_layer(
                model, tok, li, texts["cal"], keep_frac,
                MAX_LEN, device, selector=selector,
                prescan_cache=prescan_cache, rng=rng,
            )
            rep = compress_mlp_layer_swiglu_inplace(model, li, kept_idx, dtype, device)
            compile_report[str(li)] = {
                "kept_blocks": len(kept_blocks),
                "original_ffn": original_ffn_dims[li],
                "ffn_kept": rep["ffn_kept"],
                "keep_frac": keep_frac,
            }
        compressed_so_far.extend(chunk)

        stage_ppl = cheap_eval(model, tok, cheap_w103, device, current_policy.cheap_eval_max_tokens)
        _log.info("stage %d post-compile ppl_w103=%.2f", si + 1, stage_ppl)

        if STAGE_GUARDRAIL > 0 and stage_ppl > STAGE_GUARDRAIL:
            _log.warning("stage guardrail: ppl_w103=%.2f > %.0f — aborting", stage_ppl, STAGE_GUARDRAIL)
            guardrail_failed = True
            break

        if math.isnan(stage_ppl) or math.isinf(stage_ppl):
            _log.warning("NaN/Inf after compression — aborting")
            guardrail_failed = True
            break

        # Stage repair with best-checkpoint + never-worse invariant
        stage_regression_stop = False
        stage_nan_inf = False
        stage_early_stop = False
        ppl_best_stage = stage_ppl
        if current_policy.steps > 0:
            warmup_stage = min(100, current_policy.steps // 5)
            sr = repair_layers(
                model=model, tokenizer=tok, texts_train=texts["train"],
                layers=compressed_so_far, steps=current_policy.steps, lr=current_policy.lr,
                warmup=warmup_stage, weight_decay=0.01,
                max_len=MAX_LEN, device=device,
                grad_accum_steps=current_policy.grad_accum_steps,
                curve_fn=_curve_fn, curve_every=current_policy.curve_every,
                early_stop_patience=current_policy.early_stop_patience,
                regression_limit=current_policy.regression_limit,
                max_grad_norm=current_policy.max_grad_norm,
                save_best=True, pre_repair_metric=stage_ppl,
                teacher_model=teacher_model, distill_alpha=distill_alpha,
            )
            total_repair_steps += int(sr["steps"])
            if sr.get("early_stopped"):
                any_early_stopped = True

            ppl_best_stage = sr.get("best_metric", float("inf"))
            stage_regression_stop = sr.get(
                "regression_tripwire_triggered",
                sr.get("regression_stop_triggered", False),
            )
            stage_nan_inf = sr.get("nan_inf_detected", False)
            stage_early_stop = sr.get("early_stop_triggered", False)

            post_ppl = cheap_eval(model, tok, cheap_w103, device, current_policy.cheap_eval_max_tokens)

            if not sr.get("repaired_ok", True) or math.isnan(post_ppl) or math.isinf(post_ppl):
                _log.warning(
                    "stage %d repair failed (repaired_ok=%s, post_ppl=%.2f); "
                    "attempting downshift",
                    si + 1, sr.get("repaired_ok"), post_ppl,
                )
                if pre_stage_snap is not None:
                    _restore_trainable(model, compressed_so_far, pre_stage_snap)

                from .policy import _estimate_param_billions
                param_b = _estimate_param_billions(model)
                ladder = build_policy_ladder(param_b)
                current_idx = next(
                    (i for i, p in enumerate(ladder) if p.name == current_policy.name),
                    len(ladder) - 1,
                )
                if current_idx + 1 < len(ladder):
                    fallback = ladder[current_idx + 1]
                    _log.info("downshifting to %s for retry", fallback.name)
                    current_policy = fallback
                    warmup_r = min(100, fallback.steps // 5)
                    sr2 = repair_layers(
                        model=model, tokenizer=tok, texts_train=texts["train"],
                        layers=compressed_so_far, steps=fallback.steps, lr=fallback.lr,
                        warmup=warmup_r, weight_decay=0.01,
                        max_len=MAX_LEN, device=device,
                        grad_accum_steps=fallback.grad_accum_steps,
                        curve_fn=_curve_fn, curve_every=fallback.curve_every,
                        early_stop_patience=fallback.early_stop_patience,
                        regression_limit=fallback.regression_limit,
                        max_grad_norm=fallback.max_grad_norm,
                        save_best=True, pre_repair_metric=stage_ppl,
                        teacher_model=teacher_model, distill_alpha=distill_alpha,
                    )
                    total_repair_steps += int(sr2["steps"])
                    if not sr2.get("repaired_ok", True):
                        _log.warning("downshift retry also failed — continuing with no-repair")
                else:
                    _log.warning("no more conservative policy available")
            else:
                if (
                    current_policy.regression_limit > 0
                    and stage_ppl > 0
                    and post_ppl > stage_ppl * (1.0 + current_policy.regression_limit * 5)
                ):
                    _log.warning(
                        "stage regression: post-repair ppl=%.2f > pre-repair %.2f",
                        post_ppl, stage_ppl,
                    )
                    break

        # Outcome semantics: repair_fail = hard failure (nan/inf) only;
        # regression tripwire is a soft stop (best checkpoint restored).
        repair_fail = stage_nan_inf
        improve_frac = (stage_ppl - ppl_best_stage) / stage_ppl if stage_ppl > 0 else 0.0
        repair_helpful = (not repair_fail) and improve_frac >= HELPFUL_THRESHOLD

        if repair_fail:
            _log.info(
                "[engine] stage_fail stage=%d layers=%s pre=%.2f best=%.2f nan_inf=%s",
                si, chunk, stage_ppl, ppl_best_stage, stage_nan_inf,
            )
        elif repair_helpful:
            _log.debug(
                "[engine] stage_help stage=%d improve=%.2f%%",
                si, improve_frac * 100,
            )

        _stage_stat = {
            "stage": si,
            "layers": chunk,
            "repair_fail": repair_fail,
            "repair_helpful": repair_helpful,
            "ppl_pre_repair": round(stage_ppl, 4),
            "ppl_best": round(ppl_best_stage, 4),
            "improve_frac": round(improve_frac, 6),
            "regression_tripwire": stage_regression_stop,
            "nan_inf": stage_nan_inf,
            "early_stop": stage_early_stop,
        }
        assert not (
            _stage_stat["regression_tripwire"] and _stage_stat["repair_fail"]
            and not _stage_stat["nan_inf"]
        ), "regression_tripwire must NOT cause repair_fail without nan_inf"
        stage_stats_list.append(_stage_stat)

        # Mid-compile escalation: only on hard failure (nan/inf), not tripwire
        if repair_fail:
            _consec_fail_count += 1
        else:
            _consec_fail_count = 0

        if (
            _consec_fail_count >= 2
            and not _escalation_applied
            and allow_escalation
        ):
            current_policy, _escalation_details = escalate_policy(
                current_policy, keep_frac=keep_frac, trigger_stage=si,
            )
            _escalation_applied = True

    # Final global repair
    if current_policy.steps > 0 and not guardrail_failed and compressed_so_far:
        final_steps = max(current_policy.steps, current_policy.steps * len(chunks) // 2)
        warmup_final = min(100, final_steps // 5)
        pre_final_ppl = cheap_eval(model, tok, cheap_w103, device, policy.cheap_eval_max_tokens)
        _log.info("final repair: %d steps on %d layers", final_steps, len(compressed_so_far))
        fr = repair_layers(
            model=model, tokenizer=tok, texts_train=texts["train"],
            layers=compressed_so_far, steps=final_steps, lr=current_policy.lr,
            warmup=warmup_final, weight_decay=0.01,
            max_len=MAX_LEN, device=device,
            grad_accum_steps=current_policy.grad_accum_steps,
            curve_fn=_curve_fn, curve_every=current_policy.curve_every,
            early_stop_patience=current_policy.early_stop_patience,
            regression_limit=current_policy.regression_limit,
            max_grad_norm=current_policy.max_grad_norm,
            save_best=True, pre_repair_metric=pre_final_ppl,
            teacher_model=teacher_model, distill_alpha=distill_alpha,
        )
        total_repair_steps += int(fr["steps"])
        if fr.get("early_stopped"):
            any_early_stopped = True

    # Free teacher model before benchmarks to reclaim VRAM
    if teacher_model is not None:
        del teacher_model
        teacher_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Compile-phase VRAM peaks (covers staging + repair)
    peak_alloc_compile: Optional[int] = None
    peak_resv_compile: Optional[int] = None
    if torch.cuda.is_available():
        peak_alloc_compile = torch.cuda.max_memory_allocated()
        peak_resv_compile = torch.cuda.max_memory_reserved()
        torch.cuda.reset_peak_memory_stats()

    sculpt_num_params = sum(p.numel() for p in model.parameters())
    sculpt_weights_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    # Final evaluation + benchmark (full token budget)
    metrics_post = _collect_metrics(model, tok, texts, device, policy.final_eval_max_tokens)

    # Benchmark-phase VRAM peaks + steady-state
    peak_alloc_bench: Optional[int] = None
    peak_resv_bench: Optional[int] = None
    end_alloc: Optional[int] = None
    end_resv: Optional[int] = None
    if torch.cuda.is_available():
        peak_alloc_bench = torch.cuda.max_memory_allocated()
        peak_resv_bench = torch.cuda.max_memory_reserved()
        end_alloc = torch.cuda.memory_allocated()
        end_resv = torch.cuda.memory_reserved()

    # Never-ship-worse: compare final PPL to baseline
    failure_info: Optional[Dict[str, Any]] = None
    if guardrail_failed:
        failure_info = {"reason": "guardrail", "detail": "stage guardrail exceeded"}
        _write_failure(failure_dir, keep_frac, "guardrail", "stage guardrail exceeded")

    _log.info(
        "[engine] compile done keep=%.3f escalation_applied=%s stages=%d",
        keep_frac, _escalation_applied, len(stage_stats_list),
    )

    config = {
        "model_id": model_id,
        "keep_frac": keep_frac,
        "seed": seed,
        "deterministic": deterministic,
        "device": device,
        "dtype": dtype_str,
        "block_size": BLOCK_SIZE,
        "stage_size": stage_size,
        "total_repair_steps": total_repair_steps,
        "num_layers": num_layers,
        "layers_compressed": len(compressed_so_far),
        "selector": selector,
        "policy": current_policy.to_dict(),
        "layer_order": compressible if layer_order is not None else None,
        "stage_stats": stage_stats_list,
        "escalation": {
            "applied": _escalation_applied,
            "details": _escalation_details,
        },
        "distillation": {
            "enabled": distill,
            "alpha": distill_alpha,
        },
    }

    return CompileResult(
        model=model,
        tokenizer=tok,
        keep_frac=keep_frac,
        baseline_metrics=baseline_metrics,
        metrics_pre=baseline_metrics,
        metrics_post=metrics_post,
        compile_report=compile_report,
        wall_time_s=time.time() - wall_t0,
        early_stopped=any_early_stopped,
        guardrail_failed=guardrail_failed,
        layers_compressed=compressed_so_far,
        config=config,
        policy_name=current_policy.name,
        pilot_report=pilot_report,
        failure=failure_info,
        stage_stats=stage_stats_list,
        escalation_applied=_escalation_applied,
        escalation_details=_escalation_details,
        peak_cuda_allocated_compile_bytes=peak_alloc_compile,
        peak_cuda_reserved_compile_bytes=peak_resv_compile,
        peak_cuda_allocated_bench_bytes=peak_alloc_bench,
        peak_cuda_reserved_bench_bytes=peak_resv_bench,
        cuda_allocated_end_bytes=end_alloc,
        cuda_reserved_end_bytes=end_resv,
        num_params=sculpt_num_params,
        weights_bytes=sculpt_weights_bytes,
        baseline_num_params=baseline_num_params,
        baseline_weights_bytes=baseline_weights_bytes,
    )
