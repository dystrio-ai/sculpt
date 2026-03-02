"""Repair policy ladder, pilot tuner, and triggered escalation."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from ._model import load_model_and_tokenizer, resolve_dtype
from ._eval import eval_perplexity
from ._compile import compress_mlp_layer_swiglu_inplace
from .selectors import select_for_layer, BLOCK_SIZE

_log = logging.getLogger(__name__)

PILOT_KEEP_FRAC = 0.85
MAX_LEN = 256
SPIKE_LAMBDA = 6.0
MIN_SLOPE_THRESHOLD = 1e-4


# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RepairPolicy:
    """Immutable repair hyperparameter bundle."""

    name: str
    stage_size: int
    lr: float
    steps: int
    early_stop_patience: int
    regression_limit: float
    curve_every: int
    cheap_eval_texts: int
    cheap_eval_max_tokens: int
    final_eval_max_tokens: int
    grad_accum_steps: int
    max_grad_norm: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "stage_size": self.stage_size,
            "lr": self.lr,
            "steps": self.steps,
            "early_stop_patience": self.early_stop_patience,
            "regression_limit": self.regression_limit,
            "curve_every": self.curve_every,
            "cheap_eval_texts": self.cheap_eval_texts,
            "cheap_eval_max_tokens": self.cheap_eval_max_tokens,
            "final_eval_max_tokens": self.final_eval_max_tokens,
            "grad_accum_steps": self.grad_accum_steps,
            "max_grad_norm": self.max_grad_norm,
        }


@dataclass
class PilotResult:
    policy_name: str
    P0: float
    Pt: float
    Pmax: float
    steps_run: int
    elapsed_s: float
    regression_stop: bool
    nan_inf: bool
    slope_per_s: float
    score: float
    helpful: bool
    stable: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_name": self.policy_name,
            "P0": round(self.P0, 4),
            "Pt": round(self.Pt, 4),
            "Pmax": round(self.Pmax, 4),
            "steps_run": self.steps_run,
            "elapsed_s": round(self.elapsed_s, 2),
            "regression_stop": self.regression_stop,
            "nan_inf": self.nan_inf,
            "slope_per_s": round(self.slope_per_s, 8),
            "score": round(self.score, 8),
            "helpful": self.helpful,
            "stable": self.stable,
        }


@dataclass
class TuningReport:
    pilot_keep_frac: float
    pilot_budget_s: float
    candidates: List[str]
    results: List[Dict[str, Any]]
    chosen_policy: str
    chosen_reason: str
    risk_score: float
    pilot_enabled: bool = True
    ladder_start_idx: int = 0
    escalation_applied: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pilot_enabled": self.pilot_enabled,
            "pilot_keep_frac": self.pilot_keep_frac,
            "pilot_budget_s": round(self.pilot_budget_s, 1),
            "candidates": self.candidates,
            "results": self.results,
            "chosen_policy": self.chosen_policy,
            "chosen_reason": self.chosen_reason,
            "risk_score": round(self.risk_score, 4),
            "ladder_start_idx": self.ladder_start_idx,
        }


# ── Helpers ───────────────────────────────────────────────────────────────────


def _estimate_param_billions(model) -> float:
    total = sum(p.numel() for p in model.parameters())
    return total / 1e9


def _scale_steps(base_steps: int, param_b: float) -> int:
    if param_b <= 1.5:
        return base_steps
    scale = min(3.0, 1.0 + (param_b - 1.0) / 6.0)
    return int(base_steps * scale)


def _scale_patience(base_patience: int, param_b: float) -> int:
    if param_b >= 5.0:
        return max(base_patience, int(base_patience * 1.5))
    return base_patience


def build_policy_ladder(param_b: float) -> List[RepairPolicy]:
    """Return the ordered policy ladder (most aggressive -> most conservative)."""
    base_steps = 450
    steps = _scale_steps(base_steps, param_b)
    cheap_texts = min(128, max(32, int(64 * (1.0 + param_b))))
    cheap_tokens = 5000
    final_tokens = 40_000

    return [
        RepairPolicy(
            name=f"ss4_lr1e-4_p8_s{steps}",
            stage_size=4, lr=1e-4, steps=steps,
            early_stop_patience=_scale_patience(8, param_b),
            regression_limit=0.02, curve_every=50,
            cheap_eval_texts=cheap_texts, cheap_eval_max_tokens=cheap_tokens,
            final_eval_max_tokens=final_tokens, grad_accum_steps=1,
        ),
        RepairPolicy(
            name=f"ss4_lr5e-5_p8_s{steps}",
            stage_size=4, lr=5e-5, steps=steps,
            early_stop_patience=_scale_patience(8, param_b),
            regression_limit=0.02, curve_every=50,
            cheap_eval_texts=cheap_texts, cheap_eval_max_tokens=cheap_tokens,
            final_eval_max_tokens=final_tokens, grad_accum_steps=1,
        ),
        RepairPolicy(
            name=f"ss2_lr5e-5_p8_s{steps}",
            stage_size=2, lr=5e-5, steps=steps,
            early_stop_patience=_scale_patience(8, param_b),
            regression_limit=0.02, curve_every=50,
            cheap_eval_texts=cheap_texts, cheap_eval_max_tokens=cheap_tokens,
            final_eval_max_tokens=final_tokens, grad_accum_steps=1,
        ),
        RepairPolicy(
            name=f"ss1_lr5e-5_p12_s{steps}",
            stage_size=1, lr=5e-5, steps=steps,
            early_stop_patience=_scale_patience(12, param_b),
            regression_limit=0.02, curve_every=50,
            cheap_eval_texts=cheap_texts, cheap_eval_max_tokens=cheap_tokens,
            final_eval_max_tokens=final_tokens, grad_accum_steps=1,
        ),
    ]


def risk_scale_policy(policy: RepairPolicy, risk: float) -> RepairPolicy:
    """Return a copy of *policy* with steps/eval budget scaled by risk."""
    if risk < 0.55:
        return policy
    scale = 1.0 + (risk - 0.55) * 0.6
    return RepairPolicy(
        name=policy.name,
        stage_size=policy.stage_size,
        lr=policy.lr,
        steps=int(policy.steps * scale),
        early_stop_patience=policy.early_stop_patience,
        regression_limit=policy.regression_limit,
        curve_every=policy.curve_every + (25 if risk >= 0.65 else 0),
        cheap_eval_texts=min(256, policy.cheap_eval_texts + (64 if risk >= 0.65 else 0)),
        cheap_eval_max_tokens=policy.cheap_eval_max_tokens,
        final_eval_max_tokens=policy.final_eval_max_tokens,
        grad_accum_steps=policy.grad_accum_steps,
        max_grad_norm=policy.max_grad_norm,
    )


# ── Pilot scoring (A3) ───────────────────────────────────────────────────────


def _score_pilot(
    P0: float,
    Pt: float,
    Pmax: float,
    elapsed_s: float,
    regression_stop: bool,
    nan_inf: bool,
) -> tuple:
    """Compute (slope_per_s, score, stable, helpful) per spec A3.

    Returns a 4-tuple matching PilotResult fields.
    """
    if nan_inf or regression_stop:
        return (0.0, -1e9, False, False)

    if P0 <= 0 or Pt <= 0 or Pmax <= 0:
        return (0.0, -1e9, False, False)

    delta = math.log(P0) - math.log(Pt)
    slope = delta / max(1e-6, elapsed_s)

    epsilon = 0.05
    spike_raw = math.log(Pmax) - math.log(P0) - math.log(1.0 + epsilon)
    spike = max(0.0, spike_raw)
    penalty = SPIKE_LAMBDA * spike

    score = slope - penalty
    stable = True
    helpful = slope >= MIN_SLOPE_THRESHOLD
    return (slope, score, stable, helpful)


# ── Candidate selection (A2) ─────────────────────────────────────────────────


def _pilot_candidates(
    ladder: List[RepairPolicy], risk_score: float,
) -> List[RepairPolicy]:
    """Return <=3 unique candidates from the ladder based on risk."""
    n = len(ladder)
    if risk_score <= 0.35:
        indices = [0, 1, 2]
    elif risk_score >= 0.65:
        indices = [2, 3] if n > 3 else [min(2, n - 1)]
    else:
        indices = [1, 2]

    indices = [i for i in indices if i < n]
    seen = set()
    deduped = []
    for i in indices:
        if ladder[i].name not in seen:
            seen.add(ladder[i].name)
            deduped.append(ladder[i])
    return deduped[:3]


# ── Budget (A6/G) ────────────────────────────────────────────────────────────


def _compute_pilot_budget(max_compile_hours: Optional[float]) -> float:
    if max_compile_hours is not None:
        total_s = max_compile_hours * 3600
        return min(0.10 * total_s, 480.0)
    return 240.0


# ── Pilot repair runner ──────────────────────────────────────────────────────


def run_repair_pilot(
    model,
    tokenizer,
    texts_train: Sequence[str],
    layers: Sequence[int],
    eval_subset: Sequence[str],
    policy: RepairPolicy,
    pilot_steps: int,
    pilot_budget_s: float,
    device: str,
    deterministic: bool,
    seed: int,
) -> PilotResult:
    """Run a bounded pilot repair and return scoring metrics.

    Uses the existing repair_layers path with best-checkpoint restore and
    never-worse invariant.
    """
    from .repair import repair_layers

    eval_tokens = min(3000, policy.cheap_eval_max_tokens)

    P0 = eval_perplexity(model, tokenizer, eval_subset, MAX_LEN, device, eval_tokens)
    if math.isnan(P0) or math.isinf(P0):
        return PilotResult(
            policy_name=policy.name, P0=P0, Pt=P0, Pmax=P0,
            steps_run=0, elapsed_s=0.0, regression_stop=False, nan_inf=True,
            slope_per_s=0.0, score=-1e9, helpful=False, stable=False,
        )

    ppl_values = [P0]

    def curve_fn(step: int) -> Dict[str, float]:
        val = eval_perplexity(model, tokenizer, eval_subset, MAX_LEN, device, eval_tokens)
        ppl_values.append(val)
        return {"ppl_w103_valid": val}

    t0 = time.time()
    warmup = min(15, pilot_steps // 5)
    result = repair_layers(
        model=model, tokenizer=tokenizer, texts_train=texts_train,
        layers=list(layers), steps=pilot_steps, lr=policy.lr,
        warmup=warmup, weight_decay=0.01,
        max_len=MAX_LEN, device=device,
        grad_accum_steps=policy.grad_accum_steps,
        curve_fn=curve_fn, curve_every=max(10, pilot_steps // 5),
        early_stop_patience=policy.early_stop_patience,
        regression_limit=policy.regression_limit,
        save_best=True, pre_repair_metric=P0,
    )
    elapsed = time.time() - t0

    steps_run = int(result["steps"])
    regression_stop = result.get("early_stopped", False) and not result.get("repaired_ok", True)

    Pt = result.get("best_metric", P0)
    if math.isnan(Pt) or math.isinf(Pt):
        Pt = P0

    valid_ppls = [v for v in ppl_values if not (math.isnan(v) or math.isinf(v))]
    Pmax = max(valid_ppls) if valid_ppls else P0

    nan_inf = any(math.isnan(v) or math.isinf(v) for v in ppl_values)

    slope, score, stable, helpful = _score_pilot(P0, Pt, Pmax, elapsed, regression_stop, nan_inf)

    return PilotResult(
        policy_name=policy.name, P0=P0, Pt=Pt, Pmax=Pmax,
        steps_run=steps_run, elapsed_s=elapsed,
        regression_stop=regression_stop, nan_inf=nan_inf,
        slope_per_s=slope, score=score, helpful=helpful, stable=stable,
    )


# ── Pilot tuner (A1, A3 selection) ───────────────────────────────────────────


def tune_policy_with_pilot(
    model_id: str,
    selector: str,
    deterministic: bool,
    texts_cal: Sequence[str],
    texts_train: Sequence[str],
    texts_eval: Sequence[str],
    device: str,
    dtype_str: str,
    seed: int,
    risk_score: float,
    max_compile_hours: Optional[float],
    pilot_keep_frac: float,
    prescan_cache: Optional[Dict[int, Dict[str, Any]]] = None,
) -> tuple:
    """Score 2-3 candidate policies via bounded pilot repair.

    Returns (selected_policy, TuningReport).
    """
    dtype = resolve_dtype(dtype_str)
    rng = np.random.RandomState(seed) if deterministic else None
    pilot_budget_s = _compute_pilot_budget(max_compile_hours)

    probe_model, _ = load_model_and_tokenizer(model_id, device, dtype)
    param_b = _estimate_param_billions(probe_model)
    num_layers = probe_model.config.num_hidden_layers
    pilot_layer = num_layers // 2
    del probe_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    ladder = build_policy_ladder(param_b)
    candidates = _pilot_candidates(ladder, risk_score)
    pilot_steps = 75 if pilot_budget_s >= 120 else 50
    eval_subset = list(texts_eval[:64])

    _log.info(
        "[policy] pilot tuner: keep=%.2f budget=%.0fs candidates=[%s] risk=%.3f",
        pilot_keep_frac, pilot_budget_s,
        ", ".join(c.name for c in candidates), risk_score,
    )

    results: List[PilotResult] = []
    budget_start = time.time()

    for cand in candidates:
        elapsed_total = time.time() - budget_start
        if elapsed_total >= pilot_budget_s:
            _log.info("[policy] pilot budget exhausted after %d candidates", len(results))
            break

        model, tok = load_model_and_tokenizer(model_id, device, dtype)

        kept_blocks, kept_idx, _ = select_for_layer(
            model, tok, pilot_layer, texts_cal, pilot_keep_frac,
            MAX_LEN, device, selector=selector,
            prescan_cache=prescan_cache, rng=rng,
        )
        compress_mlp_layer_swiglu_inplace(model, pilot_layer, kept_idx, dtype, device)

        remaining = pilot_budget_s - (time.time() - budget_start)
        pr = run_repair_pilot(
            model=model, tokenizer=tok, texts_train=texts_train,
            layers=[pilot_layer], eval_subset=eval_subset,
            policy=cand, pilot_steps=pilot_steps,
            pilot_budget_s=remaining, device=device,
            deterministic=deterministic, seed=seed,
        )
        results.append(pr)

        _log.info(
            "[policy]   %s: P0=%.2f Pt=%.2f score=%.6f stable=%s helpful=%s (%.1fs)",
            cand.name, pr.P0, pr.Pt, pr.score, pr.stable, pr.helpful, pr.elapsed_s,
        )

        del model, tok
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Selection (A3 rule)
    stable_results = [(r, c) for r, c in zip(results, candidates) if r.stable]

    if not stable_results:
        chosen = candidates[-1]
        reason = "fallback_no_stable"
    else:
        helpful_results = [(r, c) for r, c in stable_results if r.helpful]
        if helpful_results:
            best_r, chosen = max(helpful_results, key=lambda rc: rc[0].score)
            reason = "best_score_stable"
        else:
            chosen = stable_results[-1][1]
            reason = "most_conservative_stable"

    _log.info(
        "[policy] pilot tuner: chosen=%s reason=%s",
        chosen.name, reason,
    )

    report = TuningReport(
        pilot_keep_frac=pilot_keep_frac,
        pilot_budget_s=pilot_budget_s,
        candidates=[c.name for c in candidates],
        results=[r.to_dict() for r in results],
        chosen_policy=chosen.name,
        chosen_reason=reason,
        risk_score=risk_score,
        ladder_start_idx=ladder.index(candidates[0]) if candidates[0] in ladder else 0,
    )
    return chosen, report


# ── Triggered escalation (A4) ────────────────────────────────────────────────


_LR_LADDER = [1e-4, 5e-5, 2e-5]


def escalate_policy(
    policy: RepairPolicy,
    keep_frac: float = 0.0,
    trigger_stage: int = -1,
) -> tuple:
    """Apply one escalation step: lower LR, more steps, smaller stage_size.

    Returns (escalated_policy, details_dict) where details_dict contains
    before/after parameters for logging and compile_report persistence.
    """
    new_lr = policy.lr
    for i, lr_val in enumerate(_LR_LADDER):
        if abs(policy.lr - lr_val) < 1e-8 and i + 1 < len(_LR_LADDER):
            new_lr = _LR_LADDER[i + 1]
            break

    new_steps = math.ceil(policy.steps * 1.5)

    ss_map = {4: 2, 2: 1}
    new_ss = ss_map.get(policy.stage_size, policy.stage_size)

    name = f"{policy.name}_esc"
    _log.info(
        "[policy] escalation: reason=consecutive_failures keep=%.3f stage=%d "
        "lr %.1e->%.1e  steps %d->%d  stage_size %d->%d",
        keep_frac, trigger_stage,
        policy.lr, new_lr, policy.steps, new_steps, policy.stage_size, new_ss,
    )

    details = {
        "trigger_stage": trigger_stage,
        "keep_frac": keep_frac,
        "before": {"lr": policy.lr, "steps": policy.steps, "stage_size": policy.stage_size, "name": policy.name},
        "after": {"lr": new_lr, "steps": new_steps, "stage_size": new_ss, "name": name},
    }

    escalated = RepairPolicy(
        name=name,
        stage_size=new_ss,
        lr=new_lr,
        steps=new_steps,
        early_stop_patience=policy.early_stop_patience,
        regression_limit=policy.regression_limit,
        curve_every=policy.curve_every,
        cheap_eval_texts=policy.cheap_eval_texts,
        cheap_eval_max_tokens=policy.cheap_eval_max_tokens,
        final_eval_max_tokens=policy.final_eval_max_tokens,
        grad_accum_steps=policy.grad_accum_steps,
        max_grad_norm=policy.max_grad_norm,
    )
    return escalated, details


# ── Top-level entry point ────────────────────────────────────────────────────


def auto_select_policy(
    model_id: str,
    selector: str,
    deterministic: bool,
    texts_cal: Sequence[str],
    texts_train: Sequence[str],
    texts_eval: Sequence[str],
    device: str,
    dtype_str: str,
    seed: int = 0,
    prescan_cache: Optional[Dict[int, Dict[str, Any]]] = None,
    risk_score: float = 0.5,
    enable_pilot_tuner: bool = True,
    max_compile_hours: Optional[float] = None,
    pilot_keep_frac: Optional[float] = None,
) -> tuple:
    """Select a repair policy, optionally via bounded pilot tuner.

    If the pilot tuner is enabled and budget is sufficient, runs a scored
    comparison of 2-3 candidate policies. Otherwise falls back to risk-aware
    ladder start.

    Returns (chosen_policy, pilot_report_dict).
    """
    pilot_budget_s = _compute_pilot_budget(max_compile_hours)
    keep_frac = pilot_keep_frac if pilot_keep_frac is not None else PILOT_KEEP_FRAC

    if enable_pilot_tuner and pilot_budget_s >= 60.0:
        chosen, report = tune_policy_with_pilot(
            model_id=model_id,
            selector=selector,
            deterministic=deterministic,
            texts_cal=texts_cal,
            texts_train=texts_train,
            texts_eval=texts_eval,
            device=device,
            dtype_str=dtype_str,
            seed=seed,
            risk_score=risk_score,
            max_compile_hours=max_compile_hours,
            pilot_keep_frac=keep_frac,
            prescan_cache=prescan_cache,
        )
        chosen = risk_scale_policy(chosen, risk_score)
        report_dict = report.to_dict()
        report_dict["selected"] = chosen.name
        return chosen, report_dict

    # Fallback: risk-aware ladder start (no pilot)
    _log.info(
        "[policy] pilot tuner skipped (budget=%.0fs < 60s); using risk-aware ladder",
        pilot_budget_s,
    )
    dtype = resolve_dtype(dtype_str)
    probe_model, _ = load_model_and_tokenizer(model_id, device, dtype)
    param_b = _estimate_param_billions(probe_model)
    del probe_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    ladder = build_policy_ladder(param_b)
    if risk_score >= 0.65:
        idx = min(2, len(ladder) - 1)
    elif risk_score <= 0.35:
        idx = 0
    else:
        idx = min(1, len(ladder) - 1)

    chosen = risk_scale_policy(ladder[idx], risk_score)
    report_dict = {
        "pilot_enabled": False,
        "pilot_budget_s": round(pilot_budget_s, 1),
        "risk_score": round(risk_score, 4),
        "ladder_start_idx": idx,
        "selected": chosen.name,
        "chosen_reason": "risk_ladder_fallback",
    }
    return chosen, report_dict
