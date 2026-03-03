"""Repair policy ladder, stratified pilot tuner, stage-size controller,
steps adaptation, and triggered escalation."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from ._model import load_model_and_tokenizer, resolve_dtype
from ._data import deterministic_subset
from ._eval import eval_perplexity
from ._compile import compress_mlp_layer_swiglu_inplace
from .selectors import select_for_layer, BLOCK_SIZE
from .repair import repair_layers

_log = logging.getLogger(__name__)

PILOT_KEEP_FRAC = 0.85
MAX_LEN = 256
SPIKE_LAMBDA = 6.0
MIN_SLOPE_THRESHOLD = 1e-4
HELPFUL_THRESHOLD = 0.002
WH = 1.0
WI = 10.0
WM = 12.0


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
    """Legacy per-candidate pilot result (kept for backward compat)."""
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
    stage_size_selection: Optional[Dict[str, Any]] = None
    steps_adapted: bool = False

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
            "stage_size_selection": self.stage_size_selection,
            "steps_adapted": self.steps_adapted,
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


def _with_stage_size(policy: RepairPolicy, stage_size: int) -> RepairPolicy:
    """Return a copy of *policy* with a different stage_size."""
    if policy.stage_size == stage_size:
        return policy
    return RepairPolicy(
        name=policy.name, stage_size=stage_size, lr=policy.lr,
        steps=policy.steps, early_stop_patience=policy.early_stop_patience,
        regression_limit=policy.regression_limit, curve_every=policy.curve_every,
        cheap_eval_texts=policy.cheap_eval_texts,
        cheap_eval_max_tokens=policy.cheap_eval_max_tokens,
        final_eval_max_tokens=policy.final_eval_max_tokens,
        grad_accum_steps=policy.grad_accum_steps,
        max_grad_norm=policy.max_grad_norm,
    )


# ── Stratified pilot chunk selection ─────────────────────────────────────────


def _stratified_pilot_chunks(
    layer_order: List[int], stage_size: int, K: int = 2,
) -> List[List[int]]:
    """Pick K representative stage chunks: one early, one from the ~70th pctile.

    This avoids the "first-2-stages-are-easy" bias by sampling both an easy
    (early) and a harder (late) portion of the compressibility ordering.
    """
    if not layer_order:
        return []
    chunks = [layer_order[i : i + stage_size] for i in range(0, len(layer_order), stage_size)]
    if len(chunks) <= 1:
        return chunks[:K]
    early_idx = 0
    late_idx = int(0.70 * (len(chunks) - 1))
    if late_idx == early_idx:
        late_idx = min(early_idx + 1, len(chunks) - 1)
    result = [chunks[early_idx]]
    if late_idx != early_idx and K >= 2:
        result.append(chunks[late_idx])
    return result[:K]


# ── Pilot scoring ────────────────────────────────────────────────────────────


def _score_pilot(
    P0: float, Pt: float, Pmax: float,
    elapsed_s: float, regression_stop: bool, nan_inf: bool,
) -> tuple:
    """Legacy single-metric slope scoring (kept for backward compat)."""
    if nan_inf:
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


def _score_two_stage_pilot(
    stage_stats: List[Dict[str, Any]],
    elapsed_s: float,
) -> tuple:
    """Score a pilot candidate by helpfulness, total improvement, and peak win.

    Returns (score, stable, H, I, M) where:
    - score: combined score (higher is better; -1e9 if unstable)
    - stable: False if any stage had nan_inf (hard failure)
    - H: count of helpful stages
    - I: sum of improve_frac across stages
    - M: max improve_frac across stages (peak win)
    """
    for s in stage_stats:
        assert not (
            s.get("regression_tripwire", False) and s.get("repair_fail", False)
            and not s.get("nan_inf", False)
        ), "regression_tripwire must NOT cause repair_fail without nan_inf"
    if any(s.get("nan_inf", False) for s in stage_stats):
        return (-1e9, False, 0, 0.0, 0.0)
    H = sum(1 for s in stage_stats if s.get("repair_helpful", False))
    I = sum(s.get("improve_frac", 0.0) for s in stage_stats)
    M = max((s.get("improve_frac", 0.0) for s in stage_stats), default=0.0)
    T = max(1e-6, elapsed_s)
    score = (WH * H + WI * I + WM * M) / T
    return (score, True, H, I, M)


# ── Candidate selection ──────────────────────────────────────────────────────


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


# ── Budget ────────────────────────────────────────────────────────────────────


def _compute_pilot_budget(max_compile_hours: Optional[float]) -> float:
    if max_compile_hours is not None:
        total_s = max_compile_hours * 3600
        return min(0.10 * total_s, 480.0)
    return 240.0


# ── Stratified pilot runner ──────────────────────────────────────────────────


def _run_pilot_stages(
    model_id: str,
    texts_cal: Sequence[str],
    texts_train: Sequence[str],
    eval_subset: Sequence[str],
    policy: RepairPolicy,
    keep_frac: float,
    stage_size: int,
    n_stages: int,
    device: str,
    dtype_str: str,
    seed: int,
    deterministic: bool,
    prescan_cache: Optional[Dict[int, Dict[str, Any]]] = None,
    layer_order: Optional[List[int]] = None,
    selector: str = "structural",
    pilot_steps: Optional[int] = None,
    pilot_chunks: Optional[List[List[int]]] = None,
) -> tuple:
    """Run pilot stage chunks of compress+repair and return (stage_stats, elapsed_s).

    If *pilot_chunks* is provided (from _stratified_pilot_chunks), those exact
    chunks are used.  Otherwise falls back to the first *n_stages* sequential
    chunks built from *layer_order*.

    Lightweight pilot: no rollback retry, no final repair, no benchmarking.
    Respects never-worse invariant via repair_layers' pre_repair_metric.
    """
    dtype = resolve_dtype(dtype_str)
    rng = np.random.RandomState(seed) if deterministic else None

    model, tok = load_model_and_tokenizer(model_id, device, dtype)
    num_layers = model.config.num_hidden_layers
    layers = list(range(num_layers))

    if pilot_chunks is not None:
        chunks = pilot_chunks
    else:
        if layer_order is not None:
            compressible = [li for li in layer_order if li in set(layers)]
        else:
            compressible = list(layers)
        target_layers = compressible[: n_stages * stage_size]
        chunks = [target_layers[i : i + stage_size] for i in range(0, len(target_layers), stage_size)]
        chunks = chunks[:n_stages]

    eval_tokens = min(3000, policy.cheap_eval_max_tokens)
    steps = pilot_steps if pilot_steps is not None else min(policy.steps, 75)

    compressed_so_far: List[int] = []
    stage_stats: List[Dict[str, Any]] = []
    t0 = time.time()

    for si, chunk in enumerate(chunks):
        for li in chunk:
            kept_blocks, kept_idx, _ = select_for_layer(
                model, tok, li, texts_cal, keep_frac,
                MAX_LEN, device, selector=selector,
                prescan_cache=prescan_cache, rng=rng,
            )
            compress_mlp_layer_swiglu_inplace(model, li, kept_idx, dtype, device)
        compressed_so_far.extend(chunk)

        P0 = eval_perplexity(model, tok, eval_subset, MAX_LEN, device, eval_tokens)

        if math.isnan(P0) or math.isinf(P0):
            stage_stats.append({
                "stage": si, "layers": chunk,
                "ppl_pre_repair": float("inf"), "ppl_best": float("inf"),
                "improve_frac": 0.0, "regression_tripwire": False, "nan_inf": True,
                "early_stop": False,
                "repair_fail": True, "repair_helpful": False,
            })
            break

        Pbest = P0
        reg_stop = False
        nan_inf = False
        early_stop = False

        if steps > 0:
            def _curve(step, _model=model, _tok=tok, _es=eval_subset):
                v = eval_perplexity(_model, _tok, _es, MAX_LEN, device, eval_tokens)
                return {"ppl_w103_valid": v}

            warmup = min(15, steps // 5)
            sr = repair_layers(
                model=model, tokenizer=tok, texts_train=texts_train,
                layers=compressed_so_far, steps=steps, lr=policy.lr,
                warmup=warmup, weight_decay=0.01,
                max_len=MAX_LEN, device=device,
                grad_accum_steps=policy.grad_accum_steps,
                curve_fn=_curve, curve_every=max(10, steps // 5),
                early_stop_patience=policy.early_stop_patience,
                regression_limit=policy.regression_limit,
                save_best=True, pre_repair_metric=P0,
            )

            Pbest = sr.get("best_metric", P0)
            reg_stop = sr.get(
                "regression_tripwire_triggered",
                sr.get("regression_stop_triggered", False),
            )
            nan_inf = sr.get("nan_inf_detected", False)
            early_stop = sr.get("early_stop_triggered", False)
            if math.isnan(Pbest) or math.isinf(Pbest):
                Pbest = P0

        improve = (P0 - Pbest) / P0 if P0 > 0 else 0.0
        fail = nan_inf
        helpful = (not fail) and improve >= HELPFUL_THRESHOLD

        assert not (reg_stop and fail and not nan_inf), \
            "regression_tripwire must NOT cause repair_fail without nan_inf"

        stage_stats.append({
            "stage": si, "layers": chunk,
            "ppl_pre_repair": round(P0, 4),
            "ppl_best": round(Pbest, 4),
            "improve_frac": round(improve, 6),
            "regression_tripwire": reg_stop, "nan_inf": nan_inf,
            "early_stop": early_stop,
            "repair_fail": fail, "repair_helpful": helpful,
        })

    elapsed = time.time() - t0
    del model, tok
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return stage_stats, elapsed


# ── Steps adaptation ─────────────────────────────────────────────────────────


def _adapt_steps(policy: RepairPolicy, H: int, I: float, M: float = 0.0) -> RepairPolicy:
    """Increase steps if repair proved helpful or showed a large peak win."""
    if (I >= 0.005) or (M >= 0.01):
        new_steps = min(int(policy.steps * 1.25), policy.steps + 200)
        if new_steps > policy.steps:
            _log.info(
                "[policy] adapting steps %d -> %d (H=%d, I=%.4f, M=%.4f)",
                policy.steps, new_steps, H, I, M,
            )
            return RepairPolicy(
                name=policy.name, stage_size=policy.stage_size, lr=policy.lr,
                steps=new_steps, early_stop_patience=policy.early_stop_patience,
                regression_limit=policy.regression_limit,
                curve_every=policy.curve_every,
                cheap_eval_texts=policy.cheap_eval_texts,
                cheap_eval_max_tokens=policy.cheap_eval_max_tokens,
                final_eval_max_tokens=policy.final_eval_max_tokens,
                grad_accum_steps=policy.grad_accum_steps,
                max_grad_norm=policy.max_grad_norm,
            )
    return policy


# ── Stage-size controller ────────────────────────────────────────────────────


TIE_BREAK_GAIN = 0.25


def _recovery_strength(stats: List[Dict[str, Any]]) -> float:
    """I + M: total improve_frac plus max improve_frac across stages."""
    I = sum(s.get("improve_frac", 0.0) for s in stats)
    M = max((s.get("improve_frac", 0.0) for s in stats), default=0.0)
    return I + M


def _select_stage_size(
    model_id: str,
    policy: RepairPolicy,
    texts_cal: Sequence[str],
    texts_train: Sequence[str],
    eval_subset: Sequence[str],
    keep_frac: float,
    device: str,
    dtype_str: str,
    seed: int,
    deterministic: bool,
    num_layers: int,
    prescan_cache: Optional[Dict[int, Dict[str, Any]]] = None,
    layer_order: Optional[List[int]] = None,
    selector: str = "structural",
    candidates: tuple = (4, 2),
    n_probe_stages: int = 2,
    budget_remaining_s: float = 120.0,
) -> tuple:
    """Probe stage_size candidates using stratified chunks, prefer largest
    stable but allow smaller when recovery signal is materially stronger.

    Returns (chosen_stage_size, probe_report_dict).
    """
    probe_report: Dict[str, Any] = {"candidates": list(candidates), "probes": []}
    default_order = layer_order if layer_order is not None else list(range(num_layers))

    stable_results: List[Dict[str, Any]] = []

    for ss in candidates:
        if budget_remaining_s <= 30:
            _log.info("[policy] stage_size probe: budget low (%.0fs), skipping ss=%d", budget_remaining_s, ss)
            break

        probe_chunks = _stratified_pilot_chunks(default_order, ss, K=n_probe_stages)
        probe_policy = _with_stage_size(policy, ss)
        stats, elapsed = _run_pilot_stages(
            model_id=model_id,
            texts_cal=texts_cal, texts_train=texts_train,
            eval_subset=eval_subset, policy=probe_policy,
            keep_frac=keep_frac, stage_size=ss,
            n_stages=n_probe_stages, device=device,
            dtype_str=dtype_str, seed=seed, deterministic=deterministic,
            prescan_cache=prescan_cache, layer_order=layer_order,
            selector=selector, pilot_chunks=probe_chunks,
        )
        budget_remaining_s -= elapsed

        has_fail = any(s.get("repair_fail", False) for s in stats)
        has_helpful = any(s.get("repair_helpful", False) for s in stats)
        I_val = sum(s.get("improve_frac", 0.0) for s in stats)
        M_val = max((s.get("improve_frac", 0.0) for s in stats), default=0.0)
        rec = _recovery_strength(stats)

        entry = {
            "stage_size": ss,
            "stats": stats,
            "elapsed_s": round(elapsed, 1),
            "has_fail": has_fail,
            "has_helpful": has_helpful,
            "I": round(I_val, 6),
            "M": round(M_val, 6),
            "recovery_strength": round(rec, 6),
        }
        probe_report["probes"].append(entry)
        _log.info(
            "[policy] stage_size probe: ss=%d stable=%s I=%.4f M=%.4f rec=%.4f (%.1fs)",
            ss, not has_fail, I_val, M_val, rec, elapsed,
        )

        if not has_fail:
            stable_results.append(entry)

    # Selection: largest stable by default, with recovery tie-break
    if not stable_results:
        chosen = candidates[-1]
        reason = "all_unstable_fallback"
    elif len(stable_results) == 1:
        chosen = stable_results[0]["stage_size"]
        reason = "only_stable"
    else:
        largest = stable_results[0]
        chosen = largest["stage_size"]
        reason = "largest_stable"
        rec_large = largest["recovery_strength"]
        for entry in stable_results[1:]:
            rec_small = entry["recovery_strength"]
            threshold = (1.0 + TIE_BREAK_GAIN) * rec_large if rec_large > 0 else 1e-9
            if rec_small >= threshold:
                chosen = entry["stage_size"]
                reason = "recovery_stronger"
                break

    probe_report["chosen"] = chosen
    probe_report["reason"] = reason
    _log.info("[policy] stage_size chosen: ss=%d reason=%s", chosen, reason)
    return chosen, probe_report


# ── Pilot tuner orchestrator ─────────────────────────────────────────────────


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
    layer_order: Optional[List[int]] = None,
) -> tuple:
    """Stratified pilot tuner: select LR policy, adapt steps, choose stage_size.

    Phase 1: evaluate <=3 candidate LR policies via stratified 2-stage pilot
             (early + ~70th-percentile chunk, fixed ss=4).
    Phase 2: adapt steps based on measured helpfulness / peak improvement.
    Phase 3: probe stage_sizes {4,2} with stratified chunks using winning policy.

    Returns (selected_policy, TuningReport).
    """
    pilot_budget_s = _compute_pilot_budget(max_compile_hours)
    dtype = resolve_dtype(dtype_str)

    probe_model, _ = load_model_and_tokenizer(model_id, device, dtype)
    param_b = _estimate_param_billions(probe_model)
    num_layers = probe_model.config.num_hidden_layers
    del probe_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    ladder = build_policy_ladder(param_b)
    candidates = _pilot_candidates(ladder, risk_score)
    eval_subset = list(texts_eval[:64])
    default_order = layer_order if layer_order is not None else list(range(num_layers))

    _log.info(
        "[policy] pilot tuner: keep=%.2f budget=%.0fs candidates=[%s] risk=%.3f",
        pilot_keep_frac, pilot_budget_s,
        ", ".join(c.name for c in candidates), risk_score,
    )

    # Phase 1: Policy (LR) selection via stratified 2-stage pilot at fixed ss=4
    pilot_lr_chunks = _stratified_pilot_chunks(default_order, stage_size=4, K=2)
    budget_start = time.time()
    pilot_results: List[Dict[str, Any]] = []

    for cand in candidates:
        elapsed_total = time.time() - budget_start
        remaining = pilot_budget_s - elapsed_total
        if remaining < 30:
            _log.info("[policy] pilot budget low after %d candidates", len(pilot_results))
            break

        norm_cand = _with_stage_size(cand, 4)
        stats, elapsed = _run_pilot_stages(
            model_id=model_id,
            texts_cal=texts_cal, texts_train=texts_train,
            eval_subset=eval_subset, policy=norm_cand,
            keep_frac=pilot_keep_frac, stage_size=4,
            n_stages=2, device=device, dtype_str=dtype_str,
            seed=seed, deterministic=deterministic,
            prescan_cache=prescan_cache, layer_order=layer_order,
            selector=selector, pilot_chunks=pilot_lr_chunks,
        )

        score, stable, H, I, M = _score_two_stage_pilot(stats, elapsed)
        result = {
            "policy_name": cand.name,
            "stage_stats": stats,
            "elapsed_s": round(elapsed, 2),
            "score": round(score, 8),
            "stable": stable,
            "H": H,
            "I": round(I, 6),
            "M": round(M, 6),
        }
        pilot_results.append(result)

        _log.info(
            "[policy]   %s: H=%d I=%.4f M=%.4f score=%.6f stable=%s (%.1fs)",
            cand.name, H, I, M, score, stable, elapsed,
        )

    # Selection rule
    stable_results = [(r, c) for r, c in zip(pilot_results, candidates) if r["stable"]]
    best_H, best_I, best_M = 0, 0.0, 0.0

    if not stable_results:
        chosen = candidates[-1]
        reason = "fallback_no_stable"
    else:
        helpful_results = [(r, c) for r, c in stable_results if r["H"] > 0]
        if helpful_results:
            best_r, chosen = max(helpful_results, key=lambda rc: rc[0]["score"])
            reason = "best_score_helpful"
            best_H, best_I, best_M = best_r["H"], best_r["I"], best_r["M"]
        else:
            chosen = stable_results[-1][1]
            reason = "most_conservative_stable"

    _log.info("[policy] pilot tuner: chosen=%s reason=%s", chosen.name, reason)

    # Phase 2: Steps adaptation (uses I or M)
    original_steps = chosen.steps
    chosen = _adapt_steps(chosen, best_H, best_I, best_M)
    steps_adapted = chosen.steps != original_steps

    # Phase 3: Stage-size selection with stratified probing
    budget_remaining = pilot_budget_s - (time.time() - budget_start)
    if budget_remaining >= 60:
        chosen_ss, ss_report = _select_stage_size(
            model_id=model_id, policy=chosen,
            texts_cal=texts_cal, texts_train=texts_train,
            eval_subset=eval_subset, keep_frac=pilot_keep_frac,
            device=device, dtype_str=dtype_str,
            seed=seed, deterministic=deterministic,
            num_layers=num_layers,
            prescan_cache=prescan_cache, layer_order=layer_order,
            selector=selector, budget_remaining_s=budget_remaining,
        )
        if chosen_ss != chosen.stage_size:
            chosen = _with_stage_size(chosen, chosen_ss)
    else:
        ss_report = {"chosen": chosen.stage_size, "skipped": True, "reason": "insufficient_budget"}
        _log.info("[policy] stage_size probe skipped (budget=%.0fs)", budget_remaining)

    report = TuningReport(
        pilot_keep_frac=pilot_keep_frac,
        pilot_budget_s=pilot_budget_s,
        candidates=[c.name for c in candidates],
        results=pilot_results,
        chosen_policy=chosen.name,
        chosen_reason=reason,
        risk_score=risk_score,
        ladder_start_idx=ladder.index(candidates[0]) if candidates[0] in ladder else 0,
        stage_size_selection=ss_report,
        steps_adapted=steps_adapted,
    )
    return chosen, report


# ── Triggered escalation ─────────────────────────────────────────────────────


_LR_LADDER = [1e-4, 5e-5, 2e-5]


def escalate_policy(
    policy: RepairPolicy,
    keep_frac: float = 0.0,
    trigger_stage: int = -1,
) -> tuple:
    """Apply one escalation step: lower LR, more steps, smaller stage_size.

    Returns (escalated_policy, details_dict).
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
        name=name, stage_size=new_ss, lr=new_lr, steps=new_steps,
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


# ── E2E workload speedup ─────────────────────────────────────────────────────


E2E_PROFILES = {
    "chat":  {"P": 256,  "D": 256},
    "rag":   {"P": 2048, "D": 128},
    "batch": {"P": 1024, "D": 32},
}


def compute_e2e_speedup(
    prefill_speedup: float, decode_speedup: float, P: int, D: int,
) -> float:
    """Workload-modeled end-to-end speedup for a (P-token prefill, D-token decode) job."""
    if prefill_speedup <= 0 or decode_speedup <= 0:
        return 0.0
    return (P + D) / (P / prefill_speedup + D / decode_speedup)


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
    layer_order: Optional[List[int]] = None,
) -> tuple:
    """Select a repair policy, optionally via bounded stratified pilot tuner.

    Returns (chosen_policy, report_dict).
    """
    pilot_budget_s = _compute_pilot_budget(max_compile_hours)
    keep_frac = pilot_keep_frac if pilot_keep_frac is not None else PILOT_KEEP_FRAC

    if enable_pilot_tuner and pilot_budget_s >= 60.0:
        chosen, report = tune_policy_with_pilot(
            model_id=model_id, selector=selector,
            deterministic=deterministic,
            texts_cal=texts_cal, texts_train=texts_train,
            texts_eval=texts_eval, device=device,
            dtype_str=dtype_str, seed=seed,
            risk_score=risk_score,
            max_compile_hours=max_compile_hours,
            pilot_keep_frac=keep_frac,
            prescan_cache=prescan_cache,
            layer_order=layer_order,
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
