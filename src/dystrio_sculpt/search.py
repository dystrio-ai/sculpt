"""Frontier search: Thompson Sampling Search (TSS) over uniform keep_frac.

Default objective: find the **fastest safe keep_frac** under a quality ceiling,
then emit up to N named points around the safe optimum.

Uses Thompson Sampling with Beta-conjugate posteriors (inspired by AB-MCTS,
Sakana AI, arXiv:2503.04412) to adaptively interleave exploration of new
keep_frac candidates with bisection refinement of the safe/unsafe bracket.

Uses structural risk score from prescan artifacts to steer the initial bracket,
repair policy selection, and stage ordering.
"""

from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from ._model import load_model_and_tokenizer, resolve_dtype
from ._data import CalibConfig, load_text_sets
from .engine import CompileResult, compile_model, _collect_metrics, setup_determinism, MAX_LEN
from .policy import RepairPolicy, auto_select_policy
from .risk import (
    model_risk_score,
    risk_aware_keep_candidates,
    layer_compressibility_order,
    risk_weighted_keep_schedule,
    protected_layers,
    DEFAULT_PROTECTION_THRESHOLD,
)
from .selectors import BLOCK_SIZE
from .selectors.structural import prescan_structural_artifacts

_log = logging.getLogger(__name__)

DEFAULT_PPL_CEILING = 2.0


# ── Thompson Sampling primitives ─────────────────────────────────────────────


@dataclass
class BetaArm:
    """Conjugate Beta posterior for a Bernoulli bandit arm.

    Jeffrey's prior (a=0.5, b=0.5) provides a minimally informative
    starting point.  After observing a reward r in [0,1], the posterior
    updates as a += r, b += (1 - r).

    Inspired by AB-MCTS-A (Sakana AI, arXiv:2503.04412).
    """

    a: float = 0.5
    b: float = 0.5

    def sample(self, rng: Optional[np.random.RandomState] = None) -> float:
        """Draw a single Thompson sample from the posterior."""
        if rng is not None:
            return float(rng.beta(self.a, self.b))
        return float(np.random.beta(self.a, self.b))

    def update(self, reward: float) -> None:
        """Bayesian update with an observed reward in [0, 1]."""
        self.a += reward
        self.b += (1.0 - reward)

    @property
    def mean(self) -> float:
        return self.a / (self.a + self.b)

    @property
    def n_obs(self) -> float:
        """Effective number of observations (excludes prior mass)."""
        return self.a + self.b - 1.0


def _safety_reward(pt: "FrontierPoint", ceiling: float) -> float:
    """Quality-weighted safety reward in [0, 1].

    Maps ppl_ratio linearly: 1.0 at baseline quality (ratio=1.0),
    0.0 at 1.5x the ceiling.  Failed points always return 0.
    """
    if pt.failed:
        return 0.0
    ratio = pt.ppl_ratio
    if ratio <= 1.0:
        return 1.0
    upper = ceiling * 1.5
    if ratio >= upper:
        return 0.0
    return max(0.0, 1.0 - (ratio - 1.0) / (upper - 1.0))


# ── Speed profiles ────────────────────────────────────────────────────────────


SPEED_PROFILES: Dict[str, Dict[str, float]] = {
    "balanced": {"prefill_weight": 0.5, "decode_weight": 0.5},
    "prefill_heavy": {"prefill_weight": 0.8, "decode_weight": 0.2},
    "decode_heavy": {"prefill_weight": 0.2, "decode_weight": 0.8},
    "rag": {"prefill_weight": 0.75, "decode_weight": 0.25},
    "chatbot": {"prefill_weight": 0.25, "decode_weight": 0.75},
    "throughput": {"prefill_weight": 0.6, "decode_weight": 0.4},
    "latency": {"prefill_weight": 0.4, "decode_weight": 0.6},
}


def blended_speedup(
    prefill_speedup: float,
    decode_speedup: float,
    prefill_weight: float = 0.5,
    decode_weight: float = 0.5,
) -> float:
    """Compute workload-weighted speedup from prefill and decode components."""
    total = prefill_weight + decode_weight
    if total < 1e-9:
        return prefill_speedup
    pw = prefill_weight / total
    dw = decode_weight / total
    return pw * prefill_speedup + dw * decode_speedup


# ── Data classes ─────────────────────────────────────────────────────────────


@dataclass
class FrontierPoint:
    """A single evaluated point on the keep_frac search space."""

    keep_frac: float
    ppl_w103: float
    ppl_w2: float
    prefill_tps: float
    decode_tps: float
    prefill_speedup: float
    decode_speedup: float
    wall_time_s: float
    ppl_ratio: float = 0.0
    compile_result: Optional[CompileResult] = field(default=None, repr=False)
    label: str = ""
    failed: bool = False
    failure_reason: str = ""
    risk_score: float = 0.0
    blended_speedup: float = 0.0


def _is_safe(pt: FrontierPoint, ceiling: float) -> bool:
    return not pt.failed and pt.ppl_ratio <= ceiling


_ORDERED_TIER_NAMES = ["default", "production", "throughput", "experimental", "frontier"]


def _assign_labels(
    selected: List[FrontierPoint], ceiling: Optional[float],
) -> None:
    """Assign semantic labels by keep_frac order (descending).

    Highest keep_frac = default (free-lunch), next = production, etc.
    Points above the quality ceiling are excluded from named tiers and
    given generic ``point_N`` labels.  With only 1 point, it gets
    "production" if safe, "default" otherwise.
    """
    if not selected:
        return

    # Sort by keep_frac descending (most conservative first)
    selected.sort(key=lambda p: -p.keep_frac)

    if len(selected) == 1:
        pt = selected[0]
        if ceiling and _is_safe(pt, ceiling):
            pt.label = "frontier_0_production"
        else:
            pt.label = "frontier_0_default"
        return

    # Separate safe (under ceiling) from unsafe points
    safe = [p for p in selected if ceiling is None or _is_safe(p, ceiling)]
    unsafe = [p for p in selected if p not in safe]

    # Assign named tiers to safe points in keep_frac order
    for i, pt in enumerate(safe):
        if i < len(_ORDERED_TIER_NAMES):
            pt.label = f"frontier_{i}_{_ORDERED_TIER_NAMES[i]}"
        else:
            pt.label = f"frontier_{i}_point{i}"

    # Label unsafe points generically
    next_idx = len(safe)
    for pt in unsafe:
        pt.label = f"frontier_{next_idx}_point{next_idx}"
        next_idx += 1


class FrontierSearch:
    """Thompson Sampling Search (TSS) over uniform keep_frac in [0.4, 1.0].

    Algorithm:
      1. Compute baseline and structural prescan.
      2. Derive model risk score from prescan artifacts.
      3. Choose initial keep candidates based on risk.
      4. Auto-select repair policy (risk-aware).
      5. Adaptively interleave candidate exploration and bracket bisection
         using Thompson Sampling with Beta-conjugate posteriors.
      6. Emit up to n_frontier named points under the quality ceiling.

    The explore/exploit decision is governed by two Beta-Bernoulli arms:
    one for exploring new keep_frac candidates, one for bisecting the
    current safe/unsafe bracket.  Rewards are quality-weighted safety
    scores, so the algorithm naturally converges on the boundary region.
    """

    def __init__(
        self,
        model_id: str,
        n_frontier: int = 4,
        *,
        max_ppl_multiplier: Optional[float] = None,
        target_prefill_speedup: Optional[float] = None,
        max_compile_hours: Optional[float] = None,
        deterministic: bool = False,
        seed: int = 0,
        device: str = "cuda",
        dtype_str: str = "bf16",
        n_texts_cal: int = 400,
        n_texts_train: int = 2500,
        n_texts_eval: int = 300,
        max_eval_tokens: int = 40_000,
        selector: str = "structural",
        policy_override: Optional[RepairPolicy] = None,
        outdir: Optional[Path] = None,
        calib: Optional[CalibConfig] = None,
        speed_profile: Optional[str] = None,
        use_risk_schedule: bool = False,
        protection_threshold: Optional[float] = None,
    ):
        self.model_id = model_id
        self.n_frontier = n_frontier
        # Default quality ceiling
        if max_ppl_multiplier is None or max_ppl_multiplier <= 0:
            self.max_ppl_multiplier = DEFAULT_PPL_CEILING
            self._ceiling_is_default = True
        else:
            self.max_ppl_multiplier = max_ppl_multiplier
            self._ceiling_is_default = False
        self.target_prefill_speedup = target_prefill_speedup
        self.max_compile_hours = max_compile_hours
        self.deterministic = deterministic
        self.seed = seed
        self.device = device
        self.dtype_str = dtype_str
        self.n_texts_cal = n_texts_cal
        self.n_texts_train = n_texts_train
        self.n_texts_eval = n_texts_eval
        self.max_eval_tokens = max_eval_tokens
        self.selector = selector
        self.policy_override = policy_override
        self.outdir = outdir
        self.calib = calib

        # Speed profile for workload-aware scoring
        if speed_profile and speed_profile in SPEED_PROFILES:
            self._speed_weights = SPEED_PROFILES[speed_profile]
        else:
            self._speed_weights = SPEED_PROFILES["balanced"]
        self._speed_profile_name = speed_profile or "balanced"

        # Per-layer keep_frac schedule options
        self.use_risk_schedule = use_risk_schedule
        self.protection_threshold = protection_threshold

        self.texts: Optional[Dict[str, List[str]]] = None
        self.prescan_cache: Optional[Dict[int, Dict[str, Any]]] = None
        self.baseline_metrics: Optional[Dict[str, float]] = None
        self.policy: Optional[RepairPolicy] = None
        self.pilot_report: Optional[Dict[str, Any]] = None
        self.risk_score: float = 0.5
        self.risk_detail: Optional[Dict[str, Any]] = None
        self.layer_order: Optional[List[int]] = None
        self.evaluated: List[FrontierPoint] = []
        self._start_time: float = 0.0
        self._escalation_applied: bool = False
        self._tuning_report: Optional[Dict[str, Any]] = None

    def _time_exceeded(self) -> bool:
        if self.max_compile_hours is None:
            return False
        elapsed_h = (time.time() - self._start_time) / 3600
        return elapsed_h >= self.max_compile_hours

    def _setup(self) -> None:
        self._start_time = time.time()
        setup_determinism(self.seed, self.deterministic)
        _log.info("loading datasets")
        self.texts = load_text_sets(
            self.n_texts_cal, self.n_texts_train, self.n_texts_eval, calib=self.calib,
        )

    def _compute_baseline(self) -> None:
        _log.info("computing baseline (no compression)")
        setup_determinism(self.seed, self.deterministic)
        dtype = resolve_dtype(self.dtype_str)
        model, tok = load_model_and_tokenizer(self.model_id, self.device, dtype)
        assert self.texts is not None
        self.baseline_metrics = _collect_metrics(
            model, tok, self.texts, self.device, self.max_eval_tokens,
        )
        if torch.cuda.is_available():
            self.baseline_metrics["cuda_allocated_baseline_bytes"] = torch.cuda.memory_allocated()
        _log.info(
            "baseline: ppl_w103=%.2f  prefill=%.0f tok/s  decode=%.0f tok/s",
            self.baseline_metrics["ppl_w103_valid"],
            self.baseline_metrics["prefill_tokens_per_sec"],
            self.baseline_metrics["decode_tokens_per_sec"],
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _compute_prescan(self) -> None:
        if self.selector != "structural":
            return
        _log.info("running structural prescan (cached for all iterations)")
        setup_determinism(self.seed, self.deterministic)
        dtype = resolve_dtype(self.dtype_str)
        model, tok = load_model_and_tokenizer(self.model_id, self.device, dtype)
        num_layers = model.config.num_hidden_layers
        assert self.texts is not None
        self.prescan_cache = prescan_structural_artifacts(
            model, tok, list(range(num_layers)), self.texts["cal"],
            MAX_LEN, self.device, block_size=BLOCK_SIZE,
        )
        _log.info("prescan cached %d layers", len(self.prescan_cache))
        del model, tok
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _compute_risk(self) -> None:
        if self.prescan_cache:
            self.risk_score, self.risk_detail = model_risk_score(self.prescan_cache)
            self.layer_order = layer_compressibility_order(self.prescan_cache)
        else:
            self.risk_score = 0.5
            self.risk_detail = {"aggregate": 0.5, "source": "no_prescan"}
            self.layer_order = None
        _log.info("structural risk score: %.3f", self.risk_score)

    def _select_policy(self) -> None:
        if self.policy_override is not None:
            self.policy = self.policy_override
            _log.info("using user-supplied policy override: %s", self.policy.name)
            return

        assert self.texts is not None
        candidates = risk_aware_keep_candidates(self.risk_score)
        pilot_kf = 0.85 if 0.70 <= 0.85 <= 0.95 else (candidates[0] if candidates else 0.85)

        _log.info(
            "running policy auto-selection pilot (risk=%.3f, budget_hours=%s, pilot_kf=%.2f)",
            self.risk_score, self.max_compile_hours, pilot_kf,
        )
        self.policy, self.pilot_report = auto_select_policy(
            model_id=self.model_id,
            selector=self.selector,
            deterministic=self.deterministic,
            texts_cal=self.texts["cal"],
            texts_train=self.texts["train"],
            texts_eval=self.texts["eval_w103"],
            device=self.device,
            dtype_str=self.dtype_str,
            seed=self.seed,
            prescan_cache=self.prescan_cache,
            risk_score=self.risk_score,
            enable_pilot_tuner=True,
            max_compile_hours=self.max_compile_hours,
            pilot_keep_frac=pilot_kf,
            layer_order=self.layer_order,
        )
        self._tuning_report = self.pilot_report
        _log.info("policy auto-selected: %s", self.policy.name)

    def _build_keep_schedule(self, keep_frac: float) -> Optional[Dict[int, float]]:
        """Build per-layer keep_frac schedule from risk scores if enabled."""
        if not self.use_risk_schedule or not self.prescan_cache:
            return None
        threshold = self.protection_threshold or DEFAULT_PROTECTION_THRESHOLD
        return risk_weighted_keep_schedule(
            self.prescan_cache,
            aggressiveness=1.0 - keep_frac,
            protection_threshold=threshold,
        )

    def _blended(self, prefill_su: float, decode_su: float) -> float:
        return blended_speedup(
            prefill_su, decode_su,
            self._speed_weights["prefill_weight"],
            self._speed_weights["decode_weight"],
        )

    def _evaluate(self, keep_frac: float) -> FrontierPoint:
        assert self.texts is not None
        assert self.baseline_metrics is not None
        assert self.policy is not None

        schedule = self._build_keep_schedule(keep_frac)
        mode = "risk_schedule" if schedule else "uniform"
        _log.info(
            "[search] evaluating keep_frac=%.3f policy=%s mode=%s profile=%s",
            keep_frac, self.policy.name, mode, self._speed_profile_name,
        )

        failure_subdir = (self.outdir / f"_failed_kf{keep_frac:.3f}") if self.outdir else None
        base_ppl = self.baseline_metrics["ppl_w103_valid"]

        try:
            result = compile_model(
                self.model_id,
                keep_frac,
                texts=self.texts,
                prescan_cache=self.prescan_cache,
                baseline_metrics=self.baseline_metrics,
                policy=self.policy,
                pilot_report=self.pilot_report,
                device=self.device,
                dtype_str=self.dtype_str,
                seed=self.seed,
                deterministic=self.deterministic,
                n_texts_cal=self.n_texts_cal,
                n_texts_train=self.n_texts_train,
                n_texts_eval=self.n_texts_eval,
                max_eval_tokens=self.max_eval_tokens,
                selector=self.selector,
                failure_dir=failure_subdir,
                layer_order=self.layer_order,
                allow_escalation=not self._escalation_applied,
                keep_schedule=schedule,
            )
        except Exception as exc:
            _log.error("compile_model failed for kf=%.3f: %s", keep_frac, exc)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            point = FrontierPoint(
                keep_frac=keep_frac, ppl_w103=float("inf"), ppl_w2=float("inf"),
                prefill_tps=0.0, decode_tps=0.0, prefill_speedup=0.0,
                decode_speedup=0.0, wall_time_s=0.0, ppl_ratio=float("inf"),
                failed=True, failure_reason=str(exc),
                risk_score=self.risk_score,
            )
            self.evaluated.append(point)
            return point

        # Persist escalated policy for subsequent keep_frac evaluations
        if result.escalation_applied and not self._escalation_applied:
            self._escalation_applied = True
            # Adopt the escalated policy (stored as current_policy in config)
            esc_cfg = result.config.get("policy", {})
            if esc_cfg and esc_cfg.get("name", "").endswith("_esc"):
                self.policy = RepairPolicy(**{
                    k: v for k, v in esc_cfg.items()
                    if k in RepairPolicy.__dataclass_fields__
                })
            _log.info(
                "[search] escalation persisted: policy=%s (from keep=%.3f)",
                self.policy.name, keep_frac,
            )

        # Move model to CPU to free GPU for subsequent iterations.
        # The model is still accessible for the emit step (saving to disk).
        if result.model is not None:
            result.model.cpu()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        m = result.metrics_post
        base = self.baseline_metrics
        prefill_speedup = m["prefill_tokens_per_sec"] / max(1e-9, base["prefill_tokens_per_sec"])
        decode_speedup = m["decode_tokens_per_sec"] / max(1e-9, base["decode_tokens_per_sec"])
        ppl_ratio = m["ppl_w103_valid"] / max(1e-9, base_ppl)

        failed = result.guardrail_failed or result.failure is not None
        point = FrontierPoint(
            keep_frac=keep_frac,
            ppl_w103=m["ppl_w103_valid"],
            ppl_w2=m["ppl_w2_test"],
            prefill_tps=m["prefill_tokens_per_sec"],
            decode_tps=m["decode_tokens_per_sec"],
            prefill_speedup=prefill_speedup,
            decode_speedup=decode_speedup,
            wall_time_s=result.wall_time_s,
            ppl_ratio=ppl_ratio,
            compile_result=result,
            failed=failed,
            failure_reason=result.failure["reason"] if result.failure else "",
            risk_score=self.risk_score,
            blended_speedup=self._blended(prefill_speedup, decode_speedup),
        )
        self.evaluated.append(point)

        safe_str = "SAFE" if _is_safe(point, self.max_ppl_multiplier) else "OVER_CEILING"
        _log.info(
            "keep_frac=%.3f  ppl_ratio=%.3f  prefill_speedup=%.2fx  [%s]  (%.0fs)",
            keep_frac, ppl_ratio, prefill_speedup, safe_str, result.wall_time_s,
        )
        return point

    def _safe_points(self) -> List[FrontierPoint]:
        return [p for p in self.evaluated if _is_safe(p, self.max_ppl_multiplier)]

    def _fastest_safe(self) -> Optional[FrontierPoint]:
        safe = self._safe_points()
        if not safe:
            return None
        return max(safe, key=lambda p: p.blended_speedup)

    def _best_quality(self) -> Optional[FrontierPoint]:
        safe = self._safe_points()
        if not safe:
            viable = [p for p in self.evaluated if not p.failed]
            if not viable:
                return None
            return min(viable, key=lambda p: p.ppl_ratio)
        return min(safe, key=lambda p: p.ppl_ratio)

    def run(self) -> List[FrontierPoint]:
        """Execute Thompson Sampling Search. Returns selected frontier points."""
        self._setup()
        self._compute_baseline()
        self._compute_prescan()
        self._compute_risk()
        self._select_policy()

        ceiling = self.max_ppl_multiplier
        reject_margin = ceiling * 1.05

        # Risk-aware initial candidates (descending keep_frac = conservative first)
        candidates = risk_aware_keep_candidates(self.risk_score)
        candidates.sort(reverse=True)
        _log.info("initial candidates (risk=%.3f): %s", self.risk_score, candidates)

        rng = np.random.RandomState(self.seed) if self.deterministic else None

        # Thompson Sampling state: two arms for explore vs exploit
        explore_arm = BetaArm()
        exploit_arm = BetaArm()
        remaining = list(candidates)
        k_safe_best: Optional[float] = None
        k_unsafe: Optional[float] = None

        _over_ceiling_extras = 0
        _MAX_OVER_CEILING_EXTRAS = 1
        _MIN_PREFILL_DELTA_FOR_EXTRA = 0.005
        _prev_prefill_speedup: float = 0.0
        min_resolution = 0.03
        max_iterations = len(candidates) + 6

        for iteration in range(max_iterations):
            if self._time_exceeded():
                _log.info("time budget reached at iteration %d", iteration)
                break

            # Determine available actions
            can_explore = len(remaining) > 0
            has_bracket = k_safe_best is not None and k_unsafe is not None
            can_exploit = has_bracket and (k_safe_best - k_unsafe) >= min_resolution

            if not can_explore and not can_exploit:
                break

            # Thompson Sampling: sample both arms, pick the winner
            if not can_exploit:
                action = "explore"
            elif not can_explore:
                action = "exploit"
            else:
                e_sample = explore_arm.sample(rng)
                x_sample = exploit_arm.sample(rng)
                action = "explore" if e_sample >= x_sample else "exploit"

            if action == "explore":
                kf = remaining.pop(0)
            else:
                mid = round((k_safe_best + k_unsafe) / 2, 3)
                if any(abs(p.keep_frac - mid) < 0.005 for p in self.evaluated):
                    if remaining:
                        action = "explore"
                        kf = remaining.pop(0)
                    else:
                        break
                else:
                    kf = mid

            pt = self._evaluate(kf)
            reward = _safety_reward(pt, ceiling)

            # Update bracket
            if _is_safe(pt, ceiling):
                if k_safe_best is None or kf < k_safe_best:
                    k_safe_best = kf
            elif not pt.failed:
                if pt.ppl_ratio <= reject_margin * 1.5:
                    if k_unsafe is None or kf > k_unsafe:
                        k_unsafe = kf

            # Update the arm that was used
            if action == "explore":
                explore_arm.update(reward)
            else:
                exploit_arm.update(reward)

            # Early rejection (explore path only): prune remaining candidates
            # that would be even more aggressive than a point already far over
            # the ceiling.  Preserves the over-ceiling-extras heuristic.
            if (
                action == "explore"
                and not pt.failed
                and pt.ppl_ratio > reject_margin
                and kf < 0.80
            ):
                if k_unsafe is None or kf > k_unsafe:
                    k_unsafe = kf

                close_to_ceiling = pt.ppl_ratio <= ceiling * 1.10
                helpful_count = 0
                total_improve = 0.0
                if pt.compile_result is not None:
                    for s in pt.compile_result.stage_stats:
                        if s.get("repair_helpful", False):
                            helpful_count += 1
                        total_improve += s.get("improve_frac", 0.0)
                repair_shows_promise = helpful_count >= 1 or total_improve >= 0.005
                speed_improving = (
                    pt.prefill_speedup >= _prev_prefill_speedup + _MIN_PREFILL_DELTA_FOR_EXTRA
                )

                if (
                    close_to_ceiling
                    and repair_shows_promise
                    and speed_improving
                    and _over_ceiling_extras < _MAX_OVER_CEILING_EXTRAS
                ):
                    _over_ceiling_extras += 1
                    _log.info(
                        "[search] window: continue over-ceiling "
                        "(ratio=%.3f, helpful=%d, improve=%.3f, "
                        "prefill_delta=%.3f) at kf=%.3f (%d/%d extra)",
                        pt.ppl_ratio, helpful_count, total_improve,
                        pt.prefill_speedup - _prev_prefill_speedup, kf,
                        _over_ceiling_extras, _MAX_OVER_CEILING_EXTRAS,
                    )
                else:
                    reason = "far_from_ceiling" if not close_to_ceiling else (
                        "no_helpful_repair" if not repair_shows_promise else (
                            "no_speed_gain" if not speed_improving else "cap_reached"
                        )
                    )
                    _log.info(
                        "[search] window: pruning remaining below kf=%.3f "
                        "(reason=%s, ratio=%.3f)",
                        kf, reason, pt.ppl_ratio,
                    )
                    remaining = [r for r in remaining if r > kf]

            _prev_prefill_speedup = pt.prefill_speedup

            _log.info(
                "[search] iter=%d action=%s kf=%.3f reward=%.3f "
                "arms=(explore=%.3f, exploit=%.3f) bracket=[%s, %s]",
                iteration, action, kf, reward,
                explore_arm.mean, exploit_arm.mean,
                f"{k_unsafe:.3f}" if k_unsafe is not None else "?",
                f"{k_safe_best:.3f}" if k_safe_best is not None else "?",
            )

        # ── Selection phase ──────────────────────────────────────────────
        safe = self._safe_points()
        viable = [p for p in self.evaluated if not p.failed]

        if not safe:
            _log.warning(
                "no point met quality ceiling (%.2fx) within budget; "
                "emitting conservative only",
                ceiling,
            )
            if viable:
                best_viable = min(viable, key=lambda p: p.ppl_ratio)
                best_viable.label = "frontier_0_default"
                return [best_viable]
            _log.error("no viable points at all")
            return []

        # Build selection: best quality, fastest safe, then fill
        best_q = self._best_quality()
        fastest_s = self._fastest_safe()
        selected_set: Dict[float, FrontierPoint] = {}

        if best_q is not None:
            selected_set[best_q.keep_frac] = best_q
        if fastest_s is not None and fastest_s.keep_frac not in selected_set:
            selected_set[fastest_s.keep_frac] = fastest_s

        # Fill remaining slots from safe points sorted by descending blended speedup
        safe_by_speed = sorted(safe, key=lambda p: -p.blended_speedup)
        for pt in safe_by_speed:
            if len(selected_set) >= self.n_frontier:
                break
            if pt.keep_frac not in selected_set:
                selected_set[pt.keep_frac] = pt

        selected = sorted(selected_set.values(), key=lambda p: -p.keep_frac)

        # Apply target speedup filter (uses blended metric)
        if self.target_prefill_speedup is not None:
            filtered = [p for p in selected if p.blended_speedup >= self.target_prefill_speedup]
            if filtered:
                selected = filtered

        selected = selected[:self.n_frontier]

        # Assign semantic labels (never label above-ceiling as named tier)
        _assign_labels(selected, ceiling)

        _log.info(
            "TSS complete: %d evaluated, %d safe, %d selected  "
            "(risk=%.3f, ceiling=%.2fx, explore_arm=%.3f, exploit_arm=%.3f)",
            len(self.evaluated), len(safe), len(selected),
            self.risk_score, ceiling, explore_arm.mean, exploit_arm.mean,
        )
        return selected
