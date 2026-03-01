"""Frontier search: adaptive binary refinement over uniform keep_frac.

Integrates RepairPolicy auto-selection. Uses cheap eval during search
and only runs final eval on selected Pareto points.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from ._model import load_model_and_tokenizer, resolve_dtype
from ._data import load_text_sets
from .engine import CompileResult, compile_model, _collect_metrics, setup_determinism, MAX_LEN
from .policy import RepairPolicy, auto_select_policy
from .selectors import BLOCK_SIZE
from .selectors.structural import prescan_structural_artifacts

_log = logging.getLogger(__name__)

FRONTIER_LABELS = ["conservative", "balanced", "aggressive", "extreme"]


def frontier_label(index: int, total: int) -> str:
    """Return a human-friendly label for a frontier point."""
    if total <= len(FRONTIER_LABELS):
        return FRONTIER_LABELS[index] if index < len(FRONTIER_LABELS) else f"point{index}"
    return f"point{index}"


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
    compile_result: Optional[CompileResult] = field(default=None, repr=False)
    label: str = ""
    failed: bool = False
    failure_reason: str = ""


def pareto_front(points: List[FrontierPoint]) -> List[FrontierPoint]:
    """Non-dominated sort: minimize PPL, maximize prefill speedup.

    Returns points sorted by ascending PPL where each successive point
    has strictly higher speedup than all predecessors.
    """
    viable = [p for p in points if not p.failed]
    if not viable:
        return []
    sorted_pts = sorted(viable, key=lambda p: p.ppl_w103)
    front = [sorted_pts[0]]
    max_speedup = sorted_pts[0].prefill_speedup
    for p in sorted_pts[1:]:
        if p.prefill_speedup > max_speedup:
            front.append(p)
            max_speedup = p.prefill_speedup
    return front


def select_evenly_spaced(front: List[FrontierPoint], n: int) -> List[FrontierPoint]:
    """Pick n evenly-spaced points from the sorted Pareto front."""
    if len(front) <= n:
        return list(front)
    indices = sorted(set(
        int(round(i * (len(front) - 1) / (n - 1))) for i in range(n)
    ))
    return [front[i] for i in indices]


class FrontierSearch:
    """Adaptive frontier search over uniform keep_frac in [0.4, 1.0].

    Algorithm:
      1. Auto-select a stable RepairPolicy via pilot compile.
      2. Evaluate an initial grid of keep_frac values.
      3. Build Pareto front.
      4. Bisect the largest gap on the front until resolution < 3% or time runs out.
      5. Apply user constraints (max PPL, target speedup).
      6. Select n_frontier evenly-spaced non-dominated points.
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
    ):
        self.model_id = model_id
        self.n_frontier = n_frontier
        self.max_ppl_multiplier = max_ppl_multiplier
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

        self.texts: Optional[Dict[str, List[str]]] = None
        self.prescan_cache: Optional[Dict[int, Dict[str, Any]]] = None
        self.baseline_metrics: Optional[Dict[str, float]] = None
        self.policy: Optional[RepairPolicy] = None
        self.pilot_report: Optional[Dict[str, Any]] = None
        self.evaluated: List[FrontierPoint] = []
        self._start_time: float = 0.0

    def _time_exceeded(self) -> bool:
        if self.max_compile_hours is None:
            return False
        elapsed_h = (time.time() - self._start_time) / 3600
        return elapsed_h >= self.max_compile_hours

    def _setup(self) -> None:
        self._start_time = time.time()
        setup_determinism(self.seed, self.deterministic)

        _log.info("loading datasets")
        self.texts = load_text_sets(self.n_texts_cal, self.n_texts_train, self.n_texts_eval)

    def _compute_baseline(self) -> None:
        _log.info("computing baseline (no compression)")
        setup_determinism(self.seed, self.deterministic)
        dtype = resolve_dtype(self.dtype_str)
        model, tok = load_model_and_tokenizer(self.model_id, self.device, dtype)
        assert self.texts is not None
        self.baseline_metrics = _collect_metrics(
            model, tok, self.texts, self.device, self.max_eval_tokens,
        )
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

    def _select_policy(self) -> None:
        if self.policy_override is not None:
            self.policy = self.policy_override
            _log.info("using user-supplied policy override: %s", self.policy.name)
            return

        assert self.texts is not None
        _log.info("running policy auto-selection pilot")
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
        )
        _log.info("policy auto-selected: %s", self.policy.name)

    def _evaluate(self, keep_frac: float) -> FrontierPoint:
        _log.info("evaluating keep_frac=%.3f", keep_frac)
        assert self.texts is not None
        assert self.baseline_metrics is not None
        assert self.policy is not None

        failure_subdir = (self.outdir / f"_failed_kf{keep_frac:.3f}") if self.outdir else None

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
            )
        except Exception as exc:
            _log.error("compile_model failed for kf=%.3f: %s", keep_frac, exc)
            point = FrontierPoint(
                keep_frac=keep_frac, ppl_w103=float("inf"), ppl_w2=float("inf"),
                prefill_tps=0.0, decode_tps=0.0, prefill_speedup=0.0,
                decode_speedup=0.0, wall_time_s=0.0, failed=True,
                failure_reason=str(exc),
            )
            self.evaluated.append(point)
            return point

        m = result.metrics_post
        base = self.baseline_metrics
        prefill_speedup = m["prefill_tokens_per_sec"] / max(1e-9, base["prefill_tokens_per_sec"])
        decode_speedup = m["decode_tokens_per_sec"] / max(1e-9, base["decode_tokens_per_sec"])

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
            compile_result=result,
            failed=failed,
            failure_reason=result.failure["reason"] if result.failure else "",
        )
        self.evaluated.append(point)

        _log.info(
            "keep_frac=%.3f  ppl_w103=%.2f  prefill_speedup=%.2fx  (%.0fs)%s",
            keep_frac, point.ppl_w103, prefill_speedup, result.wall_time_s,
            " [FAILED]" if failed else "",
        )
        return point

    def _apply_constraints(self, points: List[FrontierPoint]) -> List[FrontierPoint]:
        filtered = list(points)
        if self.max_ppl_multiplier is not None and self.baseline_metrics is not None:
            max_ppl = self.baseline_metrics["ppl_w103_valid"] * self.max_ppl_multiplier
            filtered = [p for p in filtered if p.ppl_w103 <= max_ppl]
        if self.target_prefill_speedup is not None:
            filtered = [p for p in filtered if p.prefill_speedup >= self.target_prefill_speedup]
        return filtered

    def run(self) -> List[FrontierPoint]:
        """Execute the full frontier search. Returns selected frontier points."""
        self._setup()
        self._compute_baseline()
        self._compute_prescan()
        self._select_policy()

        # Initial grid
        initial_grid = [0.40, 0.55, 0.70, 0.85]
        for kf in initial_grid:
            if self._time_exceeded():
                _log.info("time budget reached during initial grid")
                break
            self._evaluate(kf)

        # Adaptive binary refinement
        min_resolution = 0.03
        max_refinements = 8
        for _ in range(max_refinements):
            if self._time_exceeded():
                _log.info("time budget reached during refinement")
                break
            front = pareto_front(self.evaluated)
            if len(front) < 2:
                break
            front.sort(key=lambda p: p.keep_frac)
            max_gap = 0.0
            best_pair = None
            for i in range(len(front) - 1):
                gap = front[i + 1].keep_frac - front[i].keep_frac
                if gap > max_gap:
                    max_gap = gap
                    best_pair = (front[i].keep_frac, front[i + 1].keep_frac)
            if best_pair is None or max_gap < min_resolution:
                break
            midpoint = round((best_pair[0] + best_pair[1]) / 2, 3)
            already = any(abs(p.keep_frac - midpoint) < 0.005 for p in self.evaluated)
            if already:
                break
            self._evaluate(midpoint)

        # Select frontier
        front = pareto_front(self.evaluated)
        constrained = self._apply_constraints(front)
        if not constrained:
            _log.warning("no points satisfy constraints; using full Pareto front")
            constrained = front

        selected = select_evenly_spaced(constrained, self.n_frontier)

        # Assign human-friendly labels
        for i, pt in enumerate(selected):
            pt.label = f"frontier_{i}_{frontier_label(i, len(selected))}"

        _log.info(
            "frontier search complete: %d evaluated, %d on Pareto front, %d selected",
            len(self.evaluated), len(front), len(selected),
        )
        return selected
