"""Repair policy ladder: auto-select stable repair hyperparameters per model."""

from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from ._model import load_model_and_tokenizer, resolve_dtype
from ._eval import eval_perplexity
from ._compile import compress_mlp_layer_swiglu_inplace
from .selectors import select_for_layer, BLOCK_SIZE

_log = logging.getLogger(__name__)

PILOT_KEEP_FRAC = 0.90
PILOT_PASS_EPS = 0.02
MAX_LEN = 256


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


def _estimate_param_billions(model) -> float:
    total = sum(p.numel() for p in model.parameters())
    return total / 1e9


def _scale_steps(base_steps: int, param_b: float) -> int:
    """Scale repair steps by model size: ~450 for <=1B, ~900 for ~7B."""
    if param_b <= 1.5:
        return base_steps
    scale = min(3.0, 1.0 + (param_b - 1.0) / 6.0)
    return int(base_steps * scale)


def _scale_patience(base_patience: int, param_b: float) -> int:
    if param_b >= 5.0:
        return max(base_patience, int(base_patience * 1.5))
    return base_patience


def build_policy_ladder(param_b: float) -> List[RepairPolicy]:
    """Return the ordered policy ladder (most aggressive → most conservative)."""
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


def _risk_aware_ladder_start(ladder: List[RepairPolicy], risk: float) -> int:
    """Return the index into *ladder* at which the pilot should begin.

    High risk => skip aggressive policies and start conservative.
    """
    if risk >= 0.65:
        return min(2, len(ladder) - 1)  # start at ss2_lr5e-5
    if risk <= 0.35:
        return 0  # start at ss4_lr1e-4
    return min(1, len(ladder) - 1)  # start at ss4_lr5e-5


def risk_scale_policy(policy: RepairPolicy, risk: float) -> RepairPolicy:
    """Return a copy of *policy* with steps/eval budget scaled by risk."""
    if risk < 0.55:
        return policy
    scale = 1.0 + (risk - 0.55) * 0.6  # up to ~1.3x at risk=1.0
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
) -> tuple[RepairPolicy, Dict[str, Any]]:
    """Run short pilot compiles to choose the most aggressive stable policy.

    When *risk_score* is provided, the pilot starts further down the ladder
    for high-risk models (skipping aggressive policies that are likely to fail).

    Returns (chosen_policy, pilot_report).
    """
    from .repair import repair_layers

    dtype = resolve_dtype(dtype_str)
    rng = np.random.RandomState(seed) if deterministic else None

    # Load a fresh model to measure size and determine pilot layer
    probe_model, probe_tok = load_model_and_tokenizer(model_id, device, dtype)
    param_b = _estimate_param_billions(probe_model)
    num_layers = probe_model.config.num_hidden_layers
    pilot_layer = num_layers // 2
    _log.info(
        "policy pilot: %.2fB params, %d layers, pilot_layer=%d, risk=%.3f",
        param_b, num_layers, pilot_layer, risk_score,
    )
    del probe_model, probe_tok
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    ladder = build_policy_ladder(param_b)
    start_idx = _risk_aware_ladder_start(ladder, risk_score)
    pilot_report: Dict[str, Any] = {
        "param_billions": round(param_b, 3),
        "pilot_layer": pilot_layer,
        "pilot_keep_frac": PILOT_KEEP_FRAC,
        "risk_score": round(risk_score, 4),
        "ladder_start_idx": start_idx,
        "trials": [],
    }

    pilot_steps = min(200, ladder[0].steps // 2)
    eval_subset = list(texts_eval[:64])

    for policy in ladder[start_idx:]:
        _log.info("pilot trial: %s", policy.name)

        model, tok = load_model_and_tokenizer(model_id, device, dtype)

        # Compress the pilot layer
        kept_blocks, kept_idx, _ = select_for_layer(
            model, tok, pilot_layer, texts_cal, PILOT_KEEP_FRAC,
            MAX_LEN, device, selector=selector,
            prescan_cache=prescan_cache, rng=rng,
        )
        compress_mlp_layer_swiglu_inplace(model, pilot_layer, kept_idx, dtype, device)

        pre_ppl = eval_perplexity(model, tok, eval_subset, MAX_LEN, device, 3000)
        _log.info("  pre-repair ppl=%.2f", pre_ppl)

        if math.isnan(pre_ppl) or math.isinf(pre_ppl):
            pilot_report["trials"].append({
                "policy": policy.name, "pre_ppl": pre_ppl, "post_ppl": None,
                "passed": False, "reason": "pre_ppl_nan_inf",
            })
            del model, tok
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        def _curve_fn(opt_step: int) -> Dict[str, float]:
            return {
                "ppl_w103_valid": eval_perplexity(
                    model, tok, eval_subset, MAX_LEN, device, 3000,
                ),
            }

        warmup = min(30, pilot_steps // 5)
        result = repair_layers(
            model=model, tokenizer=tok, texts_train=texts_train,
            layers=[pilot_layer], steps=pilot_steps, lr=policy.lr,
            warmup=warmup, weight_decay=0.01,
            max_len=MAX_LEN, device=device,
            grad_accum_steps=policy.grad_accum_steps,
            curve_fn=_curve_fn, curve_every=policy.curve_every,
            early_stop_patience=policy.early_stop_patience,
            regression_limit=policy.regression_limit,
            save_best=True, pre_repair_metric=pre_ppl,
        )

        post_ppl = eval_perplexity(model, tok, eval_subset, MAX_LEN, device, 3000)
        _log.info(
            "  post-repair ppl=%.2f  best=%.2f  repaired_ok=%s",
            post_ppl, result.get("best_metric", post_ppl), result.get("repaired_ok", True),
        )

        nan_inf = math.isnan(post_ppl) or math.isinf(post_ppl)
        within_eps = post_ppl <= pre_ppl * (1.0 + PILOT_PASS_EPS)
        passed = (not nan_inf) and within_eps and result.get("repaired_ok", True)

        trial_info = {
            "policy": policy.name,
            "pre_ppl": round(pre_ppl, 4),
            "post_ppl": round(post_ppl, 4),
            "best_metric": round(result.get("best_metric", post_ppl), 4),
            "best_step": result.get("best_step", 0),
            "steps_used": int(result["steps"]),
            "passed": passed,
            "reason": "ok" if passed else (
                "nan_inf" if nan_inf else "regression"
            ),
        }
        pilot_report["trials"].append(trial_info)

        del model, tok
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if passed:
            chosen = risk_scale_policy(policy, risk_score)
            _log.info("policy selected: %s (risk-scaled)", chosen.name)
            pilot_report["selected"] = chosen.name
            return chosen, pilot_report

    # All failed — return most conservative, risk-scaled
    fallback = risk_scale_policy(ladder[-1], risk_score)
    _log.warning("all pilot trials failed; falling back to %s", fallback.name)
    pilot_report["selected"] = fallback.name
    pilot_report["fallback"] = True
    return fallback, pilot_report
