"""Repair: fine-tune compressed MLP layers with cosine LR, best-checkpoint
restore, and never-worse-than-pre-repair safety invariant."""

from __future__ import annotations

import copy
import logging
import math
import time
from typing import Any, Callable, Dict, Optional, Sequence

import torch

_log = logging.getLogger(__name__)


def cosine_lr(step: int, total: int, base_lr: float, warmup: int) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * t))


def _snapshot_trainable(model, layers: Sequence[int]) -> Dict[str, torch.Tensor]:
    """Save a CPU copy of all trainable MLP parameters for the given layers."""
    snap: Dict[str, torch.Tensor] = {}
    for li in layers:
        mlp = model.model.layers[li].mlp
        for name, p in mlp.named_parameters():
            key = f"layers.{li}.mlp.{name}"
            snap[key] = p.data.detach().cpu().clone()
    return snap


def _restore_trainable(
    model, layers: Sequence[int], snap: Dict[str, torch.Tensor],
) -> None:
    """Restore MLP parameters from a CPU snapshot."""
    for li in layers:
        mlp = model.model.layers[li].mlp
        for name, p in mlp.named_parameters():
            key = f"layers.{li}.mlp.{name}"
            if key in snap:
                p.data.copy_(snap[key].to(p.device))


def repair_layers(
    model,
    tokenizer,
    texts_train: Sequence[str],
    layers: Sequence[int],
    steps: int,
    lr: float,
    warmup: int = 100,
    weight_decay: float = 0.01,
    max_len: int = 256,
    device: str = "cuda",
    log_every: int = 200,
    grad_accum_steps: int = 1,
    curve_fn: Callable[[int], Dict[str, float]] | None = None,
    curve_every: int = 100,
    early_stop_patience: int = 3,
    early_stop_key: str = "ppl_w103_valid",
    regression_limit: float = 0.0,
    max_grad_norm: Optional[float] = None,
    save_best: bool = True,
    pre_repair_metric: Optional[float] = None,
    never_worse_eps: float = 0.005,
) -> Dict[str, Any]:
    """Train only MLP params in selected layers with best-checkpoint restore.

    ``steps`` counts optimizer steps (not microsteps).

    When *save_best* is True, the best parameter snapshot (by *early_stop_key*)
    is restored at the end of repair.

    When *pre_repair_metric* is provided, the "never worse than pre-repair"
    invariant is enforced: if the best metric seen during repair exceeds
    pre_repair_metric * (1 + never_worse_eps), pre-repair weights are restored
    and ``repaired_ok`` is set to False.

    Returns dict with: steps, microsteps, curve, early_stopped, best_metric,
    best_step, repaired_ok.
    """
    layers = list(layers)
    for p in model.parameters():
        p.requires_grad = False
    params = []
    for li in layers:
        for p in model.model.layers[li].mlp.parameters():
            p.requires_grad = True
            params.append(p)

    # Pre-repair snapshot (for never-worse rollback)
    pre_repair_snap: Optional[Dict[str, torch.Tensor]] = None
    if pre_repair_metric is not None:
        pre_repair_snap = _snapshot_trainable(model, layers)

    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    model.train()

    t0 = time.time()
    opt_step = 0
    microstep = 0
    accum_count = 0
    accum_loss_sum = 0.0
    curve_points: list[dict[str, Any]] = []
    early_stopped = False
    _regression_tripwire = False
    _nan_inf_detected = False
    _early_stop_triggered = False

    best_metric: float = float("inf")
    best_step: int = 0
    best_snap: Optional[Dict[str, torch.Tensor]] = None

    # Step-0 eval
    if curve_fn is not None:
        model.eval()
        pt = curve_fn(0)
        pt["opt_step"] = 0
        pt["wall_time_s"] = round(time.time() - t0, 2)
        curve_points.append(pt)
        val = pt.get(early_stop_key, float("inf"))
        if not (math.isnan(val) or math.isinf(val)):
            best_metric = val
            best_step = 0
            if save_best:
                best_snap = _snapshot_trainable(model, layers)
        model.train()

    _es_no_improve: int = 0

    opt.zero_grad(set_to_none=True)

    for txt in texts_train:
        inp = tokenizer(txt, return_tensors="pt", truncation=True, max_length=max_len)
        inp = {k: v.to(device) for k, v in inp.items()}
        if inp["input_ids"].shape[1] < 2:
            continue

        out = model(**inp, use_cache=False)
        logits = out.logits[:, :-1, :]
        target = inp["input_ids"][:, 1:]
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), target.reshape(-1),
        )

        if torch.isnan(loss) or torch.isinf(loss):
            _log.warning("NaN/Inf loss at microstep %d — stopping repair", microstep)
            _nan_inf_detected = True
            early_stopped = True
            break

        (loss / grad_accum_steps).backward()
        microstep += 1
        accum_count += 1
        accum_loss_sum += float(loss.detach().cpu())

        if accum_count < grad_accum_steps:
            continue

        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(params, max_grad_norm)

        lr_now = cosine_lr(opt_step, steps, lr, warmup)
        for pg in opt.param_groups:
            pg["lr"] = lr_now
        opt.step()
        opt.zero_grad(set_to_none=True)
        opt_step += 1

        if opt_step % log_every == 0:
            avg_loss = accum_loss_sum / max(accum_count, 1)
            _log.info(
                "repair step %d/%d  loss=%.4f  lr=%.2e  (%.1fs)",
                opt_step, steps, avg_loss, lr_now, time.time() - t0,
            )
        accum_loss_sum = 0.0
        accum_count = 0

        if curve_fn is not None and opt_step % curve_every == 0:
            model.eval()
            pt = curve_fn(opt_step)
            pt["opt_step"] = opt_step
            pt["wall_time_s"] = round(time.time() - t0, 2)
            curve_points.append(pt)
            model.train()

            val = pt.get(early_stop_key, float("inf"))
            if math.isnan(val) or math.isinf(val):
                _log.warning("NaN/Inf metric at step %d — stopping", opt_step)
                _nan_inf_detected = True
                early_stopped = True
                break

            if val < best_metric:
                best_metric = val
                best_step = opt_step
                _es_no_improve = 0
                if save_best:
                    best_snap = _snapshot_trainable(model, layers)
            else:
                _es_no_improve += 1

            if early_stop_patience > 0 and _es_no_improve >= early_stop_patience:
                _log.info(
                    "early stop (patience) at step %d: %s no improvement for %d checkpoints",
                    opt_step, early_stop_key, early_stop_patience,
                )
                _early_stop_triggered = True
                early_stopped = True
                break

            if (
                regression_limit > 0
                and best_metric > 0
                and val > best_metric * (1.0 + regression_limit)
            ):
                _log.info(
                    "regression tripwire at step %d: %s=%.2f exceeds best %.2f by >%.0f%%",
                    opt_step, early_stop_key, val, best_metric, regression_limit * 100,
                )
                _regression_tripwire = True
                early_stopped = True
                break

        if opt_step >= steps:
            _early_stop_triggered = True
            break

    # Final curve point
    if curve_fn is not None:
        if not curve_points or curve_points[-1]["opt_step"] != opt_step:
            model.eval()
            pt = curve_fn(opt_step)
            pt["opt_step"] = opt_step
            pt["wall_time_s"] = round(time.time() - t0, 2)
            curve_points.append(pt)
            val = pt.get(early_stop_key, float("inf"))
            if not (math.isnan(val) or math.isinf(val)) and val < best_metric:
                best_metric = val
                best_step = opt_step
                if save_best:
                    best_snap = _snapshot_trainable(model, layers)
            model.train()

    # Restore best checkpoint
    repaired_ok = True
    if save_best and best_snap is not None:
        _restore_trainable(model, layers, best_snap)
        _log.info("restored best checkpoint from step %d (metric=%.4f)", best_step, best_metric)

    # Never-worse-than-pre-repair invariant
    if pre_repair_metric is not None and pre_repair_snap is not None:
        threshold = pre_repair_metric * (1.0 + never_worse_eps)
        if best_metric > threshold or math.isnan(best_metric) or math.isinf(best_metric):
            _log.warning(
                "repair failed to improve: best=%.4f > pre_repair=%.4f * (1+%.3f)=%.4f; "
                "rolling back to pre-repair weights",
                best_metric, pre_repair_metric, never_worse_eps, threshold,
            )
            _restore_trainable(model, layers, pre_repair_snap)
            repaired_ok = False

    return {
        "steps": float(opt_step),
        "microsteps": float(microstep),
        "curve": curve_points,
        "early_stopped": early_stopped,
        "best_metric": best_metric,
        "best_step": best_step,
        "repaired_ok": repaired_ok,
        "regression_tripwire_triggered": _regression_tripwire,
        "regression_stop_triggered": _regression_tripwire,  # backward compat
        "nan_inf_detected": _nan_inf_detected,
        "early_stop_triggered": _early_stop_triggered,
    }
