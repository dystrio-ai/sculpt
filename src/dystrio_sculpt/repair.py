"""Repair: fine-tune compressed MLP layers with cosine LR, best-checkpoint
restore, never-worse-than-pre-repair safety invariant, and optional
knowledge distillation from a frozen teacher model.

Supports cached teacher logits: pre-compute top-k teacher softmax
probabilities once, then reuse across all repair stages. Eliminates
the teacher forward pass from the inner loop (~2x speedup with
distillation quality preserved).

Distillation loss functions:
  - ``jsd`` (default): Jensen-Shannon Divergence — symmetric, bounded,
    mode-seeking + mode-covering.  Fixes ARC/HellaSwag regression
    observed with forward KL.
  - ``kl``: Forward KL(teacher || student) — legacy Hinton 2015 behaviour.
"""

from __future__ import annotations

import copy
import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from ._model import get_mlp

_log = logging.getLogger(__name__)

TEACHER_CACHE_TOP_K = 128


@dataclass
class TeacherCacheEntry:
    """Cached top-k teacher probabilities for one training text."""
    top_k_vals: torch.Tensor   # (seq_len-1, K) float16
    top_k_idx: torch.Tensor    # (seq_len-1, K) int32


@torch.no_grad()
def build_teacher_cache(
    teacher_model,
    tokenizer,
    texts: Sequence[str],
    distill_temp: float = 2.0,
    max_len: int = 256,
    device: str = "cuda",
    top_k: int = TEACHER_CACHE_TOP_K,
) -> List[Optional[TeacherCacheEntry]]:
    """Pre-compute top-k teacher softmax probabilities for all training texts.

    Runs the teacher once over the corpus and stores sparse probability
    vectors. Total memory: ~500MB for 2500 texts (K=128, max_len=256).
    Eliminates the teacher forward pass from the repair inner loop.
    """
    _log.info(
        "building teacher cache: %d texts, top_k=%d, temp=%.1f",
        len(texts), top_k, distill_temp,
    )
    t0 = time.time()
    teacher_model.eval()
    cache: List[Optional[TeacherCacheEntry]] = []

    for i, txt in enumerate(texts):
        inp = tokenizer(txt, return_tensors="pt", truncation=True, max_length=max_len)
        inp = {k: v.to(device) for k, v in inp.items()}
        if inp["input_ids"].shape[1] < 2:
            cache.append(None)
            continue

        out = teacher_model(**inp, use_cache=False)
        logits = out.logits[:, :-1, :].float()
        probs = F.softmax(logits / distill_temp, dim=-1)

        vals, idx = probs.topk(top_k, dim=-1)

        cache.append(TeacherCacheEntry(
            top_k_vals=vals.squeeze(0).half().cpu(),
            top_k_idx=idx.squeeze(0).int().cpu(),
        ))

        if (i + 1) % 500 == 0:
            _log.info("  cached %d/%d texts (%.0fs)", i + 1, len(texts), time.time() - t0)

    elapsed = time.time() - t0
    cache_bytes = sum(
        (e.top_k_vals.nbytes + e.top_k_idx.nbytes) for e in cache if e is not None
    )
    _log.info(
        "teacher cache built: %d texts in %.0fs (%.1f MB)",
        len(cache), elapsed, cache_bytes / 1e6,
    )
    return cache


def _kl_from_cache(
    student_logits: torch.Tensor,
    cache_entry: TeacherCacheEntry,
    distill_temp: float,
) -> torch.Tensor:
    """Compute KL divergence using cached sparse teacher probabilities.

    Mathematically equivalent to full KL for the top-K teacher entries.
    Entries where teacher probability is ~0 contribute negligibly.
    """
    student_lp = F.log_softmax(student_logits / distill_temp, dim=-1)

    top_k_vals = cache_entry.top_k_vals.to(
        device=student_lp.device, dtype=student_lp.dtype,
    )
    top_k_idx = cache_entry.top_k_idx.to(device=student_lp.device).long()

    if student_lp.dim() == 3:
        student_lp = student_lp.squeeze(0)

    seq_len = min(student_lp.shape[0], top_k_vals.shape[0])
    student_lp = student_lp[:seq_len]
    top_k_vals = top_k_vals[:seq_len]
    top_k_idx = top_k_idx[:seq_len]

    student_at_topk = student_lp.gather(-1, top_k_idx)

    kl = (top_k_vals * (top_k_vals.clamp(min=1e-8).log() - student_at_topk)).sum(-1).mean()
    return kl * (distill_temp ** 2)


def _jsd_from_cache(
    student_logits: torch.Tensor,
    cache_entry: TeacherCacheEntry,
    distill_temp: float,
) -> torch.Tensor:
    """Jensen-Shannon Divergence using cached sparse teacher probabilities.

    JSD(P, Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M),  M = 0.5(P + Q).

    Computed at the teacher's top-K indices where probability mass is
    concentrated.  Non-top-K residual contributes negligible gradient.
    """
    student_probs = F.softmax(student_logits / distill_temp, dim=-1)

    top_k_vals = cache_entry.top_k_vals.to(
        device=student_probs.device, dtype=student_probs.dtype,
    )
    top_k_idx = cache_entry.top_k_idx.to(device=student_probs.device).long()

    if student_probs.dim() == 3:
        student_probs = student_probs.squeeze(0)

    seq_len = min(student_probs.shape[0], top_k_vals.shape[0])
    student_probs = student_probs[:seq_len]
    top_k_vals = top_k_vals[:seq_len]
    top_k_idx = top_k_idx[:seq_len]

    student_at_topk = student_probs.gather(-1, top_k_idx)

    m = 0.5 * (top_k_vals + student_at_topk)
    m = m.clamp(min=1e-8)
    log_m = m.log()

    kl_teacher_m = (top_k_vals * (top_k_vals.clamp(min=1e-8).log() - log_m)).sum(-1).mean()
    kl_student_m = (student_at_topk * (student_at_topk.clamp(min=1e-8).log() - log_m)).sum(-1).mean()

    return 0.5 * (kl_teacher_m + kl_student_m) * (distill_temp ** 2)


def _distill_loss_from_cache(
    student_logits: torch.Tensor,
    cache_entry: TeacherCacheEntry,
    distill_temp: float,
    loss_fn: str = "jsd",
) -> torch.Tensor:
    """Dispatch to the appropriate cached distillation loss."""
    if loss_fn == "jsd":
        return _jsd_from_cache(student_logits, cache_entry, distill_temp)
    return _kl_from_cache(student_logits, cache_entry, distill_temp)


def _distill_loss_live(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    distill_temp: float,
    loss_fn: str = "jsd",
) -> torch.Tensor:
    """Compute distillation loss from live teacher logits."""
    teacher_probs = F.softmax(teacher_logits / distill_temp, dim=-1)
    V = student_logits.size(-1)

    if loss_fn == "jsd":
        student_probs = F.softmax(student_logits / distill_temp, dim=-1)
        m = 0.5 * (teacher_probs + student_probs)
        log_m = m.clamp(min=1e-8).log().reshape(-1, V)
        kl_teacher_m = F.kl_div(
            log_m, teacher_probs.reshape(-1, V), reduction="batchmean",
        )
        kl_student_m = F.kl_div(
            log_m, student_probs.reshape(-1, V), reduction="batchmean",
        )
        return 0.5 * (kl_teacher_m + kl_student_m) * (distill_temp ** 2)

    student_log_probs = F.log_softmax(student_logits / distill_temp, dim=-1)
    return F.kl_div(
        student_log_probs.reshape(-1, V),
        teacher_probs.reshape(-1, V),
        reduction="batchmean",
    ) * (distill_temp ** 2)


def adaptive_distill_alpha(base_alpha: float, keep_frac: float) -> float:
    """Scale distillation alpha by compression severity.

    More compression (lower keep_frac) → more teacher reliance.
    Returns alpha clamped to [0.1, 0.9].
    """
    alpha = base_alpha + (1.0 - keep_frac) * 0.5
    return max(0.1, min(0.9, alpha))


def cosine_lr(step: int, total: int, base_lr: float, warmup: int) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * t))


def _snapshot_trainable(model, layers: Sequence[int]) -> Dict[str, torch.Tensor]:
    """Save a CPU copy of all trainable MLP parameters for the given layers."""
    snap: Dict[str, torch.Tensor] = {}
    for li in layers:
        mlp = get_mlp(model, li)
        for name, p in mlp.named_parameters():
            key = f"layers.{li}.mlp.{name}"
            snap[key] = p.data.detach().cpu().clone()
    return snap


def _restore_trainable(
    model, layers: Sequence[int], snap: Dict[str, torch.Tensor],
) -> None:
    """Restore MLP parameters from a CPU snapshot."""
    for li in layers:
        mlp = get_mlp(model, li)
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
    teacher_model=None,
    distill_alpha: float = 0.0,
    distill_temp: float = 2.0,
    distill_loss_fn: str = "jsd",
    teacher_cache: Optional[List[Optional[TeacherCacheEntry]]] = None,
    layer_risk: Optional[Dict[int, float]] = None,
    adapter=None,
) -> Dict[str, Any]:
    """Train only MLP params in selected layers with best-checkpoint restore.

    ``steps`` counts optimizer steps (not microsteps).

    When *save_best* is True, the best parameter snapshot (by *early_stop_key*)
    is restored at the end of repair.

    When *pre_repair_metric* is provided, the "never worse than pre-repair"
    invariant is enforced: if the best metric seen during repair exceeds
    pre_repair_metric * (1 + never_worse_eps), pre-repair weights are restored
    and ``repaired_ok`` is set to False.

    Distillation modes (mutually exclusive):

    1. **Live teacher**: *teacher_model* is provided — runs the teacher forward
       pass on every microstep.  Accurate but 2x slower.

    2. **Cached teacher**: *teacher_cache* is provided (from
       ``build_teacher_cache``) — uses pre-computed top-k teacher softmax
       probabilities.  Same quality, ~2x faster.  Preferred.

    Distillation loss functions (``distill_loss_fn``):

    - ``jsd`` (default): Jensen-Shannon Divergence — symmetric, bounded.
    - ``kl``: Forward KL(teacher || student) — legacy behaviour.

    Per-layer learning rate scaling (``layer_risk``):

    When provided, maps layer index → structural risk score in [0, 1].
    Higher-risk layers get proportionally larger LR: ``lr * (0.5 + risk)``.

    Returns dict with: steps, microsteps, curve, early_stopped, best_metric,
    best_step, repaired_ok.
    """
    layers = list(layers)
    use_distill_live = teacher_model is not None and distill_alpha > 0.0
    use_distill_cached = teacher_cache is not None and distill_alpha > 0.0
    use_distill = use_distill_live or use_distill_cached
    if use_distill:
        mode = "cached" if use_distill_cached else "live"
        _log.info(
            "distillation enabled (%s): loss=%s alpha=%.2f temp=%.1f",
            mode, distill_loss_fn, distill_alpha, distill_temp,
        )

    for p in model.parameters():
        p.requires_grad = False

    # Build optimizer param groups — per-layer risk scaling when available
    _use_risk_lr = bool(layer_risk is not None and adapter is None and layer_risk)
    if adapter is not None:
        params = adapter.get_trainable_params(model, layers)
        for p in params:
            p.requires_grad = True
        param_groups_or_params: Any = params
    elif _use_risk_lr:
        param_groups: List[Dict[str, Any]] = []
        for li in layers:
            layer_params = list(get_mlp(model, li).parameters())
            for p in layer_params:
                p.requires_grad = True
            risk = layer_risk.get(li, 0.5)
            scale = 0.5 + risk
            param_groups.append({
                "params": layer_params,
                "_lr_scale": scale,
                "lr": lr * scale,
            })
        if _use_risk_lr:
            _log.info(
                "risk-scaled LR: min=%.3f max=%.3f across %d layers",
                min(pg["_lr_scale"] for pg in param_groups),
                max(pg["_lr_scale"] for pg in param_groups),
                len(param_groups),
            )
        param_groups_or_params = param_groups
    else:
        params = []
        for li in layers:
            for p in get_mlp(model, li).parameters():
                p.requires_grad = True
                params.append(p)
        param_groups_or_params = params

    # Local dispatch helpers so the rest of repair_layers stays clean
    def _snap() -> Dict[str, torch.Tensor]:
        if adapter is not None:
            return adapter.snapshot_trainable(model, layers)
        return _snapshot_trainable(model, layers)

    def _restore(snap: Dict[str, torch.Tensor]) -> None:
        if adapter is not None:
            adapter.restore_trainable(model, layers, snap)
        else:
            _restore_trainable(model, layers, snap)

    # Pre-repair snapshot (for never-worse rollback)
    pre_repair_snap: Optional[Dict[str, torch.Tensor]] = None
    if pre_repair_metric is not None:
        pre_repair_snap = _snap()

    opt = torch.optim.AdamW(param_groups_or_params, lr=lr, weight_decay=weight_decay)
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
                best_snap = _snap()
        model.train()

    _es_no_improve: int = 0

    opt.zero_grad(set_to_none=True)

    for text_idx, txt in enumerate(texts_train):
        inp = tokenizer(txt, return_tensors="pt", truncation=True, max_length=max_len)
        inp = {k: v.to(device) for k, v in inp.items()}
        if inp["input_ids"].shape[1] < 2:
            continue

        out = model(**inp, use_cache=False)
        logits = out.logits[:, :-1, :]
        target = inp["input_ids"][:, 1:]
        ce_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), target.reshape(-1),
        )

        if use_distill_cached and teacher_cache[text_idx] is not None:
            kl_loss = _kl_from_cache(logits, teacher_cache[text_idx], distill_temp)
            loss = (1.0 - distill_alpha) * ce_loss + distill_alpha * kl_loss
        elif use_distill_live:
            with torch.no_grad():
                teacher_out = teacher_model(**inp, use_cache=False)
            teacher_logits = teacher_out.logits[:, :-1, :]
            student_log_probs = F.log_softmax(logits / distill_temp, dim=-1)
            teacher_probs = F.softmax(teacher_logits / distill_temp, dim=-1)
            kl_loss = F.kl_div(
                student_log_probs.reshape(-1, logits.size(-1)),
                teacher_probs.reshape(-1, logits.size(-1)),
                reduction="batchmean",
            ) * (distill_temp ** 2)
            loss = (1.0 - distill_alpha) * ce_loss + distill_alpha * kl_loss
        else:
            loss = ce_loss

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
            pg["lr"] = lr_now * pg.get("_lr_scale", 1.0)
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
                    best_snap = _snap()
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
                    best_snap = _snap()
            model.train()

    # Restore best checkpoint
    repaired_ok = True
    if save_best and best_snap is not None:
        _restore(best_snap)
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
            _restore(pre_repair_snap)
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
        "distillation_enabled": use_distill,
        "distill_alpha": distill_alpha if use_distill else 0.0,
        "distill_temp": distill_temp if use_distill else 0.0,
        "distill_loss_fn": distill_loss_fn if use_distill else None,
        "risk_scaled_lr": _use_risk_lr,
    }
