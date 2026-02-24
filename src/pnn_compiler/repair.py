"""Repair: per-text fine-tuning of compressed MLP layers with cosine LR."""

from __future__ import annotations

import math
import time
from typing import Any, Callable, Dict, Sequence

import torch


def cosine_lr(step: int, total: int, base_lr: float, warmup: int) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    t = (step - warmup) / max(1, (total - warmup))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * t))


def repair_layers(
    model,
    tokenizer,
    texts_train: Sequence[str],
    layers: Sequence[int],
    steps: int,
    lr: float,
    warmup: int,
    weight_decay: float,
    max_len: int,
    device: str,
    log_every: int = 500,
    grad_accum_steps: int = 1,
    curve_fn: Callable[[int], Dict[str, float]] | None = None,
    curve_every: int = 250,
    early_stop_patience: int = 0,
    early_stop_key: str = "ppl_w103_valid",
) -> Dict[str, Any]:
    """Train only MLP params in selected layers.

    Layers are already physically sliced, so no mask enforcement needed.

    ``steps`` counts **optimizer steps**, not microsteps.  With
    ``grad_accum_steps > 1``, gradients are accumulated over that many
    forward/backward passes before each optimizer update.

    If ``curve_fn`` is provided it is called at optimizer step 0 (pre-repair
    baseline), every ``curve_every`` optimizer steps, and once at the final
    step.  It receives the current optimizer-step index and must return a
    ``dict[str, float]`` of metrics.

    ``early_stop_patience`` (requires ``curve_fn``): stop if the metric
    given by ``early_stop_key`` has not improved for this many consecutive
    curve checkpoints.  0 disables early stopping.
    """
    for p in model.parameters():
        p.requires_grad = False

    params = []
    for li in layers:
        mlp = model.model.layers[li].mlp
        for p in mlp.parameters():
            p.requires_grad = True
            params.append(p)

    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    model.train()

    t0 = time.time()
    opt_step = 0
    microstep = 0
    accum_count = 0
    accum_loss_sum = 0.0
    curve_points: list[dict[str, Any]] = []
    early_stopped = False
    _es_best: float = float("inf")
    _es_no_improve: int = 0

    # Curve: evaluate at step 0 (pre-repair quality baseline)
    if curve_fn is not None:
        model.eval()
        pt = curve_fn(0)
        pt["opt_step"] = 0
        pt["wall_time_s"] = round(time.time() - t0, 2)
        curve_points.append(pt)
        if early_stop_patience > 0:
            _es_best = pt.get(early_stop_key, float("inf"))
        model.train()

    opt.zero_grad(set_to_none=True)

    for txt in texts_train:
        inp = tokenizer(txt, return_tensors="pt", truncation=True, max_length=max_len)
        inp = {k: v.to(device) for k, v in inp.items()}
        ids = inp["input_ids"]
        if ids.shape[1] < 2:
            continue

        out = model(**inp, use_cache=False)
        logits = out.logits[:, :-1, :]
        target = ids[:, 1:]

        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target.reshape(-1),
        )
        (loss / grad_accum_steps).backward()

        microstep += 1
        accum_count += 1
        accum_loss_sum += float(loss.detach().cpu())

        if accum_count < grad_accum_steps:
            continue

        # ── Optimizer step ─────────────────────────────────────────────────
        lr_now = cosine_lr(opt_step, steps, lr, warmup)
        for pg in opt.param_groups:
            pg["lr"] = lr_now

        opt.step()
        opt.zero_grad(set_to_none=True)
        opt_step += 1

        if opt_step % log_every == 0:
            dt = time.time() - t0
            avg_loss = accum_loss_sum / accum_count
            print(
                f"[repair] opt_step {opt_step}/{steps} "
                f"microstep {microstep} "
                f"loss {avg_loss:.4f} lr {lr_now:.2e} ({dt:.1f}s)"
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

            if early_stop_patience > 0:
                val = pt.get(early_stop_key, float("inf"))
                if val < _es_best:
                    _es_best = val
                    _es_no_improve = 0
                else:
                    _es_no_improve += 1
                if _es_no_improve >= early_stop_patience:
                    print(
                        f"[repair] Early stop at opt_step {opt_step}: "
                        f"{early_stop_key} did not improve for "
                        f"{early_stop_patience} consecutive checkpoints"
                    )
                    early_stopped = True
                    break

        if opt_step >= steps:
            break

    # Curve: final point if not already recorded at this opt_step
    if curve_fn is not None:
        if not curve_points or curve_points[-1]["opt_step"] != opt_step:
            model.eval()
            pt = curve_fn(opt_step)
            pt["opt_step"] = opt_step
            pt["wall_time_s"] = round(time.time() - t0, 2)
            curve_points.append(pt)
            model.train()

    return {
        "steps": float(opt_step),
        "microsteps": float(microstep),
        "curve": curve_points,
        "early_stopped": early_stopped,
    }
