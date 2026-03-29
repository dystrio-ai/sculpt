#!/usr/bin/env python3
"""Compile MiniCPM-o tiers and push all to HuggingFace.

Compiles specified keep_frac values, saves them alongside any existing
frontier models, then uploads everything to HF.

Usage:
    python scripts/compile_and_push_minicpmo.py
"""

import gc
import json
import logging
import os
import sys
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("minicpmo_push")

MODEL_ID = "openbmb/MiniCPM-o-4_5"
OUTDIR = Path("/ephemeral/sculpt_minicpmo")
HF_ORG = "dystrio"
DEVICE = "cuda"
DTYPE_STR = "bf16"

TIERS_TO_COMPILE = [0.820, 0.700]

TIER_LABELS = {
    0.950: ("default", "5% compression — minimal quality loss"),
    0.900: ("production", "10% compression — best quality/size tradeoff"),
    0.820: ("throughput", "18% compression — moderate quality tradeoff"),
    0.700: ("aggressive", "30% compression — maximum compression"),
}

PROBE_RESULTS = {
    0.950: {"accuracy": 0.6377, "mmlu": 0.6700, "hellaswag": 0.7250, "arc": 0.4857},
    0.900: {"accuracy": 0.6374, "mmlu": 0.6400, "hellaswag": 0.7125, "arc": 0.5571},
    0.820: {"accuracy": 0.5709, "mmlu": 0.5400, "hellaswag": 0.6750, "arc": 0.5286},
    0.700: {"accuracy": 0.5241, "mmlu": 0.5500, "hellaswag": 0.5250, "arc": 0.4714},
}

BASELINE = {"accuracy": 0.6756, "mmlu": 0.6700, "hellaswag": 0.7625, "arc": 0.6000}


def compile_tier(keep_frac: float, adapter):
    """Compile a single tier and save it."""
    from dystrio_sculpt.engine import compile_model
    from dystrio_sculpt._data import CalibConfig, calib_config_for_workload, load_text_sets
    from dystrio_sculpt.policy import build_policy_ladder
    from dystrio_sculpt.emit import emit_frontier_point
    from dystrio_sculpt.validate import validate_saved_model

    label_name, _ = TIER_LABELS[keep_frac]
    tier_dir = OUTDIR / f"tier_{label_name}"

    if (tier_dir / "model" / "config.json").exists():
        log.info("tier %s (kf=%.3f) already exists — skipping", label_name, keep_frac)
        return tier_dir

    log.info("compiling kf=%.3f (%s)", keep_frac, label_name)

    calib_cfg = calib_config_for_workload("multimodal_description")
    texts = load_text_sets(
        n_cal=400, n_train=2500, n_eval=300,
        calib=calib_cfg, mixture_workload="multimodal_description",
    )

    param_b = 8.0
    ladder = build_policy_ladder(param_b)
    policy = ladder[0]

    result = compile_model(
        MODEL_ID,
        keep_frac,
        texts=texts,
        policy=policy,
        device=DEVICE,
        dtype_str=DTYPE_STR,
        seed=0,
        deterministic=False,
        calib=calib_cfg,
        distill=True,
        distill_alpha_override=0.5,
        distill_cache=False,
        adapter=adapter,
    )

    if result.model is None or result.failure is not None:
        log.error("compile failed for kf=%.3f: %s", keep_frac, result.failure)
        return None

    tier_dir.mkdir(parents=True, exist_ok=True)
    model_dir = tier_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    from dystrio_sculpt._model import get_text_config, get_layers
    text_cfg = get_text_config(result.model)
    try:
        layers = get_layers(result.model)
        gate = layers[0].mlp.gate_proj
        new_intermediate = gate.out_features
        if text_cfg.intermediate_size != new_intermediate:
            log.info("patching intermediate_size: %d -> %d", text_cfg.intermediate_size, new_intermediate)
            text_cfg.intermediate_size = new_intermediate
            if hasattr(result.model.config, "text_config") and result.model.config.text_config is not None:
                result.model.config.text_config.intermediate_size = new_intermediate
    except Exception as e:
        log.warning("could not patch intermediate_size: %s", e)

    log.info("saving model to %s", model_dir)
    result.model.save_pretrained(str(model_dir))
    result.tokenizer.save_pretrained(str(model_dir))

    ok = validate_saved_model(model_dir, device=DEVICE, adapter=adapter)
    if not ok:
        log.error("validation FAILED for kf=%.3f", keep_frac)
        return None

    log.info("tier %s (kf=%.3f) saved and validated", label_name, keep_frac)

    del result
    gc.collect()
    torch.cuda.empty_cache()

    return tier_dir


def build_model_card(keep_frac: float) -> str:
    label_name, description = TIER_LABELS[keep_frac]
    probe = PROBE_RESULTS[keep_frac]
    retention = probe["accuracy"] / BASELINE["accuracy"] * 100

    return f"""---
license: apache-2.0
base_model: openbmb/MiniCPM-o-4_5
tags:
  - minicpm
  - multimodal
  - sculpt
  - structural-pruning
  - dystrio
---

# MiniCPM-o 4.5 — Sculpt {label_name.title()} (keep_frac={keep_frac})

{description}

Structurally pruned from [openbmb/MiniCPM-o-4_5](https://huggingface.co/openbmb/MiniCPM-o-4_5) using [Dystrio Sculpt](https://github.com/clusteroptimizerengine/BumbleB). Only the Qwen3-8B LLM backbone is pruned — vision (SigLip2), audio (Whisper), and TTS (CosyVoice2) modules are untouched.

## Quality (Downstream Probe — 250 questions)

| Metric | Baseline | This Model | Retention |
|--------|----------|------------|-----------|
| **Weighted Accuracy** | {BASELINE['accuracy']:.4f} | {probe['accuracy']:.4f} | {retention:.1f}% |
| MMLU | {BASELINE['mmlu']:.4f} | {probe['mmlu']:.4f} | {probe['mmlu']/BASELINE['mmlu']*100:.1f}% |
| HellaSwag | {BASELINE['hellaswag']:.4f} | {probe['hellaswag']:.4f} | {probe['hellaswag']/BASELINE['hellaswag']*100:.1f}% |
| ARC-Challenge | {BASELINE['arc']:.4f} | {probe['arc']:.4f} | {probe['arc']/BASELINE['arc']*100:.1f}% |

## Compression Details

- **keep_frac**: {keep_frac} ({(1-keep_frac)*100:.0f}% of MLP intermediate neurons removed)
- **Method**: Structural pruning with live teacher distillation (alpha=0.5)
- **Repair**: Full repair pass with workload-matched training data
- **Architecture**: All multimodal modules preserved; only LLM MLP layers compressed

## Intended Use

Drop-in replacement for MiniCPM-o 4.5 with reduced memory footprint. Suitable for:
- LoRA fine-tuning on memory-constrained GPUs
- File description and indexing workloads
- Multimodal inference with lower VRAM requirements

## How to Use

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "dystrio/MiniCPM-o-4_5-Sculpt-{label_name.title()}",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(
    "dystrio/MiniCPM-o-4_5-Sculpt-{label_name.title()}",
    trust_remote_code=True,
)
```
"""


def push_tier(model_dir: Path, keep_frac: float):
    from huggingface_hub import HfApi

    label_name, _ = TIER_LABELS[keep_frac]
    repo_id = f"{HF_ORG}/MiniCPM-o-4_5-Sculpt-{label_name.title()}"

    card = build_model_card(keep_frac)
    card_path = model_dir / "README.md"
    card_path.write_text(card)

    api = HfApi()
    api.create_repo(repo_id, exist_ok=True, repo_type="model")
    log.info("uploading %s -> %s", model_dir, repo_id)
    api.upload_folder(
        folder_path=str(model_dir),
        repo_id=repo_id,
        repo_type="model",
    )
    log.info("pushed %s", repo_id)
    return repo_id


def main():
    from dystrio_sculpt.architectures import fingerprint, get_adapter

    desc = fingerprint(MODEL_ID)
    adapter = get_adapter(desc)
    log.info("adapter: %s (family=%s)", type(adapter).__name__, desc.family)

    # Compile any missing tiers
    for kf in TIERS_TO_COMPILE:
        compile_tier(kf, adapter)

    # Push all tiers
    tier_dirs = {
        0.950: OUTDIR / "frontier_0_default" / "model",
        0.900: OUTDIR / "frontier_1_production" / "model",
        0.820: OUTDIR / "tier_throughput" / "model",
        0.700: OUTDIR / "tier_aggressive" / "model",
    }

    pushed = []
    for kf, model_dir in tier_dirs.items():
        if not (model_dir / "config.json").exists():
            log.warning("model dir %s not found — skipping kf=%.3f", model_dir, kf)
            continue
        repo_id = push_tier(model_dir, kf)
        pushed.append(repo_id)

    log.info("done! pushed %d models:", len(pushed))
    for r in pushed:
        log.info("  https://huggingface.co/%s", r)


if __name__ == "__main__":
    main()
