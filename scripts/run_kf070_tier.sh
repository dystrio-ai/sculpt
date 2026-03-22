#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Single-point compile at kf=0.700 + lm_eval + push to HF
#
# Usage:
#   cd ~/BumbleB && source .venv/bin/activate
#   export HF_TOKEN="..."
#   nohup bash scripts/run_kf070_tier.sh > kf070_run.log 2>&1 &
###############################################################################

MODEL="Qwen/Qwen3.5-9B"
OUTDIR="sculpt_out_qwen35_9b_kf070"
TASKS="arc_challenge,hellaswag,mmlu,truthfulqa_mc2,winogrande,gsm8k"
ORG="dystrio"
REPO="${ORG}/Qwen3.5-9B-Sculpt-Aggressive"

echo "============================================================"
echo "  Single-point compile: kf=0.700 + lm_eval + HF push"
echo "  $(date)"
echo "============================================================"

# ── 1. Compile at kf=0.700 ───────────────────────────────────────
echo ""
echo ">> [1/3] Compiling kf=0.700 with live teacher distillation..."
echo "   Started: $(date)"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 -c "
import sys, os, logging, torch
sys.path.insert(0, 'src')
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s [%(name)s] %(message)s')

from dystrio_sculpt.engine import compile_model
from dystrio_sculpt._data import CalibConfig
from dystrio_sculpt.emit import emit_frontier_point
from pathlib import Path

outdir = Path('${OUTDIR}')
outdir.mkdir(parents=True, exist_ok=True)

result = compile_model(
    '${MODEL}',
    keep_frac=0.700,
    device='cuda',
    dtype_str='bf16',
    distill=True,
    distill_alpha_override=0.5,
    distill_cache=False,
    calib=CalibConfig(
        dataset='wikitext',
        config='wikitext-103-raw-v1',
        split='train',
        text_field='text',
    ),
)

print(f'Compile done: ppl_ratio={result.metrics_post.get(\"ppl_w103_valid\", 0):.4f}')
print(f'Params: {result.num_params:,}')

# Save the model
model_dir = outdir / 'model'
model_dir.mkdir(parents=True, exist_ok=True)
result.model.save_pretrained(str(model_dir))
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('${MODEL}')
tok.save_pretrained(str(model_dir))
print(f'Model saved to {model_dir}')
"

echo "   Compile done: $(date)"

# ── 2. Run lm_eval ──────────────────────────────────────────────
echo ""
echo ">> [2/3] Running lm_eval..."
echo "   Started: $(date)"

lm_eval --model hf \
    --model_args "pretrained=${OUTDIR}/model,dtype=bfloat16" \
    --tasks ${TASKS} \
    --batch_size auto \
    --device cuda \
    --output_path "${OUTDIR}/lm_eval_results"

echo "   lm_eval done: $(date)"

# ── 3. Build model card and push to HF ──────────────────────────
echo ""
echo ">> [3/3] Pushing to HuggingFace..."

python3 -c "
import json, glob, os
from pathlib import Path
from huggingface_hub import HfApi, create_repo

model_dir = Path('${OUTDIR}/model')
results_dir = Path('${OUTDIR}/lm_eval_results')
token = os.environ.get('HF_TOKEN')
repo_id = '${REPO}'

tasks = ['arc_challenge','gsm8k','hellaswag','mmlu','truthfulqa_mc2','winogrande']
scores = {}
for f in results_dir.rglob('results*.json'):
    data = json.load(open(f))
    for t, v in data.get('results', {}).items():
        name = t.split(',')[0].strip()
        for key in ('acc_norm,none', 'acc,none', 'exact_match,strict-match', 'mc2'):
            if key in v:
                scores[name] = round(v[key] * 100, 1)
                break

print('lm_eval scores:', {t: scores.get(t, '?') for t in tasks})

baseline = {}
for f in Path('sculpt_out_qwen35_9b_live/baseline_lm_eval').rglob('results*.json'):
    data = json.load(open(f))
    for t, v in data.get('results', {}).items():
        name = t.split(',')[0].strip()
        for key in ('acc_norm,none', 'acc,none', 'exact_match,strict-match', 'mc2'):
            if key in v:
                baseline[name] = round(v[key] * 100, 1)
                break

avg_score = sum(scores.get(t, 0) for t in tasks) / len(tasks)
avg_base = sum(baseline.get(t, 0) for t in tasks) / len(tasks)
retention = avg_score / avg_base * 100

table_rows = []
for t in tasks:
    b = baseline.get(t, 0)
    s = scores.get(t, 0)
    d = s - b
    sign = '+' if d > 0 else ''
    table_rows.append(f'| {t} | {s} | {b} | {sign}{d:.1f} |')

card = '''---
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
language:
  - en
base_model: Qwen/Qwen3.5-9B
tags:
  - dystrio
  - sculpt
  - pruned
  - compressed
  - efficient
  - dense
  - drop-in-replacement
  - qwen3.5
datasets:
  - wikitext
  - cais/mmlu
  - teknium/OpenHermes-2.5
---

# Qwen3.5-9B-Sculpt-Aggressive

> **30%% FFN compression with live teacher distillation. Drop-in replacement.**

Dystrio Sculpt structurally compresses transformer FFN layers, producing dense
models that load with standard \`transformers\`. No custom kernels, no runtime changes.

This is the **Aggressive** tier of [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B).

**Use case:** Local deployment — maximum compression for resource-constrained environments.

## Benchmark Results (lm_eval)

| Benchmark | Aggressive (kf=0.70) | Baseline | Delta |
|---|---:|---:|---:|
%s

**Average retention: %.1f%%**

## Quick Start

\x60\x60\x60python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    \"%s\", torch_dtype=\"bfloat16\", device_map=\"auto\",
)
tokenizer = AutoTokenizer.from_pretrained(\"%s\")

inputs = tokenizer(\"The future of AI inference is\", return_tensors=\"pt\").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
\x60\x60\x60

## All Sculpt Tiers

| Tier | HuggingFace | Use Case |
|------|-------------|----------|
| Default (kf=0.95) | [dystrio/Qwen3.5-9B-Sculpt-Default](https://huggingface.co/dystrio/Qwen3.5-9B-Sculpt-Default) | Enterprise — max quality |
| Production (kf=0.90) | [dystrio/Qwen3.5-9B-Sculpt-Production](https://huggingface.co/dystrio/Qwen3.5-9B-Sculpt-Production) | Enterprise — balanced |
| Throughput (kf=0.88) | [dystrio/Qwen3.5-9B-Sculpt-Throughput](https://huggingface.co/dystrio/Qwen3.5-9B-Sculpt-Throughput) | Speed sweet spot |
| Experimental (kf=0.82) | [dystrio/Qwen3.5-9B-Sculpt-Experimental](https://huggingface.co/dystrio/Qwen3.5-9B-Sculpt-Experimental) | Local — aggressive |
| **Aggressive (kf=0.70)** | **[%s](https://huggingface.co/%s)** | **Local — maximum compression** |

## Technical Details

- **Method:** Structural FFN pruning + live teacher distillation (alpha=0.5)
- **Keep fraction:** 0.700 (30%% of FFN neurons removed)
- **Hardware:** 1x NVIDIA H200 141GB

## Compatibility

HuggingFace Transformers, vLLM, TGI, llama.cpp / GGUF, AWQ / GPTQ
''' % (chr(10).join(table_rows), retention, repo_id, repo_id, repo_id, repo_id)

(model_dir / 'README.md').write_text(card)
print(f'Model card written (retention: {retention:.1f}%%)')

api = HfApi(token=token)
try:
    create_repo(repo_id, token=token, private=True, exist_ok=True)
except Exception as e:
    print(f'Repo note: {e}')

api.upload_folder(folder_path=str(model_dir), repo_id=repo_id,
    commit_message='Dystrio Sculpt Aggressive tier (kf=0.700)')
print(f'Pushed: https://huggingface.co/{repo_id}')
"

echo ""
echo "============================================================"
echo "  DONE: $(date)"
echo "============================================================"
