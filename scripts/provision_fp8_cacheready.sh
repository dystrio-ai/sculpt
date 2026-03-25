#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# FP8 CacheReady — Patch + Benchmark + Push
#
# 1. Downloads Qwen/Qwen3.5-122B-A10B-FP8
# 2. Applies routing canonicalization from existing routing_patch.json
#    (gate weights are bf16 in the FP8 checkpoint — excluded from quantization)
# 3. Saves patched model
# 4. Runs the full benchmark suite (determinism, prefix caching, fp8 stability)
# 5. Pushes to HuggingFace as dystrio/Qwen3.5-122B-A10B-FP8-CacheReady
#
# Requirements: 4x A100-80GB or 2x H100 (model is ~120GB fp8)
#
# Usage:
#   export HF_TOKEN="hf_..."
#   TP_SIZE=4 bash scripts/provision_fp8_cacheready.sh
###############################################################################

BRANCH="experimental/distill-repair"
REPO="https://github.com/clusteroptimizerengine/BumbleB.git"
WORKDIR="$HOME/BumbleB"

FP8_MODEL="Qwen/Qwen3.5-122B-A10B-FP8"
BF16_CACHEREADY="dystrio/Qwen3.5-122B-A10B-CacheReady"
FP8_CACHEREADY="dystrio/Qwen3.5-122B-A10B-FP8-CacheReady"
TP_SIZE="${TP_SIZE:-4}"

if [ -d "/ephemeral" ] && [ -w "/ephemeral" ]; then
    STORAGE="/ephemeral"
elif [ -d "/mnt" ] && [ -w "/mnt" ]; then
    STORAGE="/mnt"
else
    STORAGE="$HOME"
fi

OUTPUT_DIR="${STORAGE}/fp8_cacheready_output"
PATCHED_DIR="${STORAGE}/fp8_cacheready_model"
LOG_FILE="${STORAGE}/fp8_cacheready.log"
export HF_HOME="${STORAGE}/hf_cache"
export XET_CACHE_HOME="${STORAGE}/hf_cache/xet"
mkdir -p "$HF_HOME" "$OUTPUT_DIR" "$PATCHED_DIR"

echo "============================================================"
echo "  FP8 CacheReady — Patch + Benchmark + Push"
echo "  Source:  $FP8_MODEL"
echo "  Output:  $FP8_CACHEREADY"
echo "  TP size: $TP_SIZE"
echo "  $(date)"
echo "============================================================"

# ── 0. Preflight ─────────────────────────────────────────────
if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found."
    exit 1
fi

echo ""
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "GPU count: $GPU_COUNT"

if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN not set."
    exit 1
fi

# ── 1. System deps ──────────────────────────────────────────
echo ">> Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq git python3-pip python3-venv > /dev/null 2>&1

# ── 2. Clone / update repo ─────────────────────────────────
if [ -d "$WORKDIR" ]; then
    echo ">> Repo exists, pulling latest..."
    cd "$WORKDIR"
    git fetch origin
    git checkout "$BRANCH"
    git pull origin "$BRANCH"
else
    echo ">> Cloning repo..."
    git clone --branch "$BRANCH" "$REPO" "$WORKDIR"
    cd "$WORKDIR"
fi

# ── 3. Python environment ─────────────────────────────────
echo ">> Setting up Python environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate

pip install --upgrade pip setuptools wheel -q
pip install vllm -q
pip install lm_eval ray -q
pip install -e ".[dev]" -q
pip install accelerate huggingface_hub safetensors -q
pip install 'transformers>=4.56.0,<5' -q

echo "   Python: $(python --version)"
echo "   torch:  $(python -c 'import torch; print(torch.__version__)')"
echo "   CUDA:   $(python -c 'import torch; print(torch.version.cuda)')"
echo "   GPUs:   $(python -c 'import torch; print(torch.cuda.device_count())')"
echo "   vLLM:   $(python -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo 'not installed')"

# ── 4. HuggingFace login ─────────────────────────────────
echo ">> Logging into HuggingFace..."
huggingface-cli login --token "${HF_TOKEN}" 2>/dev/null || \
    python -c "from huggingface_hub import login; login(token='${HF_TOKEN}')"
echo "   done."

# ── 5. Unit tests ────────────────────────────────────────
echo ">> Running unit tests..."
python -m pytest tests/test_routing_patch.py -x -q 2>&1 | tail -3

# ── 6. Patch FP8 model + Benchmark + Push ────────────────
echo "============================================================"
echo "  STARTING: FP8 CacheReady pipeline"
echo "  $(date)"
echo "============================================================"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup python -u -c "
import os, sys, gc, time, json, shutil
import torch
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('fp8_cacheready')

FP8_MODEL = '${FP8_MODEL}'
BF16_CACHEREADY = '${BF16_CACHEREADY}'
FP8_CACHEREADY = '${FP8_CACHEREADY}'
PATCHED_DIR = '${PATCHED_DIR}'
OUTPUT_DIR = '${OUTPUT_DIR}'
TP_SIZE = int('${TP_SIZE}')
TOKEN = os.environ.get('HF_TOKEN', '')

# ════════════════════════════════════════════════════════════
# STEP 1: Download FP8 model and routing_patch.json
# ════════════════════════════════════════════════════════════
log.info('=== Step 1: Download FP8 model + routing patch ===')

from huggingface_hub import snapshot_download, hf_hub_download
import tempfile

t0 = time.time()
log.info('downloading %s ...', FP8_MODEL)
fp8_local = snapshot_download(repo_id=FP8_MODEL, local_dir=PATCHED_DIR)
log.info('FP8 model downloaded to %s in %.0fs', fp8_local, time.time() - t0)

log.info('downloading routing_patch.json from %s', BF16_CACHEREADY)
with tempfile.TemporaryDirectory() as tmp:
    patch_path = hf_hub_download(
        BF16_CACHEREADY, 'routing_patch.json',
        token=TOKEN, cache_dir=tmp,
    )
    import shutil as _sh
    dest_patch = os.path.join(PATCHED_DIR, 'routing_patch.json')
    _sh.copy2(patch_path, dest_patch)
    log.info('routing_patch.json saved to %s', dest_patch)

# ════════════════════════════════════════════════════════════
# STEP 2: Apply routing canonicalization to FP8 gate weights
# ════════════════════════════════════════════════════════════
log.info('=== Step 2: Apply routing patch to FP8 gate weights ===')

from dystrio_sculpt.moe_routing_patch import RoutingPatch

patch = RoutingPatch.load(dest_patch)
log.info('loaded patch: %d layers, %d non-singleton classes',
         len(patch.layers),
         sum(1 for ecs in patch.layers.values() for ec in ecs if len(ec.members) > 1))

# Apply patch directly to safetensors files (no full model load needed)
from safetensors import safe_open
from safetensors.torch import save_file
from pathlib import Path

index_path = Path(PATCHED_DIR) / 'model.safetensors.index.json'
with open(index_path) as f:
    idx = json.load(f)
weight_map = idx['weight_map']

gate_keys = [k for k in weight_map if '.mlp.gate.weight' in k and not k.startswith('mtp.')]
log.info('found %d gate weight keys to patch', len(gate_keys))

# Group gate keys by shard file
from collections import defaultdict
shard_to_gates = defaultdict(list)
for k in gate_keys:
    shard_to_gates[weight_map[k]].append(k)

total_rows_modified = 0
for shard_name, keys in sorted(shard_to_gates.items()):
    shard_path = Path(PATCHED_DIR) / shard_name
    log.info('patching %s (%d gate weights)', shard_name, len(keys))

    # Load all tensors from this shard
    tensors = {}
    with safe_open(str(shard_path), framework='pt') as f:
        for tensor_name in f.keys():
            tensors[tensor_name] = f.get_tensor(tensor_name)

    # Apply row copies for each gate in this shard
    for gate_key in keys:
        # Extract layer index from key like 'model.language_model.layers.5.mlp.gate.weight'
        parts = gate_key.split('.')
        layer_idx = None
        for i, p in enumerate(parts):
            if p == 'layers' and i + 1 < len(parts):
                try:
                    layer_idx = int(parts[i + 1])
                    break
                except ValueError:
                    pass

        if layer_idx is None or layer_idx not in patch.layers:
            continue

        W = tensors[gate_key]
        n_experts = W.shape[0]
        layer_swaps = 0
        TIEBREAK_SCALE = 1.0 - 1e-4

        for ec in patch.layers[layer_idx]:
            if len(ec.members) <= 1:
                continue
            if ec.canonical >= n_experts:
                continue
            canonical_row = W[ec.canonical].clone()
            for member in ec.members:
                if member != ec.canonical and member < n_experts:
                    W[member].copy_(canonical_row * TIEBREAK_SCALE)
                    layer_swaps += 1

        tensors[gate_key] = W
        total_rows_modified += layer_swaps
        log.info('  layer %d: %d rows canonicalized', layer_idx, layer_swaps)

        # Also patch bias if present
        bias_key = gate_key.replace('.weight', '.bias')
        if bias_key in tensors:
            B = tensors[bias_key]
            for ec in patch.layers[layer_idx]:
                if len(ec.members) <= 1:
                    continue
                if ec.canonical >= n_experts:
                    continue
                canonical_bias = B[ec.canonical].clone()
                for member in ec.members:
                    if member != ec.canonical and member < n_experts:
                        B[member].copy_(canonical_bias)
            tensors[bias_key] = B

    # Save patched shard back
    save_file(tensors, str(shard_path))
    log.info('  saved patched shard')

log.info('routing patch applied: %d total gate weight rows modified', total_rows_modified)

# ════════════════════════════════════════════════════════════
# STEP 3: Push patched FP8 model to HuggingFace
# ════════════════════════════════════════════════════════════
log.info('=== Step 3: Push to HuggingFace ===')

from huggingface_hub import HfApi
api = HfApi(token=TOKEN)
api.create_repo(FP8_CACHEREADY, exist_ok=True, private=False)

log.info('uploading to %s (this may take a while)...', FP8_CACHEREADY)
t0 = time.time()
api.upload_folder(
    folder_path=PATCHED_DIR,
    repo_id=FP8_CACHEREADY,
    commit_message='FP8 CacheReady: routing canonicalization for prefix cache determinism',
)
log.info('uploaded in %.0fs', time.time() - t0)

# ════════════════════════════════════════════════════════════
# STEP 4: Run benchmark suite
# ════════════════════════════════════════════════════════════
log.info('=== Step 4: Run benchmark suite ===')
log.info('cleaning GPU memory before benchmark...')
gc.collect()
torch.cuda.empty_cache()
import subprocess
subprocess.run(['pkill', '-f', 'multiproc_executor'], capture_output=True)
import time as _t; _t.sleep(5)

log.info('starting benchmark: original=%s patched=%s tp=%d', FP8_MODEL, PATCHED_DIR, TP_SIZE)

# Run benchmark as subprocess to get clean GPU state
result = subprocess.run([
    sys.executable, '-u', 'scripts/benchmark_moe_routing.py',
    '--original', FP8_MODEL,
    '--patched', PATCHED_DIR,
    '--output', OUTPUT_DIR,
    '--tp', str(TP_SIZE),
], cwd='${WORKDIR}')

if result.returncode == 0:
    log.info('benchmark completed successfully')
else:
    log.warning('benchmark exited with code %d', result.returncode)

# ════════════════════════════════════════════════════════════
# STEP 5: Print results
# ════════════════════════════════════════════════════════════
report_path = os.path.join(OUTPUT_DIR, 'benchmark_report.md')
if os.path.exists(report_path):
    log.info('=== BENCHMARK REPORT ===')
    with open(report_path) as f:
        print(f.read())

# ════════════════════════════════════════════════════════════
# STEP 6: Push model card with benchmark results
# ════════════════════════════════════════════════════════════
log.info('=== Step 6: Push model card ===')
card_result = subprocess.run([
    sys.executable, 'scripts/push_fp8_model_card.py', OUTPUT_DIR,
], cwd='${WORKDIR}')
if card_result.returncode == 0:
    log.info('model card pushed successfully')
else:
    log.warning('model card push exited with code %d', card_result.returncode)

log.info('============================================================')
log.info('  FP8 CACHEREADY PIPELINE COMPLETE')
log.info('  Model: https://huggingface.co/%s', FP8_CACHEREADY)
log.info('  Results: %s', OUTPUT_DIR)
log.info('============================================================')
" > "$LOG_FILE" 2>&1 &

echo ""
echo ">> Pipeline running in background (nohup)."
echo "   PID: $!"
echo "   Log: $LOG_FILE"
echo ""
echo "Monitor with:"
echo "   tail -f $LOG_FILE"
echo ""
echo "Check status:"
echo "   strings $LOG_FILE | grep -E 'PASS|FAIL|COMPLETE|ERROR|===|Step'"
echo ""
echo "Estimated runtime: ~1-2 hours (patch + upload + benchmark)"
echo ""
