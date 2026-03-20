#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# MoE Routing Patch — 8×A100 Provisioning
#
# Calibrates + bakes deterministic routing for Qwen3.5-122B-A10B
# so prefix caching works in vLLM / SGLang / TGI.
#
# Architecture: 122B total, 10B active, 256 experts (8 routed + 1 shared)
#               48 layers, hidden_size=3072, moe_intermediate_size=1024
#
# Usage:
#   export HF_TOKEN="hf_..."
#   bash scripts/provision_moe_patch.sh
#
# Memory: ~244 GB bf16 → shards across 4-8 A100-80GB via device_map="auto"
# Time:   ~30-60 min calibration (batch mode, all 48 layers in one pass)
###############################################################################

BRANCH="experimental/distill-repair"
REPO="https://github.com/clusteroptimizerengine/BumbleB.git"
WORKDIR="$HOME/BumbleB"
MODEL_ID="Qwen/Qwen3.5-122B-A10B"
OUTPUT_DIR="$HOME/moe_cache_ready"
LOG_FILE="$HOME/moe_patch_run.log"

echo "============================================================"
echo "  MoE Routing Patch — 8×A100 Provisioning"
echo "  Branch: $BRANCH"
echo "  Model:  $MODEL_ID"
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
echo ""
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "GPU count: $GPU_COUNT"

TOTAL_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | awk '{s+=$1} END{print s}')
echo "Total VRAM: ${TOTAL_VRAM} MiB"

if [ "$TOTAL_VRAM" -lt 300000 ]; then
    echo "WARNING: 122B model needs ~250GB VRAM. You have ${TOTAL_VRAM} MiB."
    echo "         Model may not fit. Consider using more GPUs."
fi

if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN not set."
    echo "  export HF_TOKEN=hf_..."
    exit 1
fi

# ── 1. System deps ──────────────────────────────────────────
echo ">> Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq git python3-pip python3-venv > /dev/null 2>&1
echo "   done."

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
pip install -e ".[dev]" -q
pip install accelerate huggingface_hub -q

echo "   Python: $(python --version)"
echo "   torch:  $(python -c 'import torch; print(torch.__version__)')"
echo "   CUDA:   $(python -c 'import torch; print(torch.version.cuda)')"
echo "   GPUs:   $(python -c 'import torch; print(torch.cuda.device_count())')"
echo ""

# ── 4. HuggingFace login ─────────────────────────────────
echo ">> Logging into HuggingFace..."
huggingface-cli login --token "${HF_TOKEN}" 2>/dev/null || \
    python -c "from huggingface_hub import login; login(token='${HF_TOKEN}')"
echo "   done."

# ── 5. Run MoE tests ────────────────────────────────────
echo ">> Running MoE adapter + routing patch tests..."
python -m pytest tests/test_moe_adapter.py tests/test_routing_patch.py -x -q 2>&1 | tail -5
echo ""

# ── 6. Calibrate + Bake + Save ──────────────────────────
echo "============================================================"
echo "  STARTING: Calibrate + Bake routing patch"
echo "  Model: $MODEL_ID"
echo "  Output: $OUTPUT_DIR"
echo "  $(date)"
echo "============================================================"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup python -u -c "
import os, sys, gc, time
import torch
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('moe_patch')

MODEL_ID = '${MODEL_ID}'
OUTPUT_DIR = '${OUTPUT_DIR}'

# ── Load model ──
log.info('loading tokenizer for %s', MODEL_ID)
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

log.info('loading model (device_map=auto across %d GPUs)...', torch.cuda.device_count())
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map='auto',
    trust_remote_code=True,
)
load_time = time.time() - t0
n_params = sum(p.numel() for p in model.parameters())
log.info('model loaded in %.0fs: %.1fB parameters', load_time, n_params / 1e9)

if hasattr(model, 'hf_device_map'):
    devices_used = sorted(set(str(v) for v in model.hf_device_map.values()))
    log.info('sharded across devices: %s', devices_used)

for i in range(torch.cuda.device_count()):
    alloc = torch.cuda.memory_allocated(i) / 1e9
    total = torch.cuda.get_device_properties(i).total_mem / 1e9
    log.info('  GPU %d: %.1f / %.1f GB allocated', i, alloc, total)

# ── Load calibration data ──
log.info('loading calibration texts...')
from datasets import load_dataset

ds = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train')
texts = [t for t in ds['text'][:5000] if len(t.strip()) > 100][:500]
log.info('calibration corpus: %d texts', len(texts))

# ── Calibrate (batch mode — all 48 layers in one pass) ──
log.info('starting Physarum batch calibration...')
from dystrio_sculpt.moe_routing_patch import calibrate_routing_patch, bake_routing_patch

t0 = time.time()
patch = calibrate_routing_patch(
    model, tokenizer, texts,
    device='cuda:0',
    max_tokens=20000,
    coupling_threshold=0.7,
    margin_threshold=0.1,
    model_id=MODEL_ID,
)
cal_time = time.time() - t0
log.info('calibration completed in %.0f seconds (%.1f min)', cal_time, cal_time / 60)

os.makedirs(OUTPUT_DIR, exist_ok=True)
patch.save(os.path.join(OUTPUT_DIR, 'routing_patch.json'))
log.info('patch saved to %s/routing_patch.json', OUTPUT_DIR)

# ── Summary stats ──
total_classes = sum(len(ecs) for ecs in patch.layers.values())
non_singleton = sum(1 for ecs in patch.layers.values() for ec in ecs if len(ec.members) > 1)
experts_in_classes = sum(len(ec.members) for ecs in patch.layers.values() for ec in ecs if len(ec.members) > 1)
log.info('patch summary:')
log.info('  layers with MoE: %d', len(patch.layers))
log.info('  total eq. classes: %d (%d non-singleton)', total_classes, non_singleton)
log.info('  experts in non-singleton classes: %d', experts_in_classes)

# ── Bake into weights ──
log.info('baking routing patch into model weights...')
t0 = time.time()
n_modified = bake_routing_patch(model, patch)
bake_time = time.time() - t0
log.info('baked: %d layers modified in %.1fs', n_modified, bake_time)

# ── Validate: determinism check ──
log.info('validating routing determinism...')
test_text = 'The quick brown fox jumps over the lazy dog. Machine learning is transforming industries.'
inp = tokenizer(test_text, return_tensors='pt', truncation=True, max_length=128)
first_device = next(model.parameters()).device
inp = {k: v.to(first_device) for k, v in inp.items()}

with torch.no_grad():
    out1 = model(**inp, use_cache=False)
    out2 = model(**inp, use_cache=False)

logit_diff = (out1.logits - out2.logits).abs().max().item()
log.info('routing determinism check: max logit diff = %.2e (should be 0 or ~1e-7)', logit_diff)
if logit_diff < 1e-4:
    log.info('PASS: routing is deterministic')
else:
    log.warning('WARN: logit diff %.2e is higher than expected', logit_diff)

# ── Save model ──
log.info('saving patched model to %s...', OUTPUT_DIR)
t0 = time.time()
model.save_pretrained(OUTPUT_DIR, max_shard_size='5GB')
tokenizer.save_pretrained(OUTPUT_DIR)
save_time = time.time() - t0
log.info('model saved in %.0fs', save_time)

# ── Push to HuggingFace ──
log.info('pushing to HuggingFace...')
from huggingface_hub import HfApi
api = HfApi()
repo_id = 'dystrio/Qwen3.5-122B-A10B-CacheReady'
api.create_repo(repo_id, exist_ok=True, private=False)
api.upload_folder(
    folder_path=OUTPUT_DIR,
    repo_id=repo_id,
    commit_message='Baked Physarum routing canonicalization for prefix cache determinism',
)
log.info('pushed to https://huggingface.co/%s', repo_id)

log.info('============================================================')
log.info('  MoE ROUTING PATCH COMPLETE')
log.info('  Output: %s', OUTPUT_DIR)
log.info('  HuggingFace: https://huggingface.co/%s', repo_id)
log.info('============================================================')
" > "$LOG_FILE" 2>&1 &

echo ""
echo ">> Calibration running in background (nohup)."
echo "   PID: $!"
echo "   Log: $LOG_FILE"
echo ""
echo "Monitor with:"
echo "   tail -f $LOG_FILE"
echo ""
echo "Check status:"
echo "   strings $LOG_FILE | grep -E 'calibration completed|baked|PASS|pushed|ERROR'"
echo ""
