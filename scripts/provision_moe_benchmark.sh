#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# MoE Routing Patch — Benchmark Suite Provisioning
#
# Runs the full benchmark suite comparing original vs patched Qwen3.5-122B:
#   1. Routing determinism (bf16)
#   2. Quality preservation (lm_eval: MMLU, HellaSwag, ARC, TruthfulQA, etc.)
#   3. vLLM prefix caching throughput
#   4. nvfp4 quantization routing stability
#
# Requirements: 8×A100-80GB (or equivalent ~640GB VRAM)
#
# Usage:
#   export HF_TOKEN="hf_..."
#   bash scripts/provision_moe_benchmark.sh
#
# Expected runtime: ~7-9 hours
###############################################################################

BRANCH="experimental/distill-repair"
REPO="${SCULPT_REPO:-https://github.com/dystrio/sculpt.git}"
WORKDIR="$HOME/BumbleB"

ORIGINAL_MODEL="${ORIGINAL_MODEL:-Qwen/Qwen3.5-122B-A10B}"
PATCHED_MODEL="${PATCHED_MODEL:-dystrio/Qwen3.5-122B-A10B-CacheReady}"
TP_SIZE="${TP_SIZE:-4}"

# Use /ephemeral if available (large disk), otherwise fall back to $HOME
if [ -d "/ephemeral" ] && [ -w "/ephemeral" ]; then
    STORAGE="/ephemeral"
elif [ -d "/mnt" ] && [ -w "/mnt" ]; then
    STORAGE="/mnt"
else
    STORAGE="$HOME"
fi

OUTPUT_DIR="${STORAGE}/moe_benchmark_results"
LOG_FILE="${STORAGE}/moe_benchmark.log"
export HF_HOME="${STORAGE}/hf_cache"
export XET_CACHE_HOME="${STORAGE}/hf_cache/xet"
mkdir -p "$HF_HOME" "$OUTPUT_DIR"

echo "============================================================"
echo "  MoE Routing Patch — Benchmark Suite"
echo "  Branch: $BRANCH"
echo "  Original: $ORIGINAL_MODEL"
echo "  Patched:  $PATCHED_MODEL"
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
pip install vllm -q
pip install lm_eval ray -q
pip install -e ".[dev]" -q
pip install accelerate huggingface_hub -q
# Ensure transformers is vLLM-compatible (all tests use vLLM natively)
pip install 'transformers>=4.56.0,<5' -q

echo "   Python: $(python --version)"
echo "   torch:  $(python -c 'import torch; print(torch.__version__)')"
echo "   CUDA:   $(python -c 'import torch; print(torch.version.cuda)')"
echo "   GPUs:   $(python -c 'import torch; print(torch.cuda.device_count())')"
echo "   vLLM:   $(python -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo 'not installed')"
echo ""

# ── 4. HuggingFace login ─────────────────────────────────
echo ">> Logging into HuggingFace..."
huggingface-cli login --token "${HF_TOKEN}" 2>/dev/null || \
    python -c "from huggingface_hub import login; login(token='${HF_TOKEN}')"
echo "   done."

# ── 5. Quick sanity check ────────────────────────────────
echo ">> Running unit tests..."
python -m pytest tests/test_routing_patch.py -x -q 2>&1 | tail -3
echo ""

# ── 6. Run benchmark suite ───────────────────────────────
echo "============================================================"
echo "  STARTING: MoE Routing Patch Benchmark Suite"
echo "  Original: $ORIGINAL_MODEL"
echo "  Patched:  $PATCHED_MODEL"
echo "  TP size:  $TP_SIZE"
echo "  Output:   $OUTPUT_DIR"
echo "  $(date)"
echo "============================================================"

mkdir -p "$OUTPUT_DIR"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup python -u scripts/benchmark_moe_routing.py \
    --original "$ORIGINAL_MODEL" \
    --patched "$PATCHED_MODEL" \
    --output "$OUTPUT_DIR" \
    --tp "$TP_SIZE" \
    > "$LOG_FILE" 2>&1 &

echo ""
echo ">> Benchmark running in background (nohup)."
echo "   PID: $!"
echo "   Log: $LOG_FILE"
echo ""
echo "Monitor with:"
echo "   tail -f $LOG_FILE"
echo ""
echo "Check status:"
echo "   strings $LOG_FILE | grep -E 'PASS|FAIL|COMPLETE|ERROR|==='"
echo ""
echo "Estimated runtime: 7-9 hours"
echo ""
