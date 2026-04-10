#!/usr/bin/env bash
# =============================================================================
# One-shot: fresh Ubuntu GPU box → clone Sculpt → venv → deps → ablation study
#
# Usage (on the A100, after nvidia-smi works), once this file is on GitHub main:
#   curl -fsSL https://raw.githubusercontent.com/dystrio-ai/sculpt/main/scripts/bootstrap_ablation_box.sh | bash
# If that URL 404s, the script is not merged yet — copy this file to the box or clone the
# repo and run:  bash scripts/bootstrap_ablation_box.sh
#
# Or copy this file to the box and:
#   bash bootstrap_ablation_box.sh
#
# Optional environment (defaults shown):
#   SCULPT_REPO   https://github.com/dystrio-ai/sculpt.git
#   SCULPT_BRANCH main
#   WORKDIR       $HOME/sculpt
#   MODEL         meta-llama/Llama-3.1-8B-Instruct
#   SKIP_LMEVAL   0              # set to 1 to skip lm-eval (faster, PPL-only sculpt side)
#   HF_TOKEN      (unset)        # export hf_... for gated models / faster Hub
# =============================================================================
set -euo pipefail

SCULPT_REPO="${SCULPT_REPO:-https://github.com/dystrio-ai/sculpt.git}"
SCULPT_BRANCH="${SCULPT_BRANCH:-main}"
WORKDIR="${WORKDIR:-$HOME/sculpt}"
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
SKIP_LMEVAL="${SKIP_LMEVAL:-0}"

echo "============================================================"
echo "  Sculpt ablation box bootstrap"
echo "  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  repo:    $SCULPT_REPO ($SCULPT_BRANCH)"
echo "  workdir: $WORKDIR"
echo "  model:   $MODEL"
echo "  lm-eval: $([ "$SKIP_LMEVAL" = 1 ] && echo SKIPPED || echo ON)"
echo "============================================================"

if ! command -v nvidia-smi &>/dev/null; then
  echo "ERROR: nvidia-smi not found. Use a GPU image with NVIDIA drivers installed."
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

echo ">> apt: git, venv, pip"
sudo apt-get update -qq
sudo apt-get install -y -qq git python3-venv python3-pip

if [ -d "$WORKDIR/.git" ]; then
  echo ">> repo exists, pulling: $WORKDIR"
  git -C "$WORKDIR" fetch origin
  git -C "$WORKDIR" checkout "$SCULPT_BRANCH"
  git -C "$WORKDIR" pull origin "$SCULPT_BRANCH"
else
  echo ">> cloning: $WORKDIR"
  git clone --branch "$SCULPT_BRANCH" "$SCULPT_REPO" "$WORKDIR"
fi

cd "$WORKDIR"

echo ">> Python venv: $WORKDIR/.venv"
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
# shellcheck source=/dev/null
source .venv/bin/activate

pip install -U pip setuptools wheel -q
pip install -e ".[dev]" -q
pip install "lm-eval>=0.4" huggingface_hub -q

if [ -n "${HF_TOKEN:-}" ]; then
  echo ">> Hugging Face login (token from env)"
  python -c "from huggingface_hub import login; login(token='${HF_TOKEN}')"
else
  echo ">> HF_TOKEN not set — fine for public models; set it for gated weights."
fi

echo ">> pytest (quick sanity)"
python -m pytest tests/ -q -m "not slow" --tb=no -x || {
  echo "WARNING: pytest had failures; continuing to ablation anyway."
}

echo ""
echo ">> Running ablation study (this is the long part)..."
MODEL="$MODEL" SKIP_LMEVAL="$SKIP_LMEVAL" bash scripts/ablation_study.sh

MODEL_SHORT=$(echo "$MODEL" | sed 's|.*/||')
echo ""
echo "============================================================"
echo "  Done."
echo "  Results under: $WORKDIR/ablation_results/"
echo "  Charts:        python3 scripts/visualize_ablation.py ablation_results/ --model $MODEL_SHORT"
echo "============================================================"
