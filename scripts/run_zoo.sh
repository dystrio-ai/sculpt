#!/usr/bin/env bash
set -euo pipefail

# ── Dystrio Model Zoo Compile + Bench Script ────────────────────────────────
# Run on A100 80GB box with HF_TOKEN already set.
# Usage: bash scripts/run_zoo.sh
# All output goes to /ephemeral/zoo/

ZOO_DIR="/ephemeral/zoo"
mkdir -p "$ZOO_DIR"

MODELS=(
  "mistralai/Mistral-7B-Instruct-v0.3|mistral-7b-instruct"
  "meta-llama/Llama-3.1-8B-Instruct|llama-3.1-8b-instruct"
  "Qwen/Qwen2.5-7B-Instruct|qwen2.5-7b-instruct"
  "google/gemma-2-9b-it|gemma-2-9b-it"
)
# NOTE: Phi-3.5 uses fused gate_up_proj (not separate gate_proj/up_proj),
# which is incompatible with the current FFN pruning code.

FALLBACK="mistralai/Mistral-7B-v0.1|mistral-7b-v0.1-base"

echo "=========================================="
echo "  Dystrio Model Zoo — Compile Phase"
echo "=========================================="
echo ""

COMPILED=()
FAILED=()

for entry in "${MODELS[@]}"; do
  IFS='|' read -r model_id short <<< "$entry"
  outdir="$ZOO_DIR/$short"
  logfile="$ZOO_DIR/${short}_compile.log"

  echo "[$(date +%H:%M:%S)] Compiling $model_id → $outdir"

  if dystrio sculpt \
    --model-id "$model_id" \
    --outdir "$outdir" \
    --frontier 4 \
    --deterministic 2>&1 | tee "$logfile"; then
    echo "[$(date +%H:%M:%S)] ✓ $short compile succeeded"
    COMPILED+=("$entry")
  else
    echo "[$(date +%H:%M:%S)] ✗ $short compile FAILED — see $logfile"
    FAILED+=("$short")
  fi
  echo ""
done

if [ ${#FAILED[@]} -gt 0 ]; then
  echo "Failed models: ${FAILED[*]}"
  echo "Running fallback model..."
  IFS='|' read -r fb_model fb_short <<< "$FALLBACK"
  outdir="$ZOO_DIR/$fb_short"
  logfile="$ZOO_DIR/${fb_short}_compile.log"
  echo "[$(date +%H:%M:%S)] Compiling fallback $fb_model → $outdir"
  dystrio sculpt \
    --model-id "$fb_model" \
    --outdir "$outdir" \
    --frontier 4 \
    --deterministic 2>&1 | tee "$logfile" || true
  COMPILED+=("$FALLBACK")
fi

echo ""
echo "=========================================="
echo "  Dystrio Model Zoo — Benchmark Phase"
echo "=========================================="
echo ""

for entry in "${COMPILED[@]}"; do
  IFS='|' read -r model_id short <<< "$entry"
  outdir="$ZOO_DIR/$short"
  bench_out="$ZOO_DIR/${short}_bench"
  logfile="$ZOO_DIR/${short}_bench.log"

  con_model="$outdir/frontier_0_conservative/model"
  bal_model="$outdir/frontier_1_balanced/model"

  bench_args=(--models "$model_id" --workloads wikitext --outdir "$bench_out" --baseline-model "$model_id")

  if [ -d "$con_model" ]; then
    bench_args+=(--models "$con_model")
  fi
  if [ -d "$bal_model" ]; then
    bench_args+=(--models "$bal_model")
  fi

  echo "[$(date +%H:%M:%S)] Benchmarking $short"
  if dystrio bench "${bench_args[@]}" 2>&1 | tee "$logfile"; then
    echo "[$(date +%H:%M:%S)] ✓ $short bench succeeded"
  else
    echo "[$(date +%H:%M:%S)] ✗ $short bench FAILED — see $logfile"
  fi
  echo ""
done

echo ""
echo "=========================================="
echo "  Model Zoo Complete"
echo "=========================================="
echo ""
echo "Compile outputs:  $ZOO_DIR/*/frontier_*/model"
echo "Bench outputs:    $ZOO_DIR/*_bench/benchmarks.csv"
echo ""

for entry in "${COMPILED[@]}"; do
  IFS='|' read -r model_id short <<< "$entry"
  outdir="$ZOO_DIR/$short"
  echo "--- $short ---"
  if [ -f "$outdir/frontier_0_conservative/manifest.json" ]; then
    echo "  Conservative: $(grep -o '"compile_wall_time_s": [0-9.]*' "$outdir/frontier_0_conservative/manifest.json")"
    echo "               $(grep -o '"new_intermediate_size": [0-9]*' "$outdir/frontier_0_conservative/manifest.json")"
  fi
  if [ -f "$outdir/frontier_1_balanced/manifest.json" ]; then
    echo "  Balanced:     $(grep -o '"compile_wall_time_s": [0-9.]*' "$outdir/frontier_1_balanced/manifest.json")"
    echo "               $(grep -o '"new_intermediate_size": [0-9]*' "$outdir/frontier_1_balanced/manifest.json")"
  fi
  bench_csv="$ZOO_DIR/${short}_bench/benchmarks.csv"
  if [ -f "$bench_csv" ]; then
    echo "  Benchmarks:   $bench_csv"
  fi
  echo ""
done
