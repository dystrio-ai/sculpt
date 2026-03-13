#!/usr/bin/env bash
set -euo pipefail

# Dystrio Factory — GPU Validation + New Model Batch
#
# Pre-flight checklist (verify before running):
#   [ ] GPU available:         nvidia-smi shows device
#   [ ] HF_TOKEN set:          echo $HF_TOKEN
#   [ ] dystrio installed:     dystrio --help
#   [ ] torch sees GPU:        python3 -c "import torch; print(torch.cuda.get_device_name(0))"
#   [ ] ZOO_DIR writable:      touch $ZOO_DIR/.test && rm $ZOO_DIR/.test
#   [ ] Enough disk:           df -h $ZOO_DIR (need ~50GB per model)
#
# Sequence:
#   Step 0: Validation run (golden model, rich-record contract)
#   Step 1: Repeat validation (stability check)
#   Step 2-6: New models

export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN first}"

ZOO_DIR="${ZOO_DIR:-/data/zoo}"
DATASET="${DATASET:-${ZOO_DIR}/dystrio_efficiency_dataset.jsonl}"

echo "============================================"
echo "  Dystrio Factory — GPU Run"
echo "  Zoo:     ${ZOO_DIR}"
echo "  Dataset: ${DATASET}"
echo "============================================"

# ── Step 0: Validation run (golden model, rich-record check) ─────────────
echo ""
echo "[0/6] VALIDATION: Mistral 7B (golden model, frontier=2, full contract check)"
dystrio factory validate \
    --model-id mistralai/Mistral-7B-Instruct-v0.3 \
    --zoo-dir "${ZOO_DIR}/validation_run_1" \
    --dataset-path "${DATASET}" \
    --frontier 2

echo "  -> Validation passed. Inspecting record:"
dystrio dataset inspect --dataset-path "${DATASET}" --last 1
echo ""

# ── Step 1: Repeat validation (stability) ────────────────────────────────
echo "[1/6] REPEAT VALIDATION: Mistral 7B (verify append + stability)"
dystrio factory validate \
    --model-id mistralai/Mistral-7B-Instruct-v0.3 \
    --zoo-dir "${ZOO_DIR}/validation_run_2" \
    --dataset-path "${DATASET}" \
    --frontier 2

echo "  -> Second run passed. Dataset now has 2 fresh records."
echo ""

# ── Step 2: Phi 3.5 Mini (3.8B) ────────────────────────────────────────
echo "[2/6] microsoft/Phi-3.5-mini-instruct (phi family, 3.8B)"
dystrio factory run \
    --model-id microsoft/Phi-3.5-mini-instruct \
    --zoo-dir "${ZOO_DIR}" \
    --dataset-path "${DATASET}" \
    --skip-publish
dystrio dataset inspect --dataset-path "${DATASET}" --last 1

echo ""

# ── Step 3: Qwen 2.5 3B ────────────────────────────────────────────────
echo "[3/6] Qwen/Qwen2.5-3B-Instruct (qwen family, 3B)"
dystrio factory run \
    --model-id Qwen/Qwen2.5-3B-Instruct \
    --zoo-dir "${ZOO_DIR}" \
    --dataset-path "${DATASET}" \
    --skip-publish
dystrio dataset inspect --dataset-path "${DATASET}" --last 1

echo ""

# ── Step 4: Llama 3.2 3B ───────────────────────────────────────────────
echo "[4/6] meta-llama/Llama-3.2-3B-Instruct (llama family, 3.2B)"
dystrio factory run \
    --model-id meta-llama/Llama-3.2-3B-Instruct \
    --zoo-dir "${ZOO_DIR}" \
    --dataset-path "${DATASET}" \
    --skip-publish
dystrio dataset inspect --dataset-path "${DATASET}" --last 1

echo ""

# ── Step 5: Gemma 2 2B ─────────────────────────────────────────────────
echo "[5/6] google/gemma-2-2b-it (gemma family, 2.6B)"
dystrio factory run \
    --model-id google/gemma-2-2b-it \
    --zoo-dir "${ZOO_DIR}" \
    --dataset-path "${DATASET}" \
    --skip-publish
dystrio dataset inspect --dataset-path "${DATASET}" --last 1

echo ""

# ── Step 6: Mistral Nemo 12B ───────────────────────────────────────────
echo "[6/6] mistralai/Mistral-Nemo-Instruct-2407 (mistral family, 12B)"
dystrio factory run \
    --model-id mistralai/Mistral-Nemo-Instruct-2407 \
    --zoo-dir "${ZOO_DIR}" \
    --dataset-path "${DATASET}" \
    --skip-publish
dystrio dataset inspect --dataset-path "${DATASET}" --last 1

echo ""
echo "============================================"
echo "  All models complete."
echo "  Dataset records: $(wc -l < "${DATASET}" 2>/dev/null || echo 'N/A')"
echo "============================================"

dystrio dataset stats --dataset-path "${DATASET}"
echo ""
echo "Full dataset inspection:"
dystrio dataset inspect --dataset-path "${DATASET}"
