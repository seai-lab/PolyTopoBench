#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPARSERCNN_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${SPARSERCNN_ROOT}/../../../../.." && pwd)"
DEVENTER_VARIANT="${DEVENTER_VARIANT:-deventer_512_valtest_as_val}"
DEVENTER_PROCESSED_ROOT="${DEVENTER_PROCESSED_ROOT:-${PROJECT_ROOT}/data_processed/${DEVENTER_VARIANT}}"
DEVENTER_OUTPUT_TAG="${DEVENTER_OUTPUT_TAG:-${DEVENTER_VARIANT}}"

SCENARIO="${SCENARIO:-road}"
SPLIT="${SPLIT:-val}"
DATA_ROOT="${DATA_ROOT:-${DEVENTER_PROCESSED_ROOT}/roipoly/${SCENARIO}}"
GPU_ID="${GPU_ID:-0}"
CONDA_ENV="${CONDA_ENV:-polytopobench}"
CONFIG_FILE="${CONFIG_FILE:-configs/sparsercnn.res50.160pro.deventer512.yaml}"
NUM_WORKERS="${NUM_WORKERS:-8}"
USE_SMOKE_ANN="${USE_SMOKE_ANN:-0}"

JSON_NAME="annotation_roipoly.json"
RUN_SUFFIX=""
if [[ "${USE_SMOKE_ANN}" == "1" ]]; then
  JSON_NAME="annotation_roipoly_smoke.json"
  RUN_SUFFIX="_smoke"
fi

TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-${PROJECT_ROOT}/output/sparsercnn/${DEVENTER_OUTPUT_TAG}/${SCENARIO}/res50_160pro_512${RUN_SUFFIX}}"
MODEL_WEIGHTS="${MODEL_WEIGHTS:-${TRAIN_OUTPUT_DIR}/model_final.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/output/sparsercnn/${DEVENTER_OUTPUT_TAG}/${SCENARIO}/res50_160pro_512${RUN_SUFFIX}_eval_${SPLIT}}"

mkdir -p "${OUTPUT_DIR}"
cd "${SPARSERCNN_ROOT}"
PYTHONPATH="${SPARSERCNN_ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU_ID}" conda run -n "${CONDA_ENV}" \
  python train_net.py \
  --num-gpus 1 \
  --eval-only \
  --config-file "${CONFIG_FILE}" \
  --train-dataset deventer_train \
  --train-json "${DATA_ROOT}/train/${JSON_NAME}" \
  --train-path "${DATA_ROOT}/train/images" \
  --val-dataset "deventer_${SPLIT}" \
  --val-json "${DATA_ROOT}/${SPLIT}/${JSON_NAME}" \
  --val-path "${DATA_ROOT}/${SPLIT}/images" \
  "$@" \
  MODEL.WEIGHTS "${MODEL_WEIGHTS}" \
  OUTPUT_DIR "${OUTPUT_DIR}" \
  DATALOADER.NUM_WORKERS "${NUM_WORKERS}"
