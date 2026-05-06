#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPARSERCNN_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${SPARSERCNN_ROOT}/../../../../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data_processed/inria_building/roipoly}"

GPU_ID="${GPU_ID:-0}"
CONDA_ENV="${CONDA_ENV:-polytopobench}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/output/sparsercnn/inria_building/building/res50_160pro_512_eval_val}"
MODEL_WEIGHTS="${MODEL_WEIGHTS:-${PROJECT_ROOT}/output/sparsercnn/inria_building/building/res50_160pro_512/model_final.pth}"
NUM_WORKERS="${NUM_WORKERS:-4}"

mkdir -p "${OUTPUT_DIR}"
cd "${SPARSERCNN_ROOT}"
PYTHONPATH="${SPARSERCNN_ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU_ID}" conda run -n "${CONDA_ENV}" \
  python train_net.py \
  --num-gpus 1 \
  --eval-only \
  --config-file configs/sparsercnn.res50.160pro.inria512.yaml \
  --train-dataset inria_train \
  --train-json "${DATA_ROOT}/train/annotation_roipoly.json" \
  --train-path "${DATA_ROOT}/train/images" \
  --val-dataset inria_val \
  --val-json "${DATA_ROOT}/val/annotation_roipoly.json" \
  --val-path "${DATA_ROOT}/val/images" \
  "$@" \
  MODEL.WEIGHTS "${MODEL_WEIGHTS}" \
  OUTPUT_DIR "${OUTPUT_DIR}" \
  DATALOADER.NUM_WORKERS "${NUM_WORKERS}"
