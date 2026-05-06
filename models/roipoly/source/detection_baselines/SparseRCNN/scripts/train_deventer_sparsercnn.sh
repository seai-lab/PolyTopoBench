#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPARSERCNN_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${SPARSERCNN_ROOT}/../../../../.." && pwd)"
DEVENTER_VARIANT="${DEVENTER_VARIANT:-deventer_512_valtest_as_val}"
DEVENTER_PROCESSED_ROOT="${DEVENTER_PROCESSED_ROOT:-${PROJECT_ROOT}/data_processed/${DEVENTER_VARIANT}}"
DEVENTER_OUTPUT_TAG="${DEVENTER_OUTPUT_TAG:-${DEVENTER_VARIANT}}"

SCENARIO="${SCENARIO:-road}"
DATA_ROOT="${DATA_ROOT:-${DEVENTER_PROCESSED_ROOT}/roipoly/${SCENARIO}}"
GPU_ID="${GPU_ID:-0}"
CONDA_ENV="${CONDA_ENV:-polytopobench}"
CONFIG_FILE="${CONFIG_FILE:-configs/sparsercnn.res50.160pro.deventer512.yaml}"
MODEL_WEIGHTS="${MODEL_WEIGHTS:-detectron2://ImageNetPretrained/torchvision/R-50.pkl}"
IMS_PER_BATCH="${IMS_PER_BATCH:-4}"
BASE_LR="${BASE_LR:-0.000025}"
NUM_WORKERS="${NUM_WORKERS:-8}"
USE_SMOKE_ANN="${USE_SMOKE_ANN:-0}"

if [[ "${USE_SMOKE_ANN}" == "1" ]]; then
  MAX_ITER="${MAX_ITER:-2}"
  STEP1="${STEP1:-1}"
  STEP2="${STEP2:-}"
  CHECKPOINT_PERIOD="${CHECKPOINT_PERIOD:-1}"
  EVAL_PERIOD="${EVAL_PERIOD:-1}"
else
  MAX_ITER="${MAX_ITER:-36120}"
  STEP1="${STEP1:-27090}"
  STEP2="${STEP2:-33198}"
  CHECKPOINT_PERIOD="${CHECKPOINT_PERIOD:-1505}"
  EVAL_PERIOD="${EVAL_PERIOD:-1505}"
fi

JSON_NAME="annotation_roipoly.json"
RUN_SUFFIX=""
if [[ "${USE_SMOKE_ANN}" == "1" ]]; then
  JSON_NAME="annotation_roipoly_smoke.json"
  RUN_SUFFIX="_smoke"
fi

OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/output/sparsercnn/${DEVENTER_OUTPUT_TAG}/${SCENARIO}/res50_160pro_512${RUN_SUFFIX}}"

solver_steps=()
for raw_step in "${STEP1}" "${STEP2}"; do
  if [[ -z "${raw_step}" ]]; then
    continue
  fi
  if [[ "${raw_step}" =~ ^[0-9]+$ ]] && (( raw_step > 0 && raw_step < MAX_ITER )); then
    solver_steps+=("${raw_step}")
  fi
done

if (( ${#solver_steps[@]} > 0 )); then
  mapfile -t solver_steps < <(printf '%s\n' "${solver_steps[@]}" | sort -n -u)
  if (( ${#solver_steps[@]} == 1 )); then
    solver_steps_arg="(${solver_steps[0]},)"
  else
    solver_steps_arg="($(IFS=', '; echo "${solver_steps[*]}"))"
  fi
else
  solver_steps_arg="()"
fi

mkdir -p "${OUTPUT_DIR}"
cd "${SPARSERCNN_ROOT}"
PYTHONPATH="${SPARSERCNN_ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU_ID}" conda run -n "${CONDA_ENV}" \
  python train_net.py \
  --num-gpus 1 \
  --config-file "${CONFIG_FILE}" \
  --train-dataset deventer_train \
  --train-json "${DATA_ROOT}/train/${JSON_NAME}" \
  --train-path "${DATA_ROOT}/train/images" \
  --val-dataset deventer_val \
  --val-json "${DATA_ROOT}/val/${JSON_NAME}" \
  --val-path "${DATA_ROOT}/val/images" \
  "$@" \
  MODEL.WEIGHTS "${MODEL_WEIGHTS}" \
  OUTPUT_DIR "${OUTPUT_DIR}" \
  SOLVER.IMS_PER_BATCH "${IMS_PER_BATCH}" \
  SOLVER.BASE_LR "${BASE_LR}" \
  SOLVER.MAX_ITER "${MAX_ITER}" \
  SOLVER.STEPS "${solver_steps_arg}" \
  SOLVER.CHECKPOINT_PERIOD "${CHECKPOINT_PERIOD}" \
  TEST.EVAL_PERIOD "${EVAL_PERIOD}" \
  DATALOADER.NUM_WORKERS "${NUM_WORKERS}"
