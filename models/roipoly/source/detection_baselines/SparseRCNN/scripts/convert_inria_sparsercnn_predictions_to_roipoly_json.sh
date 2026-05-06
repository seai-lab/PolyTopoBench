#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPARSERCNN_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROIPOLY_ROOT="$(cd "${SPARSERCNN_ROOT}/../.." && pwd)"
PROJECT_ROOT="$(cd "${SPARSERCNN_ROOT}/../../../../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data_processed/inria_building/roipoly}"

CONDA_ENV="${CONDA_ENV:-polytopobench}"
DETECTOR_OUTPUT_DIR="${DETECTOR_OUTPUT_DIR:-${PROJECT_ROOT}/output/sparsercnn/inria_building/building/res50_160pro_512_eval_val}"
PREDICTIONS_JSON="${PREDICTIONS_JSON:-${DETECTOR_OUTPUT_DIR}/inference/coco_instances_results.json}"
SAVE_PATH="${SAVE_PATH:-${DETECTOR_OUTPUT_DIR}/inference/annotation_sparsercnn_val_for_roipoly.json}"
SCORE_THRESHOLD="${SCORE_THRESHOLD:-0.05}"
TOPK_PER_IMAGE="${TOPK_PER_IMAGE:-160}"

cd "${ROIPOLY_ROOT}"
conda run -n "${CONDA_ENV}" python data_preprocessing/predictions_to_coco.py \
  --json_path "${PREDICTIONS_JSON}" \
  --annotation_path "${DATA_ROOT}/val/annotation_roipoly.json" \
  --save_path "${SAVE_PATH}" \
  --type all \
  --score_threshold "${SCORE_THRESHOLD}" \
  --topk_per_image "${TOPK_PER_IMAGE}" \
  "$@"
