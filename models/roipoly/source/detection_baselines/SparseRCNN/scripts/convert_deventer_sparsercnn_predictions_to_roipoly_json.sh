#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPARSERCNN_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROIPOLY_ROOT="$(cd "${SPARSERCNN_ROOT}/../.." && pwd)"
PROJECT_ROOT="$(cd "${SPARSERCNN_ROOT}/../../../../.." && pwd)"
DEVENTER_VARIANT="${DEVENTER_VARIANT:-deventer_512_valtest_as_val}"
DEVENTER_PROCESSED_ROOT="${DEVENTER_PROCESSED_ROOT:-${PROJECT_ROOT}/data_processed/${DEVENTER_VARIANT}}"
DEVENTER_OUTPUT_TAG="${DEVENTER_OUTPUT_TAG:-${DEVENTER_VARIANT}}"

SCENARIO="${SCENARIO:-road}"
SPLIT="${SPLIT:-val}"
DATA_ROOT="${DATA_ROOT:-${DEVENTER_PROCESSED_ROOT}/roipoly/${SCENARIO}}"
CONDA_ENV="${CONDA_ENV:-polytopobench}"
USE_SMOKE_ANN="${USE_SMOKE_ANN:-0}"
SCORE_THRESHOLD="${SCORE_THRESHOLD:-0.05}"
TOPK_PER_IMAGE="${TOPK_PER_IMAGE:-160}"

JSON_NAME="annotation_roipoly.json"
RUN_SUFFIX=""
SAVE_SUFFIX=""
if [[ "${USE_SMOKE_ANN}" == "1" ]]; then
  JSON_NAME="annotation_roipoly_smoke.json"
  RUN_SUFFIX="_smoke"
  SAVE_SUFFIX="_smoke"
fi

DETECTOR_OUTPUT_DIR="${DETECTOR_OUTPUT_DIR:-${PROJECT_ROOT}/output/sparsercnn/${DEVENTER_OUTPUT_TAG}/${SCENARIO}/res50_160pro_512${RUN_SUFFIX}_eval_${SPLIT}}"
PREDICTIONS_JSON="${PREDICTIONS_JSON:-${DETECTOR_OUTPUT_DIR}/inference/coco_instances_results.json}"
SAVE_PATH="${SAVE_PATH:-${DETECTOR_OUTPUT_DIR}/inference/annotation_sparsercnn_${SPLIT}_for_roipoly${SAVE_SUFFIX}.json}"

mkdir -p "$(dirname "${SAVE_PATH}")"
cd "${ROIPOLY_ROOT}"
conda run -n "${CONDA_ENV}" python data_preprocessing/predictions_to_coco.py \
  --json_path "${PREDICTIONS_JSON}" \
  --annotation_path "${DATA_ROOT}/${SPLIT}/${JSON_NAME}" \
  --save_path "${SAVE_PATH}" \
  --type all \
  --score_threshold "${SCORE_THRESHOLD}" \
  --topk_per_image "${TOPK_PER_IMAGE}" \
  "$@"
