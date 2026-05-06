#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${ENV_NAME:-polytopobench}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
FORCE_CUDA="${FORCE_CUDA:-1}"
MAX_JOBS="${MAX_JOBS:-8}"
CONDA_EXE="${CONDA_EXE:-conda}"

if ! command -v "$CONDA_EXE" >/dev/null 2>&1; then
  echo "conda was not found. Install Anaconda/Miniconda first, or set CONDA_EXE." >&2
  exit 1
fi

if [[ "${RESET_POLYTOPOBENCH_ENV:-0}" == "1" ]]; then
  "$CONDA_EXE" env remove -n "$ENV_NAME" -y || true
fi

if ! "$CONDA_EXE" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  "$CONDA_EXE" create -y -n "$ENV_NAME" "python=$PYTHON_VERSION" pip
fi

run_py() {
  "$CONDA_EXE" run --no-capture-output -n "$ENV_NAME" python "$@"
}

run_pip() {
  run_py -m pip "$@"
}

build_ext_inplace() {
  local build_dir="$1"
  (
    cd "$build_dir"
    TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
    FORCE_CUDA="$FORCE_CUDA" \
    MAX_JOBS="$MAX_JOBS" \
    "$CONDA_EXE" run --no-capture-output -n "$ENV_NAME" python setup.py build_ext --inplace
  )
}

run_pip install --upgrade pip setuptools wheel ninja cython packaging

run_pip install \
  torch==2.4.0+cu121 \
  torchvision==0.19.0+cu121 \
  torchaudio==2.4.0+cu121 \
  --index-url https://download.pytorch.org/whl/cu121

run_pip install -r "$ROOT/requirements.txt"

run_pip install torch-scatter==2.1.2+pt24cu121 \
  -f https://data.pyg.org/whl/torch-2.4.0+cu121.html

run_pip install mmcv==2.2.0 --no-deps \
  -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html

TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
FORCE_CUDA="$FORCE_CUDA" \
MAX_JOBS="$MAX_JOBS" \
run_pip install --no-build-isolation "git+https://github.com/facebookresearch/detectron2.git@v0.6"

# SAM2 needs iopath>=0.1.10. Detectron2 metadata asks for <0.1.10, but the
# runtime smoke tests pass with 0.1.10.
run_pip install iopath==0.1.10

build_ext_inplace "$ROOT/models/hisup/source/hisup/csrc/lib/afm_op"
build_ext_inplace "$ROOT/models/hisup/source/hisup/csrc/lib/squeeze"
build_ext_inplace "$ROOT/models/roipoly/source/roipoly/ops"
build_ext_inplace "$ROOT/models/acpvnet/source/csrc/lib/afm_op"
build_ext_inplace "$ROOT/models/acpvnet/source/csrc/lib/squeeze"
build_ext_inplace "$ROOT/models/acpvnet/source/kernels/selective_scan"

run_pip install --no-build-isolation --no-deps \
  -e "$ROOT/models/ffl/source/lydorn_utils" \
  -e "$ROOT/models/ffl/source/pytorch_lydorn" \
  -e "$ROOT/models/ffl/source" \
  -e "$ROOT/models/holitracer/source" \
  -e "$ROOT/models/gcp/source"

run_py "$ROOT/smoke_test.py" --method unet_poly --skip-imports

echo "polytopobench environment is ready: $ENV_NAME"
