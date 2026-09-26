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

# Detectron2 and the in-tree CUDA extensions are compiled against
# torch 2.4.0+cu121, which requires a CUDA 12.x nvcc. Honour CUDA_HOME /
# CUDA_PATH, otherwise use the nvcc on PATH (same lookup order as torch).
CUDA_HOME="${CUDA_HOME:-${CUDA_PATH:-}}"
if [[ -z "$CUDA_HOME" ]]; then
  if command -v nvcc >/dev/null 2>&1; then
    CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
  else
    CUDA_HOME=/usr/local/cuda
  fi
fi
if [[ ! -x "$CUDA_HOME/bin/nvcc" ]]; then
  echo "nvcc was not found at $CUDA_HOME/bin/nvcc. Set CUDA_HOME to a CUDA 12.x toolkit (see ENVIRONMENT.md)." >&2
  exit 1
fi
CUDA_TOOLKIT_VERSION="$("$CUDA_HOME/bin/nvcc" --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p')"
if [[ "${CUDA_TOOLKIT_VERSION%%.*}" != "12" ]]; then
  echo "Found nvcc ${CUDA_TOOLKIT_VERSION:-unknown} at $CUDA_HOME/bin/nvcc, but torch 2.4.0+cu121 needs a CUDA 12.x nvcc." >&2
  echo "Set CUDA_HOME to a CUDA 12.x toolkit (see ENVIRONMENT.md)." >&2
  exit 1
fi
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
echo "Using CUDA $CUDA_TOOLKIT_VERSION toolkit at $CUDA_HOME"

if [[ "${RESET_POLYTOPOBENCH_ENV:-0}" == "1" ]]; then
  "$CONDA_EXE" env remove -n "$ENV_NAME" -y || true
fi

if ! "$CONDA_EXE" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  "$CONDA_EXE" create -y -n "$ENV_NAME" "python=$PYTHON_VERSION" pip
fi

# Call the env's interpreter by absolute path. `conda run -n ENV python` picks
# the first `python` on PATH, which can belong to another env or venv when the
# calling shell has one prepended to PATH; pip would then modify that env.
ENV_PREFIX="$("$CONDA_EXE" run -n "$ENV_NAME" printenv CONDA_PREFIX)"
ENV_PY="$ENV_PREFIX/bin/python"
if [[ ! -x "$ENV_PY" ]]; then
  echo "Could not locate the python interpreter of conda env '$ENV_NAME' (got '$ENV_PY')." >&2
  exit 1
fi
export PATH="$ENV_PREFIX/bin:$PATH"
export PYTHONNOUSERSITE=1

run_py() {
  "$CONDA_EXE" run --no-capture-output -n "$ENV_NAME" "$ENV_PY" "$@"
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
    run_py setup.py build_ext --inplace
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

# Import check only: the dry-run in smoke_test.py needs dataset/data_processed,
# which is usually downloaded after the environment is installed.
(cd "$ROOT" && run_py -c "import smoke_test; smoke_test.check_imports()")

echo "polytopobench environment is ready: $ENV_NAME"
