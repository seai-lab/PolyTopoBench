# Environment

PolyTopoBench is configured to use one conda environment named `polytopobench` for all 11 baselines.

## Requirements

- Linux with NVIDIA GPU.
- CUDA runtime compatible with PyTorch `2.4.0+cu121`.
- A **CUDA 12.x** `nvcc` for building Detectron2 and the custom CUDA extensions. PyTorch refuses to build extensions with an `nvcc` whose major version differs from its own (12), so CUDA 11.x or 13.x toolkits do not work.
- Conda, Git, and a C++ compiler supported by your `nvcc` (tested with GCC 11).

The install script defaults to `TORCH_CUDA_ARCH_LIST=8.6`. Override it for another GPU, for example:

```bash
TORCH_CUDA_ARCH_LIST="8.0" RESET_POLYTOPOBENCH_ENV=1 bash install_polytopobench_env.sh
```

### CUDA toolkit

The installer uses `$CUDA_HOME/bin/nvcc` if `CUDA_HOME` (or `CUDA_PATH`) is set, otherwise the `nvcc` on `PATH`, otherwise `/usr/local/cuda`. It stops before touching any conda environment if that `nvcc` is not CUDA 12.x.

If your system toolkit is not CUDA 12.x, install a CUDA 12.1 compiler toolkit with conda (about 2 GB; it needs no root access and does not affect the system CUDA) and point `CUDA_HOME` at it:

```bash
conda create -y -n cuda121 --override-channels -c nvidia/label/cuda-12.1.1 \
  cuda-nvcc cuda-cudart-dev cuda-libraries-dev
CUDA_HOME="$(conda run -n cuda121 printenv CONDA_PREFIX)" \
  RESET_POLYTOPOBENCH_ENV=1 bash install_polytopobench_env.sh
```

You only need the toolkit to build the environment. Using a newer 12.x toolkit, such as 12.4, also works, but PyTorch will print a harmless minor-version mismatch warning.

## Install

```bash
cd PolyTopoBench
RESET_POLYTOPOBENCH_ENV=1 bash install_polytopobench_env.sh
conda activate polytopobench
```

`requirements.txt` lists the regular Python packages. Use `install_polytopobench_env.sh` rather than `pip install -r requirements.txt` alone because Torch, mmcv, torch-scatter, Detectron2, editable packages, and CUDA extensions must be installed in a specific order.

The installer calls the environment's own interpreter by absolute path, so it is safe to run from a shell that has another conda env or virtualenv on `PATH`. The CUDA extensions are compiled in place under `models/*/source`. At the end, the installer runs the dependency import check from `smoke_test.py`. It does not need the dataset, so you can install the environment before downloading the data. Pip prints an `iopath` conflict between Detectron2 (`<0.1.10`) and SAM2 (`>=0.1.10`). This is expected: the installer pins `iopath==0.1.10`, which both work with at runtime.

## Smoke Test

Fast import and command dry-run:

```bash
python smoke_test.py
```

Run a real short baseline job:

```bash
python smoke_test.py --execute --method unet_poly
python smoke_test.py --execute --method maskrcnn_poly
```

SAM2 needs bbox priors from Mask R-CNN:

```bash
python smoke_test.py --execute --method maskrcnn_poly
python smoke_test.py --execute --method sam2_poly
```
