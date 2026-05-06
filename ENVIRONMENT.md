# Environment

PolyTopoBench is configured to use one conda environment named `polytopobench` for all 11 baselines.

## Requirements

- Linux with NVIDIA GPU.
- CUDA runtime compatible with PyTorch `2.4.0+cu121`.
- `nvcc` available for building Detectron2 and the custom CUDA extensions.
- Conda, Git, and a C++ compiler.

The install script defaults to `TORCH_CUDA_ARCH_LIST=8.6`. Override it for another GPU, for example:

```bash
TORCH_CUDA_ARCH_LIST="8.0" RESET_POLYTOPOBENCH_ENV=1 bash install_polytopobench_env.sh
```

## Install

```bash
cd PolyTopoBench
RESET_POLYTOPOBENCH_ENV=1 bash install_polytopobench_env.sh
conda activate polytopobench
```

`requirements.txt` lists the regular Python packages. Use `install_polytopobench_env.sh` rather than `pip install -r requirements.txt` alone because Torch, mmcv, torch-scatter, Detectron2, editable packages, and CUDA extensions must be installed in a specific order.

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
