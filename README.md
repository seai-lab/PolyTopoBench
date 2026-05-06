# PolyTopoBench

This repository provides a unified runner for 11 polygon decoding baselines.

## Installation

```bash
cd PolyTopoBench
RESET_POLYTOPOBENCH_ENV=1 bash install_polytopobench_env.sh
conda activate polytopobench
```

The installer creates one conda environment named `polytopobench`. See `ENVIRONMENT.md` for CUDA and compiler requirements.

## Data

Data is distributed separately. After downloading the data package, place or extract it under `dataset/` with this layout:

```text
dataset/data_processed/
  inria_building/
  deventer_512_valtest_as_val/
```

Raw source data used by `prepare_data.py` should be placed under:

```text
dataset/
  inria_dataset_aligned/
  deventer_512_valtest_as_val/
```

For a custom processed data location, pass `--data-processed-root`.

The prepared `dataset/data_processed/` folders can be used directly. To rebuild the processed folders from raw data for the baselines that do not require ACPV-Net latents, run:

```bash
python prepare_data.py \
  --overwrite
```

To include ACPV-Net when rebuilding data, pass `--methods all --encode-acpv-latents --acpv-autoencoder-config <path>`. By default, prepared data is written to `dataset/data_processed/`. The Deventer tasks are `road`, `vegetation`, and `unvegetated`.

## Quick Check

```bash
python smoke_test.py
```

This checks imports and dry-runs all baseline commands. To run a real short job:

```bash
python smoke_test.py --execute --method unet_poly
```

## Running Baselines

List available methods:

```bash
python main.py --list
```

Run an Inria building baseline:

```bash
python main.py --method unet_poly --dataset inria_building --task building --smoke
```

Run a Deventer baseline:

```bash
python main.py --method hisup --dataset deventer_512_valtest_as_val --task road --smoke
```

Run training followed by evaluation:

```bash
python main.py --method unet_poly --dataset inria_building --task building --mode train_eval --smoke
```

Run evaluation for an existing run:

```bash
python main.py --method unet_poly --dataset inria_building --task building --mode eval --run-name unet_poly_building_smoke
```

Evaluation metrics are written to `output/<method>/<dataset>/<task>/<run-name>/metrics.json`.

Run SAM2 with Mask R-CNN bbox priors:

```bash
python main.py --method sam2_poly --dataset inria_building --task building --mode infer --bbox-json output/maskrcnn_poly/inria_building/building/<run>/bbox_predictions_for_sam2.json
```

Outputs are written to `output/` by default. Use `--output-root` to change this. Use `--dry-run` to print commands without running them, and `--set KEY=VALUE` to override method-specific settings.
