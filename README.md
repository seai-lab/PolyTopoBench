# PolyTopoBench

This repository provides a unified runner for 11 polygon decoding baselines.

## Installation

```bash
cd PolyTopoBench
RESET_POLYTOPOBENCH_ENV=1 bash install_polytopobench_env.sh
conda activate polytopobench
```

The installer creates one conda environment named `polytopobench` for all baselines. It needs a CUDA 12.x `nvcc`; see `ENVIRONMENT.md` for how to point it to one with `CUDA_HOME`.

## Data

The dataset is hosted at https://huggingface.co/datasets/PingL/PolyTopoBench. Download it into `dataset/`:

```bash
pip install -U huggingface_hub
# Benchmark data: 512x512 patches and annotations for all four tasks (~13 GB)
hf download PingL/PolyTopoBench --repo-type dataset --local-dir dataset \
  --include "tasks.json" "inria/*" "deventer/*"
# Optional: full Inria tiles, needed only for Frame Field Learning on Inria (~13 GB)
hf download PingL/PolyTopoBench --repo-type dataset --local-dir dataset --include "raw/inria/*"
```

This creates:

```text
dataset/
  tasks.json                  # task registry with file paths and checksums
  inria/
    images/{train,val}/<city>/*.tif
    building/{train,val}.json # COCO layout; segmentation = [exterior, hole_1, ...]
    building/{train,val}.parquet
  deventer/
    images/{train,val}/*.png
    {road,vegetation,unvegetated}/{train,val}.{json,parquet}
  raw/inria/                  # optional
```

See the [dataset card](https://huggingface.co/datasets/PingL/PolyTopoBench) for the annotation format. Note that `segmentation` lists the exterior ring followed by its holes, which differs from standard COCO.

### Preparing data for the baselines

Each baseline reads its own input format. `prepare_data.py` derives these from the downloaded release and writes them to `dataset/data_processed/`:

```bash
python prepare_data.py                          # all baselines except ACPV-Net
python prepare_data.py --methods hisup roipoly  # only the baselines you need
```

The canonical ground truth used by the evaluator (`dataset/data_processed/<dataset>/hisup/...`) is always prepared. Mirrors that already exist are skipped, so you can add baselines later; pass `--overwrite` to rebuild them. Preparing everything takes about 15 minutes and 30 GB of disk (images are hard-linked where possible). FFL on Inria is skipped with a message if `raw/inria/` was not downloaded.

ACPV-Net additionally needs latent heatmaps encoded on the GPU: pass `--methods acpvnet --encode-acpv-latents --acpv-autoencoder-config <path>`.

Use `--data-root` and `--output-root` to read the release from, or write the prepared data to, another location; then pass the same output location to `main.py` and `smoke_test.py` with `--data-processed-root`. You do not need `prepare_data.py` to evaluate your own method; see below.

## Evaluating Your Own Method

Train on `dataset/<dataset>/images/train` with the matching `train.json`, predict on the validation images, and write one JSON list per task:

```json
[{"image_id": 1, "segmentation": [[x1, y1, x2, y2, ...], [hole ring], ...], "score": 0.93}, ...]
```

`image_id` must be the id from the task's `val.json`. Each record is one polygon: the first ring is its exterior and every further ring is a hole, in pixel coordinates of the 512x512 patch. Then run the unified evaluator:

```bash
python utilis/evaluate_vector_polygons.py \
  --pred predictions.json \
  --gt dataset/inria/building/val.json \
  --gt-type hisup --pred-type hisup \
  --min-hole-area 16 --num-workers 16 \
  --output metrics.json
```

If your method outputs one record per ring instead of one per polygon, use `--pred-type roipoly`; holes are then assigned to exteriors by containment.

## Quick Check

After preparing the data:

```bash
python smoke_test.py                                   # dependency imports + dry-run of every baseline
python smoke_test.py --execute --method unet_poly      # real short run on Inria
python smoke_test.py --execute --method hisup --dataset deventer_512_valtest_as_val
```

With `--execute`, each baseline is trained for a few steps, run on a few validation images and scored with the unified evaluator. The check fails if no metrics file is produced. Run `maskrcnn_poly` before `sam2_poly`, which uses its boxes. Smoke outputs go to `output_smoke_polytopobench/`.

## Running Baselines

List available methods:

```bash
python main.py --list
```

Datasets are `inria_building` (task `building`) and `deventer_512_valtest_as_val` (tasks `road`, `vegetation`, `unvegetated`). Train and evaluate a baseline:

```bash
python main.py --method hisup --dataset inria_building --task building --mode train_eval
python main.py --method hisup --dataset deventer_512_valtest_as_val --task road --mode train_eval
```

Add `--smoke` for a short run. Other modes are `train`, `infer` and `eval`. To evaluate an existing run, pass its name:

```bash
python main.py --method unet_poly --dataset inria_building --task building --mode eval --run-name unet_poly_building_train_eval
```

Outputs are written to `output/<method>/<dataset>/<task>/<run-name>/`; use `--output-root` to change this. `train_eval` ends with the unified evaluator, which writes `metrics.json` to:

| Method | Metrics file |
|---|---|
| `ffl`, `gcp`, `holitracer` | `<run-dir>/val_inference/metrics.json` |
| `roipoly` | `<run-dir>/eval_sparsercnn_val/metrics.json` |
| all others | `<run-dir>/metrics.json` |

Use `--dry-run` to print the commands without running them. Method-specific settings are changed with `--set KEY=VALUE` (repeatable), or with an environment variable `POLYTOPOBENCH_<KEY>`. Plain environment variables such as `NUM_WORKERS` are ignored.

### Baseline notes

- **Pretrained weights** for U-Net (EfficientNet-B3), Mask R-CNN and SAM2 (`facebook/sam2-hiera-small`) are downloaded automatically on first use.
- **SAM2** is not trained. It prompts SAM2 with Mask R-CNN boxes, so run `maskrcnn_poly` first, or pass `--bbox-json`.
- **ACPV-Net** needs the LDM kl-f4 autoencoder to encode its training targets. Download it before running `prepare_data.py --methods acpvnet --encode-acpv-latents`:
  ```bash
  curl -L -o kl-f4.zip https://ommer-lab.com/files/latent-diffusion/kl-f4.zip
  unzip kl-f4.zip -d models/acpvnet/source/models/first_stage_models/kl-f4
  ```
  Encoding runs on the GPU. It takes about 20 minutes and 33 GB for Inria.
- **RoIPoly** needs box proposals. `train_eval` first trains its Sparse R-CNN detector unless `--set DETECTOR_WEIGHTS=<path>` is given.
- **FFL** on Inria trains and predicts on the full 5000 × 5000 tiles from `raw/inria/`. Its predictions are mapped back to the 512 × 512 benchmark patches before evaluation.
