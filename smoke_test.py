#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path

from main import METHODS


DATASET_TASKS = (
    ("inria_building", "building"),
    ("deventer_512_valtest_as_val", "road"),
)

REQUIRED_MODULES = (
    "torch",
    "torchvision",
    "detectron2",
    "mmcv",
    "mmengine",
    "mmdet",
    "sam2",
    "torch_scatter",
    "segmentation_models_pytorch",
    "frame_field_learning",
    "lydorn_utils",
    "torch_lydorn",
    "holitracer",
    "cv2",
    "rasterio",
    "shapely",
    "geopandas",
    "albumentations",
    "transformers",
    "torchmetrics",
)

EXECUTE_SPECS = {
    "unet_poly": {
        "mode": "train_eval",
        "set": ["NUM_WORKERS=0", "MAX_EPOCHS=1", "BATCH_SIZE=2", "MAX_TRAIN_STEPS=2", "MAX_VAL_BATCHES=1", "MAX_INFER_IMAGES=2"],
    },
    "maskrcnn_poly": {
        "mode": "train_eval",
        "set": ["NUM_WORKERS=0", "MAX_EPOCHS=1", "BATCH_SIZE=1", "MAX_TRAIN_STEPS=2", "MAX_INFER_IMAGES=2"],
    },
    "sam2_poly": {
        "mode": "infer",
        "set": ["MAX_INFER_IMAGES=1", "SCORE_THRESHOLD=0.0"],
    },
    "hisup": {
        "mode": "train_eval",
        "set": ["NUM_WORKERS=0", "IMS_PER_BATCH=1", "MAX_TRAIN_STEPS=2", "MAX_EPOCH=1", "TARGET_SIZE=128", "EVAL_NUM_WORKERS=4"],
    },
    "acpvnet": {
        "mode": "train_eval",
        "set": ["NUM_WORKERS=0", "IMS_PER_BATCH=2", "TOTAL_ITERS=2", "CHECKPOINT_PERIOD=1", "EVAL_NUM_WORKERS=4", "POLYGONIZE_NUM_WORKERS=2"],
    },
    "ffl": {
        "mode": "train_eval",
        "set": ["NUM_WORKERS=1", "TRAIN_BATCH_SIZE=1", "MAX_EPOCH=1", "SMOKE_TRAIN_TILES=1", "SMOKE_VAL_TILES=1"],
    },
    "gcp": {
        "mode": "train_eval",
        "set": ["STAGE1_NUM_WORKERS=0", "STAGE2_NUM_WORKERS=0", "STAGE1_BATCH_SIZE=1", "STAGE2_BATCH_SIZE=1", "STAGE1_EPOCHS=1", "STAGE2_EPOCHS=1", "CKPT_INTERVAL=1"],
    },
    "holitracer": {
        "mode": "train_eval",
        "set": ["MAX_H5_IMAGES=2", "VIEW_SIZE=512", "SEG_NUM_WORKERS=2", "SEG_INFER_NUM_WORKERS=2", "VECTOR_NUM_WORKERS=2"],
    },
    "pix2poly": {
        "mode": "train_eval",
        "set": ["PIX2POLY_BATCH_SIZE=1", "PIX2POLY_NUM_EPOCHS=1", "PIX2POLY_NUM_WORKERS=0", "PIX2POLY_DEBUG_MAX_TRAIN_STEPS=2", "PIX2POLY_DEBUG_MAX_VAL_STEPS=1", "PIX2POLY_PRETRAINED_ENCODER=0"],
    },
    "polyworld": {
        "mode": "train_eval",
        "set": ["NUM_WORKERS=0", "BATCH_SIZE=2", "EPOCHS=1", "MAX_TRAIN_STEPS_PER_EPOCH=2", "MAX_VAL_BATCHES=1", "TRAIN_IMAGE_LIMIT=8", "VAL_IMAGE_LIMIT=4"],
    },
    "roipoly": {
        "mode": "train_eval",
        "set": ["NUM_WORKERS=0", "IMS_PER_BATCH=2", "MAX_ITER=2", "CHECKPOINT_PERIOD=1"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PolyTopoBench smoke checks.")
    parser.add_argument("--data-processed-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=Path("output_smoke_polytopobench"))
    parser.add_argument("--method", default="all", help="Baseline name, comma list, or all.")
    parser.add_argument("--dataset", choices=[name for name, _ in DATASET_TASKS], default="inria_building")
    parser.add_argument("--env", default="polytopobench")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--bbox-json", type=Path, default=None, help="Required for executing sam2_poly unless Mask R-CNN smoke output already exists.")
    parser.add_argument("--execute", action="store_true", help="Run short smoke jobs. Without this flag only imports and dry-run commands are checked.")
    parser.add_argument("--skip-imports", action="store_true")
    return parser.parse_args()


def selected_methods(value: str) -> list[str]:
    if value == "all":
        return METHODS.copy()
    methods = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise SystemExit(f"Unknown method(s): {', '.join(unknown)}")
    return methods


def check_imports() -> None:
    for module_name in REQUIRED_MODULES:
        importlib.import_module(module_name)
    print("Dependency imports: OK", flush=True)


def task_for_dataset(dataset: str) -> str:
    for name, task in DATASET_TASKS:
        if name == dataset:
            return task
    raise ValueError(dataset)


def run_main(release_root: Path, args: argparse.Namespace, method: str, execute: bool) -> None:
    task = task_for_dataset(args.dataset)
    spec = EXECUTE_SPECS[method]
    cmd = [
        sys.executable,
        str(release_root / "main.py"),
        "--method",
        method,
        "--dataset",
        args.dataset,
        "--task",
        task,
        "--mode",
        spec["mode"],
        "--smoke",
        "--env",
        args.env,
        "--gpu",
        args.gpu,
        "--output-root",
        str(args.output_root),
    ]
    if not execute:
        cmd.append("--dry-run")
    if args.data_processed_root is not None:
        cmd += ["--data-processed-root", str(args.data_processed_root)]
    for item in spec["set"]:
        cmd += ["--set", item]
    if method == "sam2_poly":
        bbox_json = args.bbox_json or args.output_root / "maskrcnn_poly" / args.dataset / task / f"maskrcnn_poly_{task}_smoke" / "bbox_predictions_for_sam2.json"
        if execute and not (release_root / bbox_json).exists() and not bbox_json.exists():
            raise SystemExit("sam2_poly execute needs --bbox-json or an existing Mask R-CNN smoke bbox JSON.")
        cmd += ["--bbox-json", str(bbox_json)]
    subprocess.run(cmd, cwd=release_root, check=True)
    if execute:
        check_unified_metrics(release_root, args, method, task)


def check_unified_metrics(release_root: Path, args: argparse.Namespace, method: str, task: str) -> None:
    output_root = args.output_root if args.output_root.is_absolute() else release_root / args.output_root
    run_dir = output_root / method / args.dataset / task / f"{method}_{task}_smoke"
    for path in sorted(run_dir.rglob("metrics.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue  # e.g. detectron2's own line-delimited training log
        if isinstance(payload, dict) and "ground_truth" in payload and "metrics" in payload:
            ap50 = payload["metrics"].get("poly_ap", {}).get("AP50")
            print(f"Unified metrics: {path} (AP50={ap50})", flush=True)
            return
    raise SystemExit(f"{method}: no unified-evaluator metrics.json under {run_dir}")


def main() -> None:
    args = parse_args()
    release_root = Path(__file__).resolve().parent
    if not args.skip_imports:
        check_imports()

    for method in selected_methods(args.method):
        print(f"Smoke {'execute' if args.execute else 'dry-run'}: {method} / {args.dataset}", flush=True)
        run_main(release_root, args, method, args.execute)
    print(f"Smoke {'execute' if args.execute else 'dry-run'}: OK", flush=True)


if __name__ == "__main__":
    main()
