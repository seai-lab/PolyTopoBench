#!/usr/bin/env python3
"""SAM2 inference-only baseline prompted by maskrcnn_seg bboxes.

Reads:
- `--mirror-root`                    : sam2_seg mirror with val/images and val/annotation.json
- `--bbox-json` (json)              : list of {image_id, bbox_xyxy, score}, produced by maskrcnn_seg
- `--sam2-checkpoint` + `--sam2-config` : SAM2 weights + hydra config
- `--score-threshold` (default 0.5)  : filter on detector score

Writes `<output>/predictions.json` in the HiSup flat list schema.

Handles SAM2 returning masks with variable shape: (N, H, W), (N, 1, H, W),
or with a singleton multimask dim. Always normalizes to per-instance (H, W).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mask_to_hisup import mask_to_hisup_rings

Image.MAX_IMAGE_PIXELS = None


def _load_image_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _normalize_mask_output(mask_out) -> np.ndarray:
    """Normalize SAM2's predict() output to a `(N, H, W)` numpy array of 0/1 bools."""
    arr = mask_out
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr)
    # squeeze any singleton dims except the leading N
    while arr.ndim > 3 and 1 in arr.shape[1:]:
        axis = 1 + list(arr.shape[1:]).index(1)
        arr = np.squeeze(arr, axis=axis)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3:
        raise ValueError(f"Unexpected SAM2 mask shape after normalize: {arr.shape}")
    return arr.astype(bool)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mirror-root", type=Path, required=True)
    p.add_argument("--bbox-json", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--score-threshold", type=float, default=0.5)
    p.add_argument("--category-id", type=int, default=100)
    p.add_argument("--simplify-tol", type=float, default=1.0)
    p.add_argument("--min-area", type=float, default=16.0)
    p.add_argument("--min-hole-area", type=float, default=16.0)
    p.add_argument("--connectivity", type=int, default=4)
    p.add_argument("--sam2-model-id", type=str, default="facebook/sam2-hiera-small",
                   help="HuggingFace model id passed to SAM2ImagePredictor.from_pretrained().")
    p.add_argument("--sam2-checkpoint", type=Path, default=None,
                   help="Local SAM2 checkpoint path (used if --sam2-config is set).")
    p.add_argument("--sam2-config", type=str, default=None,
                   help="Hydra config path for build_sam2, e.g. sam2_hiera_s.yaml.")
    p.add_argument("--summary-out", type=Path, default=None,
                   help="Optional path to write a small summary json.")
    p.add_argument("--max-images", type=int, default=0)
    return p.parse_args()


def _build_sam2_predictor(args):
    try:
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except Exception as exc:
        raise SystemExit(f"Cannot import sam2.sam2_image_predictor; did you `pip install sam2`? err={exc}") from exc
    if args.sam2_checkpoint is not None and args.sam2_config is not None:
        from sam2.build_sam import build_sam2
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_sam2(args.sam2_config, str(args.sam2_checkpoint), device=device)
        return SAM2ImagePredictor(model)
    return SAM2ImagePredictor.from_pretrained(args.sam2_model_id)


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.bbox_json.open("r") as f:
        raw_bboxes = json.load(f)
    raw_count = len(raw_bboxes)

    filtered = [b for b in raw_bboxes if float(b.get("score", 0.0)) >= args.score_threshold]
    filtered_count = len(filtered)
    if filtered_count == 0:
        raise RuntimeError(
            f"No bbox in {args.bbox_json} after score>={args.score_threshold} filter "
            f"(raw={raw_count}); cannot run SAM2 inference."
        )

    # Group bboxes per image_id
    by_image: dict[int, list[dict]] = defaultdict(list)
    for b in filtered:
        by_image[int(b["image_id"])].append(b)

    val_dir = args.mirror_root / "val"
    ann_json_path = val_dir / "annotation.json"
    if not ann_json_path.exists():
        raise FileNotFoundError(f"Missing val annotation.json: {ann_json_path}")
    with ann_json_path.open("r") as f:
        coco = json.load(f)
    image_id_to_filename = {int(img["id"]): img["file_name"] for img in coco["images"]}
    image_id_to_hw = {
        int(img["id"]): (int(img["height"]), int(img["width"])) for img in coco["images"]
    }

    # Guard against bbox priors from another dataset/task/split: image ids are contiguous
    # from 1 in every mirror, so a wrong --bbox-json would otherwise be silently accepted.
    mismatched = [
        b for b in filtered
        if "file_name" in b and image_id_to_filename.get(int(b["image_id"])) != b["file_name"]
    ]
    if mismatched:
        b = mismatched[0]
        raise RuntimeError(
            f"{len(mismatched)} bbox record(s) in {args.bbox_json} do not match {ann_json_path}: "
            f"image_id={b['image_id']} is '{b['file_name']}' in the bbox JSON but "
            f"'{image_id_to_filename.get(int(b['image_id']))}' in the val split. "
            "Use Mask R-CNN predictions from the same dataset/task."
        )
    unknown_ids = sorted(set(by_image) - set(image_id_to_filename))
    if unknown_ids:
        raise RuntimeError(
            f"{len(unknown_ids)} image_id(s) in {args.bbox_json} are not in {ann_json_path} "
            f"(e.g. {unknown_ids[:5]}). Use Mask R-CNN predictions from the same dataset/task."
        )

    predictor = _build_sam2_predictor(args)

    records: list[dict] = []
    next_id = 1
    total_polys = 0
    t0 = time.time()

    sorted_image_ids = sorted(set(by_image.keys()) | set(image_id_to_filename.keys()))

    with torch.inference_mode():
        processed_images = 0
        for image_id in tqdm(sorted_image_ids, desc=f"sam2_seg infer [{args.mirror_root.name}]"):
            if image_id not in image_id_to_filename:
                continue
            these_bboxes = by_image.get(image_id, [])
            if not these_bboxes:
                continue
            if args.max_images and processed_images >= args.max_images:
                break

            fn = image_id_to_filename[image_id]
            img_path = val_dir / "images" / fn
            image = _load_image_rgb(img_path)
            predictor.set_image(image)

            boxes_xyxy = np.asarray([b["bbox_xyxy"] for b in these_bboxes], dtype=np.float32)
            scores_det = np.asarray([b["score"] for b in these_bboxes], dtype=np.float32)

            try:
                masks_out, iou_preds, _low_res = predictor.predict(
                    box=boxes_xyxy,
                    multimask_output=False,
                )
            except Exception as exc:
                print(f"[WARN] image_id={image_id} SAM2 predict failed: {exc}", flush=True)
                continue
            mask_arr = _normalize_mask_output(masks_out)
            H, W = image_id_to_hw.get(image_id, image.shape[:2])
            # If SAM2 returned a different H/W, resize via argmax (rare; H/W == image shape expected)
            for k in range(mask_arr.shape[0]):
                mask_bin = mask_arr[k].astype(np.uint8)
                if mask_bin.shape != (H, W):
                    # fallback: re-raster via PIL nearest
                    mask_img = Image.fromarray((mask_bin * 255).astype(np.uint8))
                    mask_img = mask_img.resize((W, H), resample=Image.NEAREST)
                    mask_bin = (np.asarray(mask_img) > 127).astype(np.uint8)
                if mask_bin.sum() == 0:
                    continue
                polys = mask_to_hisup_rings(
                    mask_bin,
                    simplify_tol=args.simplify_tol,
                    min_area=args.min_area,
                    min_hole_area=args.min_hole_area,
                    connectivity=args.connectivity,
                    score_map=None,
                )
                det_score = float(scores_det[k])
                for p in polys:
                    records.append({
                        "id": next_id,
                        "image_id": image_id,
                        "category_id": args.category_id,
                        "segmentation": p["rings"],
                        "bbox": p["bbox"],
                        "area": p["area"],
                        "score": det_score,
                    })
                    next_id += 1
                    total_polys += 1
            processed_images += 1

    with args.output.open("w") as f:
        json.dump(records, f)

    summary = {
        "raw_bbox_count": raw_count,
        "filtered_bbox_count": filtered_count,
        "score_threshold": args.score_threshold,
        "num_predictions": total_polys,
        "bbox_source": str(args.bbox_json),
        "sam2_model_id": args.sam2_model_id,
        "elapsed_sec": time.time() - t0,
    }
    if args.summary_out is not None:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        with args.summary_out.open("w") as f:
            json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
