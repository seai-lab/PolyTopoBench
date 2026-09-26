#!/usr/bin/env python3
"""Inference for maskrcnn_seg.

Outputs:
- `<output-dir>/predictions.json`                  (HiSup-format polygon records)
- `<output-dir>/bbox_predictions_for_sam2.json`    (list of {image_id, file_name, bbox_xyxy, score})
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mask_to_hisup import mask_to_hisup_rings
from maskrcnn_seg.dataset import MaskRcnnValImageDataset, collate_val
from maskrcnn_seg.train import build_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mirror-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--predictions-out", type=Path, required=True)
    p.add_argument("--bbox-out", type=Path, required=True)
    p.add_argument("--category-id", type=int, default=100)
    p.add_argument("--score-threshold", type=float, default=0.05)
    p.add_argument("--mask-threshold", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--simplify-tol", type=float, default=1.0)
    p.add_argument("--min-area", type=float, default=16.0)
    p.add_argument("--min-hole-area", type=float, default=16.0)
    p.add_argument("--connectivity", type=int, default=4)
    p.add_argument("--max-images", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.predictions_out.parent.mkdir(parents=True, exist_ok=True)
    args.bbox_out.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    model.eval()

    ds = MaskRcnnValImageDataset(args.mirror_root)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_val,
    )

    predictions: list[dict] = []
    bbox_records: list[dict] = []
    next_id = 1
    total_polys = 0
    processed_images = 0
    t0 = time.time()

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"maskrcnn_seg infer [{args.mirror_root.name}]"):
            images = [img.to(device, non_blocking=True) for img in batch["image"]]
            outputs = model(images)
            for i, out in enumerate(outputs):
                image_id = int(batch["image_id"][i])
                if args.max_images and processed_images >= args.max_images:
                    break
                H = int(batch["height"][i])
                W = int(batch["width"][i])

                scores = out["scores"].detach().cpu().numpy()
                masks_soft = out["masks"].detach().cpu().numpy()  # (N, 1, H, W)
                boxes = out["boxes"].detach().cpu().numpy()      # (N, 4) xyxy

                for j in range(scores.shape[0]):
                    score = float(scores[j])
                    if score < args.score_threshold:
                        continue
                    box = boxes[j].tolist()
                    bbox_records.append({
                        "image_id": image_id,
                        # lets sam2 verify the image_id refers to the same val patch
                        "file_name": batch["file_name"][i],
                        "bbox_xyxy": [float(v) for v in box],
                        "score": score,
                    })
                    soft = masks_soft[j, 0]
                    mask_bin = (soft > args.mask_threshold).astype(np.uint8)
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
                    for p in polys:
                        predictions.append({
                            "id": next_id,
                            "image_id": image_id,
                            "category_id": args.category_id,
                            "segmentation": p["rings"],
                            "bbox": p["bbox"],
                            "area": p["area"],
                            "score": score,
                        })
                        next_id += 1
                        total_polys += 1
                processed_images += 1
            if args.max_images and processed_images >= args.max_images:
                break

    with args.predictions_out.open("w") as f:
        json.dump(predictions, f)
    with args.bbox_out.open("w") as f:
        json.dump(bbox_records, f)

    elapsed = time.time() - t0
    print(
        f"[maskrcnn_seg infer done] polygons={total_polys} bboxes={len(bbox_records)} "
        f"elapsed={elapsed:.1f}s -> {args.predictions_out}"
    )


if __name__ == "__main__":
    main()
