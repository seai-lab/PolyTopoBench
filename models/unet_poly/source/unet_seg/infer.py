#!/usr/bin/env python3
"""Inference for unet_seg: softmax → argmax → binary mask → HiSup rings.

Writes `predictions.json` (flat HiSup list) at `<output-dir>/predictions.json`.
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
from unet_seg.dataset import UnetSegValDataset
from unet_seg.train import build_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mirror-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--category-id", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--simplify-tol", type=float, default=1.0)
    p.add_argument("--min-area", type=float, default=16.0)
    p.add_argument("--min-hole-area", type=float, default=16.0)
    p.add_argument("--connectivity", type=int, default=4)
    p.add_argument("--max-images", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    model.eval()

    ds = UnetSegValDataset(args.mirror_root)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    t0 = time.time()
    records: list[dict] = []
    next_id = 1
    total_fg_poly = 0
    processed_images = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"unet_seg infer [{args.mirror_root.name}]"):
            images = batch["image"].to(device, non_blocking=True)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1).cpu().numpy().astype(np.uint8)
            fg_probs = probs[:, 1].cpu().numpy().astype(np.float32)

            for i in range(preds.shape[0]):
                image_id = int(batch["image_id"][i])
                if args.max_images and processed_images >= args.max_images:
                    break
                mask_i = preds[i]
                score_i = fg_probs[i]
                polys = mask_to_hisup_rings(
                    mask_i,
                    simplify_tol=args.simplify_tol,
                    min_area=args.min_area,
                    min_hole_area=args.min_hole_area,
                    connectivity=args.connectivity,
                    score_map=score_i,
                )
                for p in polys:
                    records.append({
                        "id": next_id,
                        "image_id": image_id,
                        "category_id": args.category_id,
                        "segmentation": p["rings"],
                        "bbox": p["bbox"],
                        "area": p["area"],
                        "score": float(p["score"]) if p["score"] is not None else 0.0,
                    })
                    next_id += 1
                    total_fg_poly += 1
                processed_images += 1
            if args.max_images and processed_images >= args.max_images:
                break

    with args.output.open("w") as f:
        json.dump(records, f)
    elapsed = time.time() - t0
    print(f"[unet_seg infer done] polygons={total_fg_poly} elapsed={elapsed:.1f}s -> {args.output}")


if __name__ == "__main__":
    main()
