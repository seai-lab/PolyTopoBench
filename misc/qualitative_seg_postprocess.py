#!/usr/bin/env python3
"""Produce a small qualitative inspection for seg+postprocess runs.

For a (model, dataset, scenario) run, picks 3 val patches and overlays the
predicted polygons (blue) + GT polygons (green) on the raw image, writing
PNGs to `<benchmark-root>/qualitative/<dataset>/<scenario>/<model>/`.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def _load_rings(seg: list[list[float]]) -> list[list[tuple[float, float]]]:
    rings = []
    for flat in seg:
        if not flat or len(flat) < 6 or len(flat) % 2:
            continue
        pts = [(float(flat[i]), float(flat[i + 1])) for i in range(0, len(flat), 2)]
        rings.append(pts)
    return rings


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True, help="The run output dir with predictions.json")
    p.add_argument("--gt-json", type=Path, required=True)
    p.add_argument("--images-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--num-patches", type=int, default=3)
    p.add_argument("--score-threshold", type=float, default=0.5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with (args.gt_json).open("r") as f:
        gt = json.load(f)
    id_to_image = {int(img["id"]): img for img in gt["images"]}
    gt_by_image = defaultdict(list)
    for ann in gt["annotations"]:
        gt_by_image[int(ann["image_id"])].append(ann)

    with (args.run_dir / "predictions.json").open("r") as f:
        preds = json.load(f)
    pred_by_image = defaultdict(list)
    for p in preds:
        if float(p.get("score", 0.0)) < args.score_threshold:
            continue
        pred_by_image[int(p["image_id"])].append(p)

    candidate_ids = sorted(
        set(gt_by_image.keys()) & set(pred_by_image.keys()),
        key=lambda iid: -len(gt_by_image[iid]),
    )[: args.num_patches]

    for iid in candidate_ids:
        meta = id_to_image.get(iid)
        if meta is None:
            continue
        img_path = args.images_dir / meta["file_name"]
        if not img_path.exists():
            continue
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        for ann in gt_by_image[iid]:
            for ring in _load_rings(ann.get("segmentation", [])):
                draw.line(ring + [ring[0]], fill=(0, 220, 0), width=2)
        for pred in pred_by_image[iid]:
            for ring in _load_rings(pred.get("segmentation", [])):
                draw.line(ring + [ring[0]], fill=(30, 90, 255), width=2)
        out_path = args.output_dir / f"overlay_image_id_{iid}.png"
        img.save(out_path)
        print(f"[qualitative] wrote {out_path}")


if __name__ == "__main__":
    main()
