#!/usr/bin/env python3
"""Convert a GCP/MMDetection ``*.segm.json`` dump into hisup-format predictions.

GCP's CocoMetric (``mask_type='polygon'``, ``format_only=True``) writes one record per
instance with ``polygon = [exterior, hole1, ...]`` (flat x,y lists, patch pixel coords)
next to the RLE ``segmentation``. The polygon is what the unified evaluator scores
(``--pred-type hisup``); image ids come straight from the COCO annotation file that was
used as the test set, so they are checked against the canonical GT here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="GCP *.segm.json")
    parser.add_argument("--gt", type=Path, required=True, help="GT annotation.json used for the id check.")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def rings(candidate) -> list[list[float]]:
    if not isinstance(candidate, list):
        return []
    out = []
    for ring in candidate:
        if isinstance(ring, list) and len(ring) >= 6 and len(ring) % 2 == 0:
            out.append([float(v) for v in ring])
    return out


def main() -> None:
    args = parse_args()
    raw = json.loads(args.input.read_text(encoding="utf-8"))
    gt = json.loads(args.gt.read_text(encoding="utf-8"))
    gt_ids = {int(img["id"]) for img in gt["images"]}

    records = []
    missing_polygon = invalid_polygon = 0
    for item in raw:
        if "polygon" not in item:
            missing_polygon += 1
            continue
        seg = rings(item["polygon"])
        if not seg:
            invalid_polygon += 1
            continue
        xs = [v for ring in seg for v in ring[0::2]]
        ys = [v for ring in seg for v in ring[1::2]]
        records.append({
            "image_id": int(item["image_id"]),
            "category_id": int(item.get("category_id", 1)),
            "score": float(item.get("score", 0.0)),
            "segmentation": seg,
            "bbox": [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)],
        })

    pred_ids = {r["image_id"] for r in records}
    outside = sorted(pred_ids - gt_ids)
    if outside:
        raise RuntimeError(f"{len(outside)} predicted image ids are not in {args.gt} (first: {outside[:5]})")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records), encoding="utf-8")
    print(json.dumps({
        "input": str(args.input),
        "output": str(args.output),
        "num_input": len(raw),
        "num_output": len(records),
        "skipped_missing_polygon": missing_polygon,
        "skipped_invalid_polygon": invalid_polygon,
        "images_with_predictions": len(pred_ids),
        "gt_images": len(gt_ids),
        "holes": sum(len(r["segmentation"]) - 1 for r in records),
    }, indent=2))


if __name__ == "__main__":
    main()
