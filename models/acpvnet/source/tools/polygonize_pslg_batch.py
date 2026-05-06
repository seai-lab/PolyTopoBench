#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch polygonize semantic label maps with the PSLG-based post-processing pipeline.

This script runs the multi-category polygonization workflow used by the
reported ACPV-Net baseline, saves per-image polygon JSON files, exports
per-category COCO-style predictions, and renders overview visualizations.

Example:
    python tools/polygonize_pslg_batch.py \
        --seg_dir /path/to/seg_npy \
        --junction_dir /path/to/vertex_json \
        --categories 0 1 2 3 4 \
        --cats_out_dir /path/to/coco_json \
        --vss --dp_fix
"""

import os
import cv2
import json
import argparse
import numpy as np
from polygonize_pslg_one_image import (
    process_one_image_categories, flatten_to_xylist
)
from shapely.geometry import Polygon
from tqdm import tqdm


ID2NAME = {
    0: "building",
    1: "road",
    2: "unvegetated",
    3: "vegetation",
    4: "water",
}
def _bbox_from_xy(ext_xy):
    xs = [p[0] for p in ext_xy]; ys = [p[1] for p in ext_xy]
    x0, y0 = float(min(xs)), float(min(ys))
    x1, y1 = float(max(xs)), float(max(ys))
    return [x0, y0, x1 - x0, y1 - y0]


def area_bbox_with_holes(ext_xy, holes_xy):
    """
    Compute polygon area and bounding box from one exterior ring and its holes.

    The bounding box is always measured from the exterior ring and returned as
    `[x_min, y_min, width, height]`.
    """
    # Use the exterior ring for the bounding box.
    bbox = _bbox_from_xy(ext_xy) if len(ext_xy) >= 3 else [0.0, 0.0, 0.0, 0.0]

    holes_valid = [h for h in holes_xy if len(h) >= 3]
    poly = Polygon(ext_xy, holes=holes_valid)
    if not poly.is_valid:
        # Standard fix for self-intersections, duplicate points, or tiny gaps.
        poly = poly.buffer(0)
    return float(max(poly.area, 0.0)), [float(b) for b in bbox]


def str2bool(v):
    if isinstance(v, bool):
        return v
    return v.lower() in ("1","true","t","yes","y")


def _flatten_xy(cnt):
        """Convert an `Nx1x2` or `Nx2` contour to `[x1, y1, x2, y2, ...]`."""
        c = np.squeeze(cnt, axis=1) if cnt.ndim == 3 else cnt
        if c.ndim != 2 or c.shape[1] != 2:
            return []
        return [float(v) for xy in c for v in xy]

def _approx(cnt, eps):
    """Apply DP simplification to a contour while keeping at least three points."""
    if cnt is None or len(cnt) == 0:
        return None
    approx = cv2.approxPolyDP(cnt, epsilon=eps, closed=True)
    # Reject degenerate contours caused by repeated points.
    if approx is None or len(approx) < 3:
        return None
    return approx


def instance_score_mean_prob(prob_map, cat_id, ext_xy, holes_xy, H, W):
    """
    prob_map: (C, H, W) float32 in [0,1]
    cat_id: int
    ext_xy: [(x,y),...]
    holes_xy: list of [(x,y),...]
    """
    if prob_map is None:
        return 1.0
    if prob_map.ndim != 3 or cat_id < 0 or cat_id >= prob_map.shape[0]:
        return 1.0

    p = prob_map[cat_id]  # (H, W)

    # exterior mask
    mask_ext = np.zeros((H, W), dtype=np.uint8)
    poly_ext = np.array(ext_xy, dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask_ext, [poly_ext], 1)

    # holes mask
    if holes_xy:
        mask_hole = np.zeros((H, W), dtype=np.uint8)
        for h in holes_xy:
            if len(h) < 3:
                continue
            poly_h = np.array(h, dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(mask_hole, [poly_h], 1)
        mask = (mask_ext == 1) & (mask_hole == 0)
    else:
        mask = (mask_ext == 1)

    if not np.any(mask):
        return 0.0

    return float(np.mean(p[mask]))


def main():
    parser = argparse.ArgumentParser(
        description="Batch polygonization with the reported ACPV-Net PSLG+VSS pipeline."
    )

    # ------------------------------------------------------------------
    # Required I/O
    # ------------------------------------------------------------------
    parser.add_argument("--seg_dir", required=True,
                        help="Directory of predicted label maps (.npy).")
    parser.add_argument("--junction_dir", required=True,
                        help="Directory of predicted vertices JSONs: [[x,y], ...] (matched by basename).")
    parser.add_argument("--categories", type=int, nargs="+", required=True,
                        help="Target category IDs, e.g. --categories 0 1 2")
    parser.add_argument("--cats_out_dir", required=True,
                        help="Directory to save per-category COCO prediction JSONs.")
    parser.add_argument("--prob_dir", default=None,
                    help="Directory of predicted probability maps (.npy), shape (C,H,W). Matched by basename.")


    # ------------------------------------------------------------------
    # Optional visualization
    # ------------------------------------------------------------------
    parser.add_argument("--vis_dir", default=None,
                        help="Optional root dir for per-image step visualizations: <vis_dir>/<name>/...")
    parser.add_argument("--vis_scale", type=int, default=1,
                        help="Visualization scale factor (default: 1).")

    # ------------------------------------------------------------------
    # Core geometric hyper-parameters (ours-mode)
    # ------------------------------------------------------------------
    parser.add_argument("--dist_thresh", type=float, default=5.0,
                        help="Snap radius for predicted vertices to boundary polylines (pixels).")
    parser.add_argument("--corner_eps", type=float, default=2.0,
                        help="Radius to treat a point as too close to a structural corner (pixels).")

    # ------------------------------------------------------------------
    # Point selection strategy (ours-mode)
    # ------------------------------------------------------------------
    parser.add_argument("--vss", action="store_true",
                        help="Enable vertex-guided subset selection (VSS). "
                             "If not set, pure DP simplification is used.")
    parser.add_argument("--dp_fix", action="store_true",
                        help="Enable DP-based safeguard (effective only when --vss is set).")
    parser.add_argument("--dp_epsilon_dp", type=float, default=1.0,
                        help="RDP epsilon for pure DP fallback (used when vss is OFF in ours-mode)."
    )
    parser.add_argument("--dp_epsilon_fix", type=float, default=2.5,
                        help="RDP epsilon for DP safeguard in VSS mode (used when dp_fix is ON).")

    args = parser.parse_args()

    # Prepare output dirs
    os.makedirs(args.cats_out_dir, exist_ok=True)
    if args.vis_dir:
        os.makedirs(args.vis_dir, exist_ok=True)

    # COCO predictions per category
    per_cat_preds = {int(c): [] for c in args.categories}

    # ------------------------------------------------------------------
    # Iterate inputs
    # ------------------------------------------------------------------
    seg_files = sorted([f for f in os.listdir(args.seg_dir) if f.endswith('.npy')])

    for f in tqdm(seg_files, desc="Polygonizing"):
        name = os.path.splitext(f)[0]
        seg_path  = os.path.join(args.seg_dir, f)
        junc_path = os.path.join(args.junction_dir, name + '.json')

        prob_map = None
        if args.prob_dir is not None:
            prob_path = os.path.join(args.prob_dir, name + ".npy")
            if os.path.exists(prob_path):
                prob_map = np.load(prob_path).astype(np.float32)

        if not os.path.exists(junc_path):
            continue

        vis_subdir = os.path.join(args.vis_dir, name) if args.vis_dir else None

        # --------------------------------------------------------------
        # Run polygonization
        # --------------------------------------------------------------
        results = process_one_image_categories(
            seg_path=seg_path,
            junction_json=junc_path,
            categories=args.categories,
            dist_thresh=args.dist_thresh,
            corner_eps=args.corner_eps,
            vis_dir=vis_subdir,
            vis_scale=args.vis_scale,
            vss=args.vss,
            dp_fix=args.dp_fix,
            dp_epsilon_dp=args.dp_epsilon_dp,
            dp_epsilon_fix=args.dp_epsilon_fix,
        )

        # --------------------------------------------------------------
        # COCO predictions (per-category)
        # --------------------------------------------------------------
        L = np.load(seg_path).astype(np.int32)
        H, W = L.shape
        image_id = int(name.split('_')[-1])

        for cat in args.categories:
            cat = int(cat)
            inst_list = results.get(cat, [])
            for inst in inst_list:
                if not inst:
                    continue
                ext_xy = flatten_to_xylist(inst[0])
                holes_xy = [flatten_to_xylist(hv) for hv in inst[1:]]
                area, bbox = area_bbox_with_holes(ext_xy, holes_xy)

                score = instance_score_mean_prob(
                    prob_map=prob_map,
                    cat_id=cat,
                    ext_xy=ext_xy,
                    holes_xy=holes_xy,
                    H=H,
                    W=W
                )

                coco_obj = {
                    "image_id": image_id,
                    "category_id": 100,
                    "segmentation": inst,  # [[x1,y1,...], [hole...], ...]
                    "score": score,
                    "iscrowd": 0,
                    "area": area,
                    "bbox": bbox,
                }
                per_cat_preds[cat].append(coco_obj)

    # ------------------------------------------------------------------
    # Save COCO prediction json per category
    # ------------------------------------------------------------------
    for cat in args.categories:
        cat = int(cat)
        cat_name = ID2NAME.get(cat, f"cat_{cat}")
        save_path = os.path.join(args.cats_out_dir, f"{cat_name}.json")
        with open(save_path, 'w') as fp:
            json.dump(per_cat_preds[cat], fp)

if __name__ == "__main__":
    main()

