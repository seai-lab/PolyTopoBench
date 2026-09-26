#!/usr/bin/env python3
"""Polygonize ACPV-Net val outputs with the upstream PSLG reconstruction and write hisup predictions.

Inputs (all produced by ``scripts/test.py`` + ``latent_decoder_accelerated.py`` +
``tools/extract_vertices_from_heatmap.py``):

  --seg-dir     <stem>.npy  (H, W) int label map (0 = background, 1 = foreground class)
  --vertex-dir  <stem>.json predicted vertices
  --prob-dir    <stem>.npy  (C, H, W) class probabilities (score = mean prob inside the polygon)
  --gt-json     canonical hisup annotation.json; provides the file_name -> image_id mapping and
                the set of images that must have predictions.

Output: a JSON list of ``{"image_id", "category_id", "segmentation": [exterior, hole1, ...],
"bbox", "score", "area"}`` records for ``evaluate_vector_polygons.py --pred-type hisup``.
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Polygon

TOOLS_DIR = Path(__file__).resolve().parent / "source" / "tools"
sys.path.insert(0, str(TOOLS_DIR))

from polygonize_pslg_one_image import process_one_image_categories  # noqa: E402
from polygonize_utils import flatten_to_xylist  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seg-dir", type=Path, required=True)
    parser.add_argument("--vertex-dir", type=Path, required=True)
    parser.add_argument("--prob-dir", type=Path, default=None)
    parser.add_argument("--gt-json", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--target-cat", type=int, default=1, help="Label of the foreground class in the seg maps.")
    parser.add_argument("--dist-thresh", type=float, default=5.0)
    parser.add_argument("--corner-eps", type=float, default=2.0)
    parser.add_argument("--no-vss", action="store_true", help="Disable vertex-guided subset selection (pure DP).")
    parser.add_argument("--no-dp-fix", action="store_true")
    parser.add_argument("--dp-epsilon-dp", type=float, default=1.0)
    parser.add_argument("--dp-epsilon-fix", type=float, default=2.5)
    parser.add_argument("--num-workers", type=int, default=8)
    return parser.parse_args()


def instance_score(prob_map, cat_id: int, ext_xy, holes_xy, height: int, width: int) -> float:
    if prob_map is None or prob_map.ndim != 3 or not 0 <= cat_id < prob_map.shape[0]:
        return 1.0
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [np.asarray(ext_xy, dtype=np.int32).reshape(-1, 1, 2)], 1)
    for hole in holes_xy:
        if len(hole) >= 3:
            cv2.fillPoly(mask, [np.asarray(hole, dtype=np.int32).reshape(-1, 1, 2)], 0)
    inside = mask == 1
    return float(prob_map[cat_id][inside].mean()) if inside.any() else 0.0


def polygonize_one(job):
    args, stem, image_id, category_id = job
    seg_path = args.seg_dir / f"{stem}.npy"
    vertex_path = args.vertex_dir / f"{stem}.json"
    try:
        results = process_one_image_categories(
            seg_path=str(seg_path),
            junction_json=str(vertex_path),
            categories=[args.target_cat],
            dist_thresh=args.dist_thresh,
            corner_eps=args.corner_eps,
            vis_dir=None,
            vis_scale=1,
            vss=not args.no_vss,
            dp_fix=not args.no_dp_fix,
            dp_epsilon_dp=args.dp_epsilon_dp,
            dp_epsilon_fix=args.dp_epsilon_fix,
        )
    except Exception as exc:  # keep going; failures are counted and reported
        return stem, [], f"{type(exc).__name__}: {exc}"

    height, width = np.load(seg_path).shape[:2]
    prob_map = None
    if args.prob_dir is not None and (args.prob_dir / f"{stem}.npy").is_file():
        prob_map = np.load(args.prob_dir / f"{stem}.npy").astype(np.float32)

    records = []
    for inst in results.get(int(args.target_cat), []):
        if not inst:
            continue
        ext_xy = flatten_to_xylist(inst[0])
        if len(ext_xy) < 3:
            continue
        holes_xy = [ring for ring in (flatten_to_xylist(h) for h in inst[1:]) if len(ring) >= 3]
        geom = Polygon(ext_xy, holes=holes_xy)
        if not geom.is_valid:
            geom = geom.buffer(0)
        if geom.area <= 0.0:
            continue
        xs = [p[0] for p in ext_xy]
        ys = [p[1] for p in ext_xy]
        records.append({
            "image_id": int(image_id),
            "category_id": int(category_id),
            "segmentation": [[float(v) for xy in ring for v in xy] for ring in [ext_xy, *holes_xy]],
            "bbox": [float(min(xs)), float(min(ys)), float(max(xs) - min(xs)), float(max(ys) - min(ys))],
            "score": instance_score(prob_map, int(args.target_cat), ext_xy, holes_xy, height, width),
            "area": float(geom.area),
        })
    return stem, records, None


def main() -> None:
    args = parse_args()
    gt = json.loads(args.gt_json.read_text(encoding="utf-8"))
    category_id = int(gt["categories"][0]["id"]) if gt.get("categories") else 100
    stem_to_id = {Path(image["file_name"]).stem: int(image["id"]) for image in gt["images"]}

    missing = sorted(
        stem for stem in stem_to_id
        if not (args.seg_dir / f"{stem}.npy").is_file() or not (args.vertex_dir / f"{stem}.json").is_file()
    )
    if missing:
        raise SystemExit(
            f"{len(missing)}/{len(stem_to_id)} GT images have no seg/vertex output "
            f"(first: {missing[:3]}); inference and GT are out of sync."
        )
    extra = {p.stem for p in args.seg_dir.glob("*.npy")} - set(stem_to_id)
    if extra:
        print(f"[convert] ignoring {len(extra)} seg maps that are not in {args.gt_json}")

    jobs = [(args, stem, image_id, category_id) for stem, image_id in sorted(stem_to_id.items(), key=lambda kv: kv[1])]
    predictions, failures = [], []
    with Pool(max(1, args.num_workers)) as pool:
        for stem, records, error in pool.imap(polygonize_one, jobs, chunksize=4):
            predictions.extend(records)
            if error:
                failures.append((stem, error))

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(predictions) + "\n", encoding="utf-8")
    print(
        f"[convert] wrote {len(predictions)} polygons for {len(jobs)} images "
        f"({len({p['image_id'] for p in predictions})} with predictions) to {args.out_json}"
    )
    if failures:
        print(f"[convert] polygonization failed on {len(failures)} images, e.g. {failures[:3]}")


if __name__ == "__main__":
    main()
