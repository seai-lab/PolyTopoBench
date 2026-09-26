#!/usr/bin/env python3
"""Convert FFL val-fold outputs into hisup-format predictions on the canonical 512 patches.

FFL (Inria) evaluates on its own 725x725 data patches (stride 512 over the 5000x5000
tiles), centre-cropped to a 512x512 model input. Its per-patch polygons are written as

  <results>/<fold>/<tile>.rowmin_R_colmin_C_rowmax_.._colmax_...poly_geojson.<key>.geojson
  <results>/<fold>.annotation.poly.<key>.json      (COCO list, image_id = crc32(patch stem))

The COCO list drops interior rings (upstream ``save_utils.poly_coco`` only keeps the
exterior), so holes are read from the geojson files and scores from the COCO list (same
polygon order per patch).

Each polygon is shifted to tile coordinates by (colmin + off, rowmin + off) with
off = (data_patch_size - input_patch_size) // 2 and clipped to every canonical patch
(``source_image_name`` + ``patch_origin`` of the GT) it overlaps.

Output records are ``{"image_id", "category_id", "score", "segmentation": [exterior,
hole1, ...], "bbox", "area"}`` for ``--pred-type hisup``.

Known limitations of FFL's native val protocol (reported in the summary, not corrected):
the centre crops start 106 px after the tile origin and stop 107 px before the far edge,
so a border strip of every Inria tile never receives predictions
(``uncovered_patch_area_fraction``), and a building crossing the seam between two crops
stays split into two polygons.

For Deventer each FFL "tile" is one canonical 512 patch renamed ``<split>_<file_name>``
(data patch == input patch == 512, offset 0); ``--split-csv`` maps ``image_name`` back
to the canonical ``source_image_name`` so the same code reduces to an id mapping.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import zlib
from collections import defaultdict
from pathlib import Path

from shapely.affinity import translate
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box, shape
from shapely.ops import unary_union
from shapely.validation import make_valid


PATCH_RE = re.compile(
    r"^(?P<tile>.+?)\.rowmin_(?P<rowmin>\d+)_colmin_(?P<colmin>\d+)_rowmax_(?P<rowmax>\d+)_colmax_(?P<colmax>\d+)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, required=True, help="FFL eval dir that contains <fold>/ and <fold>.annotation.poly.<key>.json, or its parent (see --run-name).")
    parser.add_argument("--run-name", default=None, help="If --results-dir is the parent, pick the newest '<run-name> | <timestamp>' subdir.")
    parser.add_argument("--split-csv", type=Path, default=None, help="FFL split CSV with image_name/source_image_name columns (Deventer).")
    parser.add_argument("--require-all-tiles", action="store_true", help="Fail unless every GT tile received FFL output (full runs).")
    parser.add_argument("--gt", type=Path, required=True, help="Canonical hisup val annotation.json.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold", default="val")
    parser.add_argument("--method-key", default="acm", help="Polygonization key in the FFL output names (default: acm).")
    parser.add_argument("--data-patch-size", type=int, default=725)
    parser.add_argument("--input-patch-size", type=int, default=512)
    parser.add_argument("--min-area", type=float, default=4.0, help="Drop clipped parts smaller than this (pixels).")
    parser.add_argument("--subset-gt-output", type=Path, default=None, help="Write the GT restricted to the tiles FFL predicted (for smoke runs).")
    parser.add_argument("--summary", type=Path, default=None)
    return parser.parse_args()


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def polygonal_parts(geom):
    if geom is None or geom.is_empty:
        return
    if isinstance(geom, Polygon):
        yield geom
    elif isinstance(geom, (MultiPolygon, GeometryCollection)):
        for part in geom.geoms:
            yield from polygonal_parts(part)


def clean(geom):
    if geom.is_empty:
        return []
    if not geom.is_valid:
        geom = make_valid(geom)
    return [part for part in polygonal_parts(geom) if part.area > 0]


def ring_flat(coords) -> list[float]:
    pts = list(coords)
    if pts and pts[0] != pts[-1]:
        pts.append(pts[0])
    return [round(float(v), 3) for xy in pts for v in xy[:2]]


def load_scores(results_dir: Path, fold: str, key: str) -> dict[int, list[dict]]:
    coco_path = results_dir / f"{fold}.annotation.poly.{key}.json"
    if not coco_path.is_file():
        raise FileNotFoundError(f"Missing FFL COCO output with scores: {coco_path}")
    by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in read_json(coco_path):
        by_image[int(ann["image_id"])].append(ann)
    return by_image


def exterior_matches(poly: Polygon, ann: dict) -> bool:
    seg = ann.get("segmentation") or [[]]
    flat = seg[0]
    coords = list(poly.exterior.coords)
    if len(flat) != 2 * len(coords):
        return False
    return all(abs(flat[2 * i] - x) < 0.02 and abs(flat[2 * i + 1] - y) < 0.02 for i, (x, y) in enumerate(coords[:4]))


def resolve_results_dir(results_dir: Path, fold: str, run_name: str | None) -> Path:
    if (results_dir / fold).is_dir():
        return results_dir
    if run_name is None:
        raise FileNotFoundError(f"{results_dir / fold} does not exist and --run-name was not given")
    candidates = sorted(p for p in results_dir.glob(f"{run_name} | *") if (p / fold).is_dir())
    if not candidates:
        raise FileNotFoundError(f"No '{run_name} | <timestamp>/{fold}' under {results_dir}")
    return candidates[-1]


def read_split_csv(split_csv: Path | None) -> tuple[dict[str, str], dict[str, int]]:
    """image_name stem -> canonical source stem, and -> FFL sample image_id (Deventer)."""
    aliases: dict[str, str] = {}
    sample_ids: dict[str, int] = {}
    if split_csv is None:
        return aliases, sample_ids
    with split_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            stem = Path(row["image_name"]).stem
            if row.get("source_image_name"):
                aliases[stem] = Path(row["source_image_name"]).stem
            if (row.get("image_id") or "").strip():
                sample_ids[stem] = int(row["image_id"])
    return aliases, sample_ids


def main() -> None:
    args = parse_args()
    args.results_dir = resolve_results_dir(args.results_dir, args.fold, args.run_name)
    print(f"[ffl->hisup] results dir: {args.results_dir}")
    aliases, sample_ids = read_split_csv(args.split_csv)
    fold_dir = args.results_dir / args.fold
    suffix = f".poly_geojson.{args.method_key}.geojson"
    geojson_paths = sorted(p for p in fold_dir.glob(f"*{suffix}"))
    if not geojson_paths:
        raise FileNotFoundError(f"No FFL per-patch geojson '*{suffix}' under {fold_dir}")
    scores_by_image = load_scores(args.results_dir, args.fold, args.method_key)

    gt = read_json(args.gt)
    patches_by_tile: dict[str, list[tuple[int, int, int, int, int]]] = defaultdict(list)
    for img in gt["images"]:
        # Inria: patches of a 5000 px tile; Deventer: every image is its own "tile".
        tile = Path(img.get("source_image_name") or img["file_name"]).stem
        x0, y0 = (int(v) for v in img.get("patch_origin", (0, 0)))
        patches_by_tile[tile].append((int(img["id"]), x0, y0, x0 + int(img["width"]), y0 + int(img["height"])))
    category_id = int(gt["categories"][0]["id"]) if gt.get("categories") else 100

    offset = (args.data_patch_size - args.input_patch_size) // 2
    stats = defaultdict(int)
    tile_polys: dict[str, list[tuple[Polygon, float]]] = defaultdict(list)
    tile_windows: dict[str, list] = defaultdict(list)
    for path in geojson_paths:
        stem = path.name[: -len(suffix)]
        m = PATCH_RE.match(stem)
        if m is None:
            stats["unparsed_patch_names"] += 1
            continue
        ffl_tile = m.group("tile")
        tile = aliases.get(ffl_tile, ffl_tile)
        rowmin, colmin, rowmax, colmax = (int(m.group(k)) for k in ("rowmin", "colmin", "rowmax", "colmax"))
        if rowmax - rowmin != args.data_patch_size or colmax - colmin != args.data_patch_size:
            raise ValueError(f"{path.name}: patch size {rowmax - rowmin} != --data-patch-size {args.data_patch_size}")
        if tile not in patches_by_tile:
            stats["patches_without_gt_tile"] += 1
            continue
        stats["ffl_patches"] += 1
        dx, dy = colmin + offset, rowmin + offset
        window = box(dx, dy, dx + args.input_patch_size, dy + args.input_patch_size)
        tile_windows[tile].append(window)
        geoms = [shape(g) for g in read_json(path).get("geometries", [])]
        # FFL's COCO image_id is the sample's image_id when the dataset provides one
        # (Deventer: split-CSV image_id), else crc32(patch name) (Inria).
        coco_id = sample_ids.get(ffl_tile, zlib.crc32(stem.encode("utf-8")) & 0xFFFFFFFF)
        anns = scores_by_image.get(coco_id, [])
        same_order = len(anns) == len(geoms)
        for idx, geom in enumerate(geoms):
            if same_order and isinstance(geom, Polygon) and exterior_matches(geom, anns[idx]):
                score = float(anns[idx]["score"])
            else:
                score = float(anns[idx]["score"]) if same_order else 1.0
                stats["score_fallbacks"] += 1
            for poly in clean(geom):
                stats["ffl_polygons"] += 1
                stats["ffl_holes"] += len(poly.interiors)
                glob_poly = translate(poly, xoff=dx, yoff=dy).intersection(window)
                for part in clean(glob_poly):
                    tile_polys[tile].append((part, score))

    records = []
    covered_ids: set[int] = set()
    for tile, items in tile_polys.items():
        for poly, score in items:
            minx, miny, maxx, maxy = poly.bounds
            for image_id, x0, y0, x1, y1 in patches_by_tile[tile]:
                if maxx <= x0 or minx >= x1 or maxy <= y0 or miny >= y1:
                    continue
                for part in clean(poly.intersection(box(x0, y0, x1, y1))):
                    if part.area < args.min_area:
                        stats["dropped_small_parts"] += 1
                        continue
                    local = translate(part, xoff=-x0, yoff=-y0)
                    seg = [ring_flat(local.exterior.coords)] + [ring_flat(r.coords) for r in local.interiors]
                    bx0, by0, bx1, by1 = local.bounds
                    records.append({
                        "image_id": image_id,
                        "category_id": category_id,
                        "score": float(score),
                        "segmentation": seg,
                        "bbox": [bx0, by0, bx1 - bx0, by1 - by0],
                        "area": float(local.area),
                    })
                    covered_ids.add(image_id)
                    stats["output_holes"] += len(seg) - 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records), encoding="utf-8")

    predicted_tiles = sorted(tile_windows)
    gt_ids = {img["id"] for img in gt["images"]}
    tile_ids = {pid for t in predicted_tiles for pid, *_ in patches_by_tile[t]}
    # Share of canonical patch area that no FFL model-input window covers (tile borders).
    uncovered = 0.0
    for t in predicted_tiles:
        cover = unary_union(tile_windows[t])
        for _, x0, y0, x1, y1 in patches_by_tile[t]:
            cell = box(x0, y0, x1, y1)
            uncovered += cell.difference(cover).area
    summary = {
        **stats,
        "output_predictions": len(records),
        "predicted_tiles": len(predicted_tiles),
        "gt_tiles": len(patches_by_tile),
        "canonical_images_in_predicted_tiles": len(tile_ids),
        "canonical_images_with_predictions": len(covered_ids),
        "ids_outside_gt": len(covered_ids - gt_ids),
        "uncovered_patch_area_fraction": uncovered / max(1, sum((x1 - x0) * (y1 - y0) for t in predicted_tiles for _, x0, y0, x1, y1 in patches_by_tile[t])),
        "output": str(args.output),
    }
    if summary["ids_outside_gt"]:
        raise RuntimeError(f"{summary['ids_outside_gt']} predicted image ids are not in the GT")
    if stats["score_fallbacks"] > 0.01 * max(1, stats["ffl_polygons"]):
        raise RuntimeError(f"{stats['score_fallbacks']} of {stats['ffl_polygons']} FFL polygons could not be matched to their COCO score")
    if stats["patches_without_gt_tile"]:
        raise RuntimeError(f"{stats['patches_without_gt_tile']} FFL patches could not be matched to a GT tile (check --split-csv)")
    if args.require_all_tiles and len(predicted_tiles) != len(patches_by_tile):
        raise RuntimeError(f"FFL output covers {len(predicted_tiles)} of {len(patches_by_tile)} GT tiles")

    if args.subset_gt_output is not None:
        subset = dict(gt)
        subset["images"] = [img for img in gt["images"] if img["id"] in tile_ids]
        subset["annotations"] = [ann for ann in gt["annotations"] if ann["image_id"] in tile_ids]
        args.subset_gt_output.parent.mkdir(parents=True, exist_ok=True)
        args.subset_gt_output.write_text(json.dumps(subset), encoding="utf-8")
        summary["subset_gt"] = str(args.subset_gt_output)
        summary["subset_gt_images"] = len(subset["images"])

    text = json.dumps(summary, indent=2)
    print(text)
    if args.summary is not None:
        args.summary.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
