#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, mapping
from shapely.strtree import STRtree

try:
    from shapely.validation import make_valid
except ImportError:  # pragma: no cover
    make_valid = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert patch-level RoIPoly predictions into source-image GeoJSON and "
            "reconstruct hole rings with containment-based post-processing."
        )
    )
    parser.add_argument("--predictions", type=Path, required=True, help="RoIPoly predictions_*.json file.")
    parser.add_argument(
        "--annotation-raw",
        type=Path,
        required=True,
        help="Patch-level annotation_raw.json with source_image_name and patch_origin metadata.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for GeoJSON exports.")
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.5,
        help="Drop predictions below this confidence before post-processing.",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=10.0,
        help="Drop polygons smaller than this source-image pixel area.",
    )
    parser.add_argument(
        "--nms-iou-threshold",
        type=float,
        default=0.6,
        help="Suppress lower-scoring duplicate polygons when IoU exceeds this threshold.",
    )
    parser.add_argument(
        "--nms-overlap-threshold",
        type=float,
        default=0.85,
        help="Suppress lower-scoring duplicate polygons when overlap / min(area) exceeds this threshold.",
    )
    parser.add_argument(
        "--containment-threshold",
        type=float,
        default=0.98,
        help="Minimum fraction of a child polygon that must be covered by a parent to infer nesting.",
    )
    parser.add_argument(
        "--write-debug-rings",
        action="store_true",
        help="Also write per-ring GeoJSON with inferred exterior/hole roles before ring merging.",
    )
    return parser.parse_args()


def iter_polygonal_geometries(geometry):
    if geometry.is_empty:
        return
    if isinstance(geometry, Polygon):
        yield geometry
        return
    if isinstance(geometry, MultiPolygon):
        for part in geometry.geoms:
            if not part.is_empty:
                yield part
        return
    if isinstance(geometry, GeometryCollection):
        for part in geometry.geoms:
            yield from iter_polygonal_geometries(part)


def sanitize_polygon(polygon: Polygon, min_area: float):
    geometry = polygon
    if geometry.is_empty:
        return None
    if not geometry.is_valid:
        if make_valid is not None:
            geometry = make_valid(geometry)
        else:  # pragma: no cover
            geometry = geometry.buffer(0)
    parts = [part for part in iter_polygonal_geometries(geometry)]
    if not parts:
        return None
    polygon = max(parts, key=lambda item: item.area)
    if polygon.area < min_area or len(polygon.exterior.coords) < 4:
        return None
    return polygon


def polygon_from_flat_coords(flat_coords, origin_x: float, origin_y: float, min_area: float):
    if not flat_coords or len(flat_coords) < 6 or len(flat_coords) % 2 != 0:
        return None
    coords = [(float(flat_coords[i]) + origin_x, float(flat_coords[i + 1]) + origin_y) for i in range(0, len(flat_coords), 2)]
    polygon = Polygon(coords)
    return sanitize_polygon(polygon, min_area=min_area)


def build_index_lookup(geometries):
    lookup = defaultdict(list)
    for index, geometry in enumerate(geometries):
        lookup[geometry.wkb].append(index)
    return lookup


def query_tree_indices(tree: STRtree, geometry, wkb_lookup):
    result = tree.query(geometry)
    if len(result) == 0:
        return []
    first = result[0]
    if isinstance(first, (int, np.integer)):
        return [int(index) for index in result]
    indices = []
    for item in result:
        indices.extend(wkb_lookup.get(item.wkb, []))
    return indices


def polygon_overlap_scores(left: Polygon, right: Polygon):
    intersection = left.intersection(right).area
    if intersection <= 0:
        return 0.0, 0.0
    union = left.area + right.area - intersection
    iou = intersection / union if union > 0 else 0.0
    overlap_small = intersection / min(left.area, right.area)
    return iou, overlap_small


def deduplicate_predictions(items, iou_threshold: float, overlap_threshold: float):
    if len(items) <= 1:
        return items
    geometries = [item["geometry"] for item in items]
    tree = STRtree(geometries)
    wkb_lookup = build_index_lookup(geometries)
    order = sorted(
        range(len(items)),
        key=lambda index: (items[index]["score"], items[index]["geometry"].area),
        reverse=True,
    )
    rank = {index: position for position, index in enumerate(order)}
    suppressed = set()
    kept = []
    for index in order:
        if index in suppressed:
            continue
        kept.append(index)
        geometry = geometries[index]
        for candidate in query_tree_indices(tree, geometry, wkb_lookup):
            if candidate == index or candidate in suppressed or rank[candidate] < rank[index]:
                continue
            iou, overlap_small = polygon_overlap_scores(geometry, geometries[candidate])
            if iou >= iou_threshold or overlap_small >= overlap_threshold:
                suppressed.add(candidate)
    return [items[index] for index in kept]


def infer_parent_indices(items, containment_threshold: float):
    if not items:
        return []
    geometries = [item["geometry"] for item in items]
    tree = STRtree(geometries)
    wkb_lookup = build_index_lookup(geometries)
    areas = [geometry.area for geometry in geometries]
    parents = [None] * len(items)
    for index in sorted(range(len(items)), key=lambda idx: areas[idx]):
        geometry = geometries[index]
        anchor = geometry.representative_point()
        candidates = []
        for candidate in query_tree_indices(tree, geometry, wkb_lookup):
            if candidate == index or areas[candidate] <= areas[index]:
                continue
            parent = geometries[candidate]
            if not parent.covers(anchor):
                continue
            covered_ratio = parent.intersection(geometry).area / max(geometry.area, 1e-8)
            if covered_ratio < containment_threshold:
                continue
            candidates.append(candidate)
        if candidates:
            parents[index] = min(candidates, key=lambda idx: areas[idx])
    return parents


def compute_depths(parents):
    depths = [None] * len(parents)

    def resolve(index: int) -> int:
        if depths[index] is not None:
            return depths[index]
        parent = parents[index]
        if parent is None:
            depths[index] = 0
        else:
            depths[index] = resolve(parent) + 1
        return depths[index]

    for index in range(len(parents)):
        resolve(index)
    return depths


def to_feature(geometry, properties):
    return {"type": "Feature", "geometry": mapping(geometry), "properties": properties}


def export_source_image(
    items,
    source_image_name: str,
    output_dir: Path,
    write_debug_rings: bool,
    containment_threshold: float,
    nms_iou_threshold: float,
    nms_overlap_threshold: float,
):
    deduped = deduplicate_predictions(
        items,
        iou_threshold=nms_iou_threshold,
        overlap_threshold=nms_overlap_threshold,
    )
    parents = infer_parent_indices(deduped, containment_threshold=containment_threshold)
    depths = compute_depths(parents)
    children_by_parent = defaultdict(list)
    for child_index, parent_index in enumerate(parents):
        if parent_index is not None:
            children_by_parent[parent_index].append(child_index)

    feature_collection = {"type": "FeatureCollection", "features": []}
    debug_collection = {"type": "FeatureCollection", "features": []}
    inferred_holes = 0

    for index, item in enumerate(deduped):
        geometry = item["geometry"]
        depth = depths[index]
        role = "hole" if depth % 2 == 1 else "exterior"
        parent_index = parents[index]
        if write_debug_rings:
            debug_collection["features"].append(
                to_feature(
                    geometry,
                    {
                        "prediction_index": item["prediction_index"],
                        "score": item["score"],
                        "source_image_name": source_image_name,
                        "source_patch_name": item["source_patch_name"],
                        "source_patch_origin": item["source_patch_origin"],
                        "ring_depth": depth,
                        "ring_role": role,
                        "parent_index": parent_index,
                        "area": geometry.area,
                    },
                )
            )
        if role == "hole":
            inferred_holes += 1
            continue

        hole_indices = [child for child in children_by_parent[index] if depths[child] % 2 == 1]
        hole_indices.sort(key=lambda child: deduped[child]["geometry"].area, reverse=True)
        holes = [list(deduped[child]["geometry"].exterior.coords) for child in hole_indices]
        merged_geometry = Polygon(list(geometry.exterior.coords), holes)
        merged_geometry = sanitize_polygon(merged_geometry, min_area=0.0) or geometry
        feature_collection["features"].append(
            to_feature(
                merged_geometry,
                {
                    "prediction_index": item["prediction_index"],
                    "score": item["score"],
                    "source_image_name": source_image_name,
                    "source_patch_name": item["source_patch_name"],
                    "source_patch_origin": item["source_patch_origin"],
                    "num_holes": len(hole_indices),
                    "ring_depth": depth,
                    "area": merged_geometry.area,
                    "coordinate_space": "source_image_pixels",
                },
            )
        )

    image_stem = Path(source_image_name).stem
    image_geojson = output_dir / "by_image" / f"{image_stem}.geojson"
    image_geojson.parent.mkdir(parents=True, exist_ok=True)
    image_geojson.write_text(json.dumps(feature_collection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if write_debug_rings:
        debug_geojson = output_dir / "by_image_debug_rings" / f"{image_stem}.geojson"
        debug_geojson.parent.mkdir(parents=True, exist_ok=True)
        debug_geojson.write_text(json.dumps(debug_collection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return {
        "source_image_name": source_image_name,
        "num_input_predictions": len(items),
        "num_after_nms": len(deduped),
        "num_features": len(feature_collection["features"]),
        "num_inferred_holes": inferred_holes,
        "num_debug_rings": len(debug_collection["features"]),
    }, feature_collection["features"], debug_collection["features"]


def load_patch_image_lookup(annotation_raw_path: Path):
    with annotation_raw_path.open("r", encoding="utf-8") as handle:
        annotation_raw = json.load(handle)
    return {image["id"]: image for image in annotation_raw["images"]}


def collect_source_predictions(predictions_path: Path, patch_images, score_threshold: float, min_area: float):
    with predictions_path.open("r", encoding="utf-8") as handle:
        predictions = json.load(handle)

    grouped = defaultdict(list)
    skipped_invalid = 0
    skipped_low_score = 0
    for prediction_index, prediction in enumerate(predictions):
        score = float(prediction.get("score", 0.0))
        if score < score_threshold:
            skipped_low_score += 1
            continue
        image_info = patch_images.get(prediction["image_id"])
        if image_info is None:
            skipped_invalid += 1
            continue
        segmentation = prediction.get("segmentation", [])
        if not segmentation:
            skipped_invalid += 1
            continue
        polygon = polygon_from_flat_coords(
            segmentation[0],
            origin_x=float(image_info["patch_origin"][0]),
            origin_y=float(image_info["patch_origin"][1]),
            min_area=min_area,
        )
        if polygon is None:
            skipped_invalid += 1
            continue
        grouped[image_info["source_image_name"]].append(
            {
                "prediction_index": prediction_index,
                "geometry": polygon,
                "score": score,
                "source_patch_id": image_info["id"],
                "source_patch_name": image_info["file_name"],
                "source_patch_origin": image_info["patch_origin"],
            }
        )
    return grouped, {
        "num_raw_predictions": len(predictions),
        "num_after_score_threshold": sum(len(items) for items in grouped.values()),
        "num_skipped_low_score": skipped_low_score,
        "num_skipped_invalid": skipped_invalid,
    }


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    patch_images = load_patch_image_lookup(args.annotation_raw)
    grouped_predictions, ingest_stats = collect_source_predictions(
        args.predictions,
        patch_images,
        score_threshold=args.score_threshold,
        min_area=args.min_area,
    )

    combined = {"type": "FeatureCollection", "features": []}
    combined_debug = {"type": "FeatureCollection", "features": []}
    per_image_stats = []

    for source_image_name in sorted(grouped_predictions):
        image_stats, features, debug_features = export_source_image(
            grouped_predictions[source_image_name],
            source_image_name=source_image_name,
            output_dir=args.output_dir,
            write_debug_rings=args.write_debug_rings,
            containment_threshold=args.containment_threshold,
            nms_iou_threshold=args.nms_iou_threshold,
            nms_overlap_threshold=args.nms_overlap_threshold,
        )
        per_image_stats.append(image_stats)
        combined["features"].extend(features)
        if args.write_debug_rings:
            combined_debug["features"].extend(debug_features)

    combined_path = args.output_dir / f"{args.predictions.stem}.source_image.geojson"
    combined_path.write_text(json.dumps(combined, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if args.write_debug_rings:
        combined_debug_path = args.output_dir / f"{args.predictions.stem}.debug_rings.geojson"
        combined_debug_path.write_text(json.dumps(combined_debug, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary = {
        "predictions": str(args.predictions),
        "annotation_raw": str(args.annotation_raw),
        "output_dir": str(args.output_dir),
        "score_threshold": args.score_threshold,
        "min_area": args.min_area,
        "nms_iou_threshold": args.nms_iou_threshold,
        "nms_overlap_threshold": args.nms_overlap_threshold,
        "containment_threshold": args.containment_threshold,
        "num_source_images": len(grouped_predictions),
        "num_exported_features": len(combined["features"]),
        "num_exported_debug_rings": len(combined_debug["features"]),
        "per_image": per_image_stats,
    }
    summary.update(ingest_stats)
    summary_path = args.output_dir / f"{args.predictions.stem}.summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
