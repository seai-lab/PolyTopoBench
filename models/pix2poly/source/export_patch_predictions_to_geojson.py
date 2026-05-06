import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert Pix2Poly COCO-style patch predictions to patch-pixel GeoJSON."
    )
    parser.add_argument(
        "--pred-json",
        required=True,
        help="Path to Pix2Poly prediction JSON.",
    )
    parser.add_argument(
        "--output-geojson",
        default="",
        help="Output GeoJSON path. Defaults to <pred-json stem>.geojson",
    )
    parser.add_argument(
        "--annotation-json",
        default="",
        help="Optional COCO annotation.json to enrich features with patch metadata.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=None,
        help="Optional score filter.",
    )
    return parser.parse_args()


def load_image_lookup(annotation_json_path: str) -> dict[int, dict]:
    if not annotation_json_path:
        return {}
    with open(annotation_json_path, "r", encoding="utf-8") as handle:
        coco = json.load(handle)
    return {int(image["id"]): image for image in coco.get("images", [])}


def flatten_to_ring(segmentation_part: list[float]) -> list[list[float]]:
    if len(segmentation_part) < 6 or len(segmentation_part) % 2 != 0:
        raise ValueError("Invalid segmentation polygon: expected an even-length list with >= 3 points.")

    ring = []
    for idx in range(0, len(segmentation_part), 2):
        ring.append([float(segmentation_part[idx]), float(segmentation_part[idx + 1])])

    if ring[0] != ring[-1]:
        ring.append(ring[0].copy())
    return ring


def annotation_to_feature(annotation: dict, image_lookup: dict[int, dict]) -> dict:
    image_id = int(annotation["image_id"])
    image_info = image_lookup.get(image_id, {})
    segmentation = annotation.get("segmentation", [])
    rings = [flatten_to_ring(part) for part in segmentation if part]

    if not rings:
        raise ValueError(f"Prediction for image_id={image_id} has no valid segmentation.")

    if len(rings) == 1:
        geometry = {
            "type": "Polygon",
            "coordinates": [rings[0]],
        }
    else:
        geometry = {
            "type": "MultiPolygon",
            "coordinates": [[[ring_pt for ring_pt in ring]] for ring in rings],
        }

    properties = {
        "image_id": image_id,
        "category_id": annotation.get("category_id"),
        "score": annotation.get("score"),
        "bbox_xywh": annotation.get("bbox"),
        "segment_count": len(rings),
        "coordinate_space": "patch_pixel",
        "pixel_origin": "top_left",
        "axis_order": "x_y",
    }
    if image_info:
        properties["patch_file_name"] = image_info.get("file_name")
        properties["patch_width"] = image_info.get("width")
        properties["patch_height"] = image_info.get("height")

    return {
        "type": "Feature",
        "geometry": geometry,
        "properties": properties,
    }


def safe_annotation_to_feature(annotation: dict, image_lookup: dict[int, dict]) -> dict | None:
    try:
        return annotation_to_feature(annotation, image_lookup)
    except ValueError:
        return None


def main():
    args = parse_args()

    pred_json_path = Path(args.pred_json)
    output_geojson_path = Path(args.output_geojson) if args.output_geojson else pred_json_path.with_suffix(".geojson")
    image_lookup = load_image_lookup(args.annotation_json)

    with pred_json_path.open("r", encoding="utf-8") as handle:
        predictions = json.load(handle)

    features = []
    skipped_invalid = 0
    for annotation in predictions:
        if args.score_threshold is not None:
            score = annotation.get("score")
            if score is None or float(score) < args.score_threshold:
                continue
        feature = safe_annotation_to_feature(annotation, image_lookup)
        if feature is None:
            skipped_invalid += 1
            continue
        features.append(feature)

    geojson = {
        "type": "FeatureCollection",
        "meta": {
            "coordinate_space": "patch_pixel",
            "units": "pixels",
            "pixel_origin": "top_left",
            "axis_order": "x_y",
            "source_prediction_json": str(pred_json_path),
            "source_annotation_json": str(args.annotation_json) if args.annotation_json else None,
            "feature_count": len(features),
            "skipped_invalid_predictions": skipped_invalid,
        },
        "features": features,
    }

    output_geojson_path.parent.mkdir(parents=True, exist_ok=True)
    output_geojson_path.write_text(
        json.dumps(geojson, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved GeoJSON to: {output_geojson_path}")
    print(f"Feature count: {len(features)}")
    print(f"Skipped invalid predictions: {skipped_invalid}")


if __name__ == "__main__":
    main()
