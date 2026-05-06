#!/usr/bin/env python3
"""Visualize patch-level RoIPoly predictions over patch images."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageColor, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True, help="RoIPoly predictions_*.json file.")
    parser.add_argument(
        "--annotation-json",
        type=Path,
        required=True,
        help="COCO annotation_roipoly.json used to map prediction image_id to patch filenames.",
    )
    parser.add_argument("--image-dir", type=Path, required=True, help="Directory containing patch images.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for rendered PNG overlays.")
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.0,
        help="Skip predictions below this score before rendering.",
    )
    parser.add_argument(
        "--image-stems",
        nargs="*",
        default=None,
        help="Optional subset of patch stems to render.",
    )
    parser.add_argument("--max-images", type=int, default=0, help="Render at most this many images. 0 means all.")
    parser.add_argument("--scale", type=float, default=1.0, help="Display scale applied to the patch image.")
    parser.add_argument("--polygon-color", default="#ffd400", help="Color for polygon fill and outline.")
    parser.add_argument("--polygon-fill-alpha", type=int, default=28, help="Polygon fill alpha in [0, 255].")
    parser.add_argument(
        "--use-gt-hole-role",
        action="store_true",
        help="Color predictions by GT hole/exterior semantics exported in predictions.json.",
    )
    parser.add_argument("--hole-color", default="#ff2a2a", help="Color used when gt_is_hole=true.")
    parser.add_argument("--hole-fill-alpha", type=int, default=48, help="Hole polygon fill alpha in [0, 255].")
    parser.add_argument(
        "--render-holes",
        action="store_true",
        help="Infer exterior/hole relations among predicted rings and render holes as cut-outs.",
    )
    parser.add_argument(
        "--nms-iou-threshold",
        type=float,
        default=0.6,
        help="Hole reconstruction NMS IoU threshold, aligned with export_roipoly_predictions_to_geojson.py.",
    )
    parser.add_argument(
        "--nms-overlap-threshold",
        type=float,
        default=0.85,
        help="Hole reconstruction overlap/min-area threshold, aligned with export_roipoly_predictions_to_geojson.py.",
    )
    parser.add_argument(
        "--containment-threshold",
        type=float,
        default=0.98,
        help="Minimum child-covered-by-parent ratio used to infer hole nesting.",
    )
    parser.add_argument("--line-width", type=int, default=2, help="Polygon outline width on the resized image.")
    parser.add_argument("--vertex-color", default="#1f6dff", help="Color for vertices.")
    parser.add_argument("--no-vertices", action="store_true", help="Disable vertex markers.")
    parser.add_argument("--vertex-radius", type=int, default=0, help="Vertex radius in resized-image pixels.")
    parser.add_argument("--draw-bboxes", action="store_true", help="Also draw each prediction bbox.")
    parser.add_argument("--bbox-color", default="#00d1ff", help="Color for bbox outlines.")
    parser.add_argument("--bbox-width", type=int, default=2, help="BBox outline width on the resized image.")
    parser.add_argument(
        "--bbox-expand-ratio",
        type=float,
        default=0.0,
        help="Expand drawn bbox by this ratio around its center. 0.2 means width/height become 1.2x.",
    )
    parser.add_argument("--draw-gt-bboxes", action="store_true", help="Draw GT bboxes from annotation_json.")
    parser.add_argument("--gt-bbox-color", default="#ff2a2a", help="Color for GT bbox outlines.")
    parser.add_argument("--gt-bbox-width", type=int, default=2, help="GT bbox outline width on the resized image.")
    parser.add_argument("--draw-scores", action="store_true", help="Draw score text near each polygon bbox.")
    parser.add_argument("--score-decimals", type=int, default=3, help="Decimal places for score text.")
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, max(1, (os.cpu_count() or 1) // 2)),
        help="Number of worker threads used for rendering.",
    )
    parser.add_argument(
        "--contact-sheet",
        action="store_true",
        help="Also write a simple contact sheet of rendered PNGs.",
    )
    parser.add_argument("--contact-columns", type=int, default=2, help="Column count for the contact sheet.")
    return parser.parse_args()


def draw_vertex(draw: ImageDraw.ImageDraw, x: float, y: float, radius: int, color: tuple[int, int, int, int]) -> None:
    if radius <= 0:
        draw.point((x, y), fill=color)
        return
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)


def load_annotation_json(annotation_json_path: Path) -> dict:
    with annotation_json_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_image_lookup(annotation_json: dict) -> dict[int, dict]:
    return {int(image["id"]): image for image in annotation_json["images"]}


def load_gt_bboxes(annotation_json: dict) -> dict[int, list[list[float]]]:
    grouped: dict[int, list[list[float]]] = defaultdict(list)
    for annotation in annotation_json.get("annotations", []):
        bbox = annotation.get("bbox") or []
        if len(bbox) != 4:
            continue
        grouped[int(annotation["image_id"])].append([float(value) for value in bbox])
    return grouped


def load_predictions(predictions_path: Path, image_lookup: dict[int, dict], score_threshold: float) -> tuple[dict[int, list[dict]], dict]:
    with predictions_path.open("r", encoding="utf-8") as handle:
        predictions = json.load(handle)

    grouped: dict[int, list[dict]] = defaultdict(list)
    skipped_missing_image = 0
    skipped_low_score = 0
    skipped_invalid_polygon = 0

    for prediction_index, prediction in enumerate(predictions):
        image_id = int(prediction["image_id"])
        image_info = image_lookup.get(image_id)
        if image_info is None:
            skipped_missing_image += 1
            continue

        score = float(prediction.get("score", 0.0))
        if score < score_threshold:
            skipped_low_score += 1
            continue

        segmentation = prediction.get("segmentation") or []
        if not segmentation or len(segmentation[0]) < 6 or len(segmentation[0]) % 2 != 0:
            skipped_invalid_polygon += 1
            continue

        flat_coords = segmentation[0]
        points = [
            (float(flat_coords[index]), float(flat_coords[index + 1]))
            for index in range(0, len(flat_coords), 2)
        ]
        if len(points) < 3:
            skipped_invalid_polygon += 1
            continue

        bbox = prediction.get("bbox") or [0.0, 0.0, 0.0, 0.0]
        grouped[image_id].append(
            {
                "prediction_index": prediction_index,
                "score": score,
                "points": points,
                "bbox": [float(value) for value in bbox],
                "gt_is_hole": bool(prediction.get("gt_is_hole", False)),
                "gt_ring_role": prediction.get("gt_ring_role"),
            }
        )

    stats = {
        "num_raw_predictions": len(predictions),
        "num_images_with_predictions": len(grouped),
        "num_rendered_predictions": sum(len(items) for items in grouped.values()),
        "num_skipped_missing_image": skipped_missing_image,
        "num_skipped_low_score": skipped_low_score,
        "num_skipped_invalid_polygon": skipped_invalid_polygon,
    }
    return grouped, stats


def build_tasks(args: argparse.Namespace):
    if not args.predictions.is_file():
        raise FileNotFoundError(f"Predictions file not found: {args.predictions}")
    if not args.annotation_json.is_file():
        raise FileNotFoundError(f"Annotation JSON not found: {args.annotation_json}")
    if not args.image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {args.image_dir}")

    annotation_json = load_annotation_json(args.annotation_json)
    image_lookup = load_image_lookup(annotation_json)
    gt_bboxes_by_image = load_gt_bboxes(annotation_json)
    grouped_predictions, load_stats = load_predictions(args.predictions, image_lookup, args.score_threshold)

    allowed_stems = set(args.image_stems) if args.image_stems else None
    image_items = []
    missing_images = 0
    for image_id in sorted(grouped_predictions, key=lambda current_id: image_lookup[current_id]["file_name"]):
        image_info = image_lookup[image_id]
        image_path = args.image_dir / image_info["file_name"]
        if allowed_stems is not None and Path(image_info["file_name"]).stem not in allowed_stems:
            continue
        if not image_path.is_file():
            missing_images += 1
            continue
        image_items.append((image_info, image_path, grouped_predictions[image_id]))

    if args.max_images > 0:
        image_items = image_items[: args.max_images]

    polygon_rgb = ImageColor.getrgb(args.polygon_color)
    hole_rgb = ImageColor.getrgb(args.hole_color)
    vertex_rgb = ImageColor.getrgb(args.vertex_color)
    bbox_rgb = ImageColor.getrgb(args.bbox_color)
    gt_bbox_rgb = ImageColor.getrgb(args.gt_bbox_color)

    polygon_fill_rgba = (*polygon_rgb, max(0, min(args.polygon_fill_alpha, 255)))
    polygon_outline_rgba = (*polygon_rgb, 255)
    hole_fill_rgba = (*hole_rgb, max(0, min(args.hole_fill_alpha, 255)))
    hole_outline_rgba = (*hole_rgb, 255)
    vertex_rgba = (*vertex_rgb, 255)
    bbox_rgba = (*bbox_rgb, 255)
    gt_bbox_rgba = (*gt_bbox_rgb, 255)

    tasks = [
        (
            image_info,
            image_path,
            predictions,
            gt_bboxes_by_image.get(int(image_info["id"]), []),
            args.output_dir,
            args.scale,
            polygon_fill_rgba,
            polygon_outline_rgba,
            hole_fill_rgba,
            hole_outline_rgba,
            vertex_rgba,
            bbox_rgba,
            gt_bbox_rgba,
            args.use_gt_hole_role,
            args.render_holes,
            args.nms_iou_threshold,
            args.nms_overlap_threshold,
            args.containment_threshold,
            not args.no_vertices,
            args.vertex_radius,
            args.line_width,
            args.draw_bboxes,
            args.bbox_width,
            args.draw_gt_bboxes,
            args.gt_bbox_width,
            args.bbox_expand_ratio,
            args.draw_scores,
            args.score_decimals,
        )
        for image_info, image_path, predictions in image_items
    ]
    return tasks, load_stats | {"num_missing_patch_images": missing_images}


def draw_score_label(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    score: float,
    decimals: int,
    text_color: tuple[int, int, int, int],
    background_color: tuple[int, int, int, int],
) -> None:
    font = ImageFont.load_default()
    text = f"{score:.{max(0, decimals)}f}"
    left, top, right, bottom = draw.textbbox((x, y), text, font=font)
    padding_x = 3
    padding_y = 2
    draw.rectangle(
        (left - padding_x, top - padding_y, right + padding_x, bottom + padding_y),
        fill=background_color,
    )
    draw.text((x, y), text, fill=text_color, font=font)


def expand_bbox(bbox: list[float], expand_ratio: float, image_width: int, image_height: int) -> list[float]:
    if len(bbox) != 4:
        return bbox
    x, y, w, h = bbox
    if expand_ratio <= 0:
        return [x, y, w, h]
    cx = x + 0.5 * w
    cy = y + 0.5 * h
    scale = 1.0 + expand_ratio
    new_w = w * scale
    new_h = h * scale
    new_x = max(0.0, cx - 0.5 * new_w)
    new_y = max(0.0, cy - 0.5 * new_h)
    new_right = min(float(image_width), cx + 0.5 * new_w)
    new_bottom = min(float(image_height), cy + 0.5 * new_h)
    return [new_x, new_y, max(0.0, new_right - new_x), max(0.0, new_bottom - new_y)]


def build_hole_aware_predictions(
    predictions: list[dict],
    nms_iou_threshold: float,
    nms_overlap_threshold: float,
    containment_threshold: float,
) -> tuple[list[dict], int]:
    try:
        from shapely.geometry import Polygon

        from export_roipoly_predictions_to_geojson import compute_depths
        from export_roipoly_predictions_to_geojson import deduplicate_predictions
        from export_roipoly_predictions_to_geojson import infer_parent_indices
        from export_roipoly_predictions_to_geojson import sanitize_polygon
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "--render-holes requires shapely. Run this script in the roipoly environment."
        ) from exc

    items = []
    for prediction in predictions:
        geometry = sanitize_polygon(Polygon(prediction["points"]), min_area=0.0)
        if geometry is None:
            continue
        items.append(
            {
                "prediction_index": prediction["prediction_index"],
                "geometry": geometry,
                "score": prediction["score"],
            }
        )

    deduped = deduplicate_predictions(
        items,
        iou_threshold=nms_iou_threshold,
        overlap_threshold=nms_overlap_threshold,
    )
    parents = infer_parent_indices(deduped, containment_threshold=containment_threshold)
    depths = compute_depths(parents)
    children_by_parent: dict[int, list[int]] = defaultdict(list)
    for child_index, parent_index in enumerate(parents):
        if parent_index is not None:
            children_by_parent[parent_index].append(child_index)

    merged_predictions = []
    rendered_hole_count = 0
    for index, item in enumerate(deduped):
        if depths[index] % 2 == 1:
            rendered_hole_count += 1
            continue
        geometry = item["geometry"]
        hole_indices = [child for child in children_by_parent[index] if depths[child] % 2 == 1]
        holes = [list(deduped[child]["geometry"].exterior.coords) for child in hole_indices]
        merged_geometry = sanitize_polygon(Polygon(list(geometry.exterior.coords), holes), min_area=0.0) or geometry
        merged_predictions.append(
            {
                "prediction_index": item["prediction_index"],
                "score": item["score"],
                "geometry": merged_geometry,
            }
        )
    return merged_predictions, rendered_hole_count


def draw_geometry_with_holes(
    overlay: Image.Image,
    geometry: Polygon,
    scale: float,
    polygon_fill_rgba: tuple[int, int, int, int],
    polygon_outline_rgba: tuple[int, int, int, int],
    vertex_rgba: tuple[int, int, int, int],
    draw_vertices: bool,
    vertex_radius: int,
    line_width: int,
) -> Image.Image:
    fill_mask = Image.new("L", overlay.size, 0)
    fill_draw = ImageDraw.Draw(fill_mask)
    outline_draw = ImageDraw.Draw(overlay, "RGBA")

    exterior = [(float(x) * scale, float(y) * scale) for x, y in geometry.exterior.coords]
    fill_draw.polygon(exterior, fill=polygon_fill_rgba[3])
    outline_draw.line(exterior, fill=polygon_outline_rgba, width=line_width)

    if draw_vertices:
        for x, y in exterior[:-1]:
            draw_vertex(outline_draw, x, y, vertex_radius, vertex_rgba)

    for interior in geometry.interiors:
        hole = [(float(x) * scale, float(y) * scale) for x, y in interior.coords]
        fill_draw.polygon(hole, fill=0)
        outline_draw.line(hole, fill=polygon_outline_rgba, width=line_width)
        if draw_vertices:
            for x, y in hole[:-1]:
                draw_vertex(outline_draw, x, y, vertex_radius, vertex_rgba)

    tint = Image.new("RGBA", overlay.size, polygon_fill_rgba[:3] + (0,))
    tint.putalpha(fill_mask)
    return Image.alpha_composite(overlay, tint)


def render_one(
    image_info: dict,
    image_path: Path,
    predictions: list[dict],
    gt_bboxes: list[list[float]],
    output_dir: Path,
    scale: float,
    polygon_fill_rgba: tuple[int, int, int, int],
    polygon_outline_rgba: tuple[int, int, int, int],
    hole_fill_rgba: tuple[int, int, int, int],
    hole_outline_rgba: tuple[int, int, int, int],
    vertex_rgba: tuple[int, int, int, int],
    bbox_rgba: tuple[int, int, int, int],
    gt_bbox_rgba: tuple[int, int, int, int],
    use_gt_hole_role: bool,
    render_holes: bool,
    nms_iou_threshold: float,
    nms_overlap_threshold: float,
    containment_threshold: float,
    draw_vertices: bool,
    vertex_radius: int,
    line_width: int,
    draw_bboxes: bool,
    bbox_width: int,
    draw_gt_bboxes: bool,
    gt_bbox_width: int,
    bbox_expand_ratio: float,
    draw_scores: bool,
    score_decimals: int,
) -> dict[str, str | int | float]:
    image = Image.open(image_path).convert("RGB")
    resized = image.resize(
        (max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale)))),
        resample=Image.Resampling.LANCZOS,
    )
    canvas = resized.convert("RGBA")
    overlay = Image.new("RGBA", resized.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    rendered_hole_count = 0
    render_predictions = predictions
    if render_holes:
        render_predictions, rendered_hole_count = build_hole_aware_predictions(
            predictions,
            nms_iou_threshold=nms_iou_threshold,
            nms_overlap_threshold=nms_overlap_threshold,
            containment_threshold=containment_threshold,
        )

    for prediction in render_predictions:
        is_gt_hole = bool(prediction.get("gt_is_hole", False)) if use_gt_hole_role else False
        current_fill_rgba = hole_fill_rgba if is_gt_hole else polygon_fill_rgba
        current_outline_rgba = hole_outline_rgba if is_gt_hole else polygon_outline_rgba

        if render_holes:
            overlay = draw_geometry_with_holes(
                overlay,
                prediction["geometry"],
                scale=scale,
                polygon_fill_rgba=current_fill_rgba,
                polygon_outline_rgba=current_outline_rgba,
                vertex_rgba=vertex_rgba,
                draw_vertices=draw_vertices,
                vertex_radius=vertex_radius,
                line_width=line_width,
            )
            draw = ImageDraw.Draw(overlay, "RGBA")
            min_x, min_y, max_x, max_y = prediction["geometry"].bounds
            display_bbox = expand_bbox(
                [float(min_x), float(min_y), float(max_x - min_x), float(max_y - min_y)],
                bbox_expand_ratio,
                image.width,
                image.height,
            )
        else:
            scaled_points = [(x * scale, y * scale) for x, y in prediction["points"]]
            if current_fill_rgba[3] > 0:
                draw.polygon(scaled_points, fill=current_fill_rgba, outline=current_outline_rgba, width=line_width)
            draw.line(scaled_points + [scaled_points[0]], fill=current_outline_rgba, width=line_width)

            if draw_vertices:
                if vertex_radius <= 0:
                    draw.point(scaled_points, fill=vertex_rgba)
                else:
                    for x, y in scaled_points:
                        draw_vertex(draw, x, y, vertex_radius, vertex_rgba)

            bbox = prediction["bbox"]
            display_bbox = expand_bbox(bbox, bbox_expand_ratio, image.width, image.height)

        if draw_bboxes and len(display_bbox) == 4:
            left = display_bbox[0] * scale
            top = display_bbox[1] * scale
            right = (display_bbox[0] + display_bbox[2]) * scale
            bottom = (display_bbox[1] + display_bbox[3]) * scale
            draw.rectangle((left, top, right, bottom), outline=bbox_rgba, width=bbox_width)

        if draw_scores and len(display_bbox) == 4:
            label_x = display_bbox[0] * scale + 2
            label_y = max(0.0, display_bbox[1] * scale + 2)
            draw_score_label(
                draw,
                label_x,
                label_y,
                prediction["score"],
                decimals=score_decimals,
                text_color=(0, 0, 0, 255),
                background_color=current_fill_rgba[:3] + (220,),
            )

    if draw_gt_bboxes:
        for gt_bbox in gt_bboxes:
            display_gt_bbox = expand_bbox(gt_bbox, bbox_expand_ratio, image.width, image.height)
            if len(display_gt_bbox) != 4:
                continue
            left = display_gt_bbox[0] * scale
            top = display_gt_bbox[1] * scale
            right = (display_gt_bbox[0] + display_gt_bbox[2]) * scale
            bottom = (display_gt_bbox[1] + display_gt_bbox[3]) * scale
            draw.rectangle((left, top, right, bottom), outline=gt_bbox_rgba, width=gt_bbox_width)

    rendered = Image.alpha_composite(canvas, overlay).convert("RGB")
    output_name = f"{Path(image_info['file_name']).stem}.png"
    rendered.save(output_dir / output_name)

    scores = [prediction["score"] for prediction in render_predictions]
    return {
        "image_id": int(image_info["id"]),
        "file_name": str(image_info["file_name"]),
        "source_image_name": str(image_info.get("source_image_name", "")),
        "polygon_count": len(render_predictions),
        "gt_bbox_count": len(gt_bboxes),
        "rendered_hole_count": rendered_hole_count,
        "gt_hole_prediction_count": sum(1 for prediction in render_predictions if prediction.get("gt_is_hole", False)),
        "max_score": max(scores) if scores else 0.0,
        "mean_score": sum(scores) / len(scores) if scores else 0.0,
        "output_png": output_name,
    }


def render_task(task):
    return render_one(*task)


def write_contact_sheet(output_dir: Path, rows: list[dict[str, str | int | float]], columns: int) -> Path | None:
    png_paths = [output_dir / row["output_png"] for row in rows if (output_dir / row["output_png"]).is_file()]
    if not png_paths:
        return None

    images = [Image.open(path).convert("RGB") for path in png_paths]
    thumb_size = (max(1, images[0].width), max(1, images[0].height))
    margin = 20
    header = 42
    columns = max(1, columns)
    rows_count = (len(images) + columns - 1) // columns
    sheet = Image.new(
        "RGB",
        (columns * thumb_size[0] + (columns + 1) * margin, rows_count * (thumb_size[1] + header) + margin),
        (255, 255, 255),
    )
    draw = ImageDraw.Draw(sheet)

    for index, (image, meta) in enumerate(zip(images, rows)):
        row = index // columns
        col = index % columns
        x = margin + col * (thumb_size[0] + margin)
        y = margin + row * (thumb_size[1] + header)
        sheet.paste(image, (x, y + header))
        draw.text((x, y), Path(str(meta["output_png"])).stem, fill=(0, 0, 0))

    contact_path = output_dir / "contact_sheet.png"
    sheet.save(contact_path)
    return contact_path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks, load_stats = build_tasks(args)

    if args.workers <= 1:
        rows = [render_task(task) for task in tasks]
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            rows = list(executor.map(render_task, tasks, chunksize=1))

    print(f"Wrote {len(rows)} visualization(s) to {args.output_dir}")
    print(json.dumps(load_stats, ensure_ascii=False, indent=2))

    if args.contact_sheet:
        contact_path = write_contact_sheet(args.output_dir, rows, columns=args.contact_columns)
        if contact_path is not None:
            print(f"Contact sheet: {contact_path}")


if __name__ == "__main__":
    main()
