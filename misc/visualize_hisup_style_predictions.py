#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


GT_OUTER_COLOR = (46, 204, 113, 255)
GT_HOLE_COLOR = (26, 188, 156, 255)
PRED_OUTER_COLOR = (231, 76, 60, 255)
PRED_HOLE_COLOR = (241, 196, 15, 255)
TEXT_BG_COLOR = (255, 255, 255, 230)
TEXT_COLOR = (20, 20, 20, 255)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render HiSup-style polygon predictions on patch images. Supports "
            "prediction JSONs stored as either a list of annotations or a COCO-style dict."
        )
    )
    parser.add_argument("--annotations", type=Path, required=True, help="COCO annotation JSON with image metadata.")
    parser.add_argument("--predictions", type=Path, required=True, help="Prediction JSON.")
    parser.add_argument("--images-dir", type=Path, required=True, help="Patch image directory.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write rendered PNGs.")
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.5,
        help="Drop predictions below this score before rendering.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Maximum number of images to render. Use 0 for all images.",
    )
    parser.add_argument(
        "--sort-by",
        choices=["image_id", "prediction_count"],
        default="image_id",
        help="Image ordering used before applying --max-images.",
    )
    parser.add_argument(
        "--line-width",
        type=int,
        default=2,
        help="Ring outline width in pixels.",
    )
    parser.add_argument(
        "--draw-gt",
        action="store_true",
        help="Overlay ground-truth polygons from the annotation JSON.",
    )
    parser.add_argument(
        "--overview-page-size",
        type=int,
        default=25,
        help="Number of thumbnails per overview page. Use 0 to disable overview pages.",
    )
    parser.add_argument(
        "--thumb-size",
        type=int,
        default=256,
        help="Maximum thumbnail edge length used in overview pages.",
    )
    return parser.parse_args()


def ensure_uint8_rgb(image: Image.Image) -> np.ndarray:
    array = np.asarray(image.convert("RGB"))
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_image_index(annotation_path: Path) -> tuple[dict[int, dict], dict[int, list[dict]]]:
    annotation_json = load_json(annotation_path)
    images = {int(item["id"]): item for item in annotation_json["images"]}
    ann_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in annotation_json.get("annotations", []):
        ann_by_image[int(ann["image_id"])].append(ann)
    return images, ann_by_image


def load_predictions(prediction_path: Path, score_threshold: float) -> dict[int, list[dict]]:
    payload = load_json(prediction_path)
    if isinstance(payload, dict):
        items = payload.get("annotations", [])
    else:
        items = payload
    grouped: dict[int, list[dict]] = defaultdict(list)
    for item in items:
        score = float(item.get("score", 1.0))
        if score < score_threshold:
            continue
        grouped[int(item["image_id"])].append(item)
    return grouped


def polygon_rings(annotation: dict) -> list[np.ndarray]:
    rings: list[np.ndarray] = []
    for ring in annotation.get("segmentation", []):
        if not isinstance(ring, list) or len(ring) < 6 or len(ring) % 2 != 0:
            continue
        coords = np.asarray(ring, dtype=np.float32).reshape(-1, 2)
        if coords.shape[0] >= 3:
            rings.append(coords)
    return rings


def draw_ring_set(
    draw: ImageDraw.ImageDraw,
    annotations: list[dict],
    outer_color: tuple[int, int, int, int],
    hole_color: tuple[int, int, int, int],
    line_width: int,
) -> None:
    for annotation in annotations:
        rings = polygon_rings(annotation)
        for ring_index, ring in enumerate(rings):
            color = outer_color if ring_index == 0 else hole_color
            points = [tuple(np.round(point).astype(int).tolist()) for point in ring]
            if len(points) >= 2:
                draw.line(points + [points[0]], fill=color, width=line_width)


def draw_header(
    canvas: Image.Image,
    *,
    image_id: int,
    file_name: str,
    gt_count: int,
    pred_count: int,
    score_threshold: float,
) -> Image.Image:
    header_height = 44
    panel = Image.new("RGBA", (canvas.width, canvas.height + header_height), (255, 255, 255, 255))
    panel.alpha_composite(canvas, dest=(0, header_height))
    draw = ImageDraw.Draw(panel, "RGBA")
    draw.rectangle((0, 0, panel.width, header_height), fill=TEXT_BG_COLOR)
    font = ImageFont.load_default()
    message = (
        f"image_id={image_id}  file={file_name}  "
        f"gt={gt_count}  pred={pred_count}  score>={score_threshold:g}"
    )
    draw.text((10, 14), message, fill=TEXT_COLOR, font=font)
    return panel.convert("RGB")


def render_one(
    *,
    image_meta: dict,
    image_path: Path,
    gt_annotations: list[dict],
    pred_annotations: list[dict],
    line_width: int,
    draw_gt: bool,
    score_threshold: float,
    output_path: Path,
) -> dict:
    image_rgb = ensure_uint8_rgb(Image.open(image_path))
    overlay = Image.fromarray(image_rgb).convert("RGBA")
    draw = ImageDraw.Draw(overlay, "RGBA")

    if draw_gt:
        draw_ring_set(
            draw,
            gt_annotations,
            outer_color=GT_OUTER_COLOR,
            hole_color=GT_HOLE_COLOR,
            line_width=line_width,
        )
    draw_ring_set(
        draw,
        pred_annotations,
        outer_color=PRED_OUTER_COLOR,
        hole_color=PRED_HOLE_COLOR,
        line_width=line_width,
    )

    rendered = draw_header(
        overlay,
        image_id=int(image_meta["id"]),
        file_name=str(image_meta["file_name"]),
        gt_count=len(gt_annotations),
        pred_count=len(pred_annotations),
        score_threshold=score_threshold,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered.save(output_path)
    return {
        "image_id": int(image_meta["id"]),
        "file_name": str(image_meta["file_name"]),
        "gt_count": len(gt_annotations),
        "pred_count": len(pred_annotations),
        "output_path": str(output_path),
    }


def build_overview_pages(
    *,
    rows: list[dict],
    output_dir: Path,
    page_size: int,
    thumb_size: int,
) -> list[str]:
    if page_size <= 0 or not rows:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    cols = 5
    rows_per_page = max(1, math.ceil(page_size / cols))
    font = ImageFont.load_default()
    margin = 12
    label_height = 24
    canvas_w = cols * thumb_size + (cols + 1) * margin
    canvas_h = rows_per_page * (thumb_size + label_height) + (rows_per_page + 1) * margin
    saved_paths: list[str] = []

    total_pages = math.ceil(len(rows) / page_size)
    for page_index in range(total_pages):
        batch = rows[page_index * page_size : (page_index + 1) * page_size]
        canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        for item_index, item in enumerate(batch):
            row_index = item_index // cols
            col_index = item_index % cols
            x = margin + col_index * thumb_size
            y = margin + row_index * (thumb_size + label_height)
            image = Image.open(item["output_path"]).convert("RGB")
            image.thumbnail((thumb_size, thumb_size))
            canvas.paste(image, (x, y))
            label = f"id={item['image_id']} pred={item['pred_count']}"
            draw.text((x, y + thumb_size + 4), label, fill=(20, 20, 20), font=font)
        out_path = output_dir / f"overview_page_{page_index + 1:02d}.png"
        canvas.save(out_path)
        saved_paths.append(str(out_path))
    return saved_paths


def main() -> None:
    args = parse_args()

    image_index, gt_by_image = load_image_index(args.annotations)
    pred_by_image = load_predictions(args.predictions, score_threshold=args.score_threshold)

    image_ids = sorted(image_index)
    if args.sort_by == "prediction_count":
        image_ids.sort(key=lambda image_id: (-len(pred_by_image.get(image_id, [])), image_id))

    if args.max_images > 0:
        image_ids = image_ids[: args.max_images]

    single_dir = args.output_dir / "single_images"
    overview_dir = args.output_dir / "overview_pages"
    rows: list[dict] = []

    for index, image_id in enumerate(image_ids, 1):
        image_meta = image_index[image_id]
        image_path = args.images_dir / str(image_meta["file_name"])
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")
        output_path = single_dir / f"{Path(str(image_meta['file_name'])).stem}.png"
        row = render_one(
            image_meta=image_meta,
            image_path=image_path,
            gt_annotations=gt_by_image.get(image_id, []),
            pred_annotations=pred_by_image.get(image_id, []),
            line_width=args.line_width,
            draw_gt=args.draw_gt,
            score_threshold=args.score_threshold,
            output_path=output_path,
        )
        rows.append(row)
        if index % 25 == 0 or index == len(image_ids):
            print(f"rendered={index}/{len(image_ids)}", flush=True)

    overview_paths = build_overview_pages(
        rows=rows,
        output_dir=overview_dir,
        page_size=args.overview_page_size,
        thumb_size=args.thumb_size,
    )

    summary = {
        "annotations": str(args.annotations),
        "predictions": str(args.predictions),
        "images_dir": str(args.images_dir),
        "score_threshold": args.score_threshold,
        "draw_gt": args.draw_gt,
        "rendered_images": len(rows),
        "overview_pages": overview_paths,
        "items": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} visualization(s) to {args.output_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
