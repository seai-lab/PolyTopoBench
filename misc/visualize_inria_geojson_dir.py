#!/usr/bin/env python3
"""Visualize a directory of Inria GeoJSON files over their matching source images."""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageColor, ImageDraw

from visualize_inria_gt_polygonized import render_one


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True, help="Directory containing source .tif images.")
    parser.add_argument("--geojson-dir", type=Path, required=True, help="Directory containing per-image .geojson files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for rendered PNG overlays.")
    parser.add_argument(
        "--image-stems",
        nargs="*",
        default=None,
        help="Optional subset of image stems to render. Defaults to all matching stems.",
    )
    parser.add_argument("--max-images", type=int, default=0, help="Render at most this many images. 0 means all.")
    parser.add_argument("--scale", type=float, default=0.25, help="Display scale applied to the source image.")
    parser.add_argument("--polygon-color", default="#ffd400", help="Color for polygon fill and outline.")
    parser.add_argument("--hole-color", default="#ff2a2a", help="Color for hole fill and outline.")
    parser.add_argument("--polygon-fill-alpha", type=int, default=20, help="Polygon fill alpha in [0, 255].")
    parser.add_argument("--hole-fill-alpha", type=int, default=80, help="Hole fill alpha in [0, 255].")
    parser.add_argument("--line-width", type=int, default=2, help="Outline width on the resized image.")
    parser.add_argument("--vertex-color", default="#1f6dff", help="Color for vertices.")
    parser.add_argument("--no-vertices", action="store_true", help="Disable vertex markers.")
    parser.add_argument("--vertex-radius", type=int, default=0, help="Vertex radius in pixels.")
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, max(1, (os.cpu_count() or 1) // 2)),
        help="Number of worker threads used for rendering.",
    )
    parser.add_argument(
        "--contact-sheet",
        action="store_true",
        help="Also write a simple contact sheet of the rendered PNGs.",
    )
    parser.add_argument("--contact-columns", type=int, default=2, help="Column count for the contact sheet.")
    return parser.parse_args()


def build_tasks(args: argparse.Namespace):
    if not args.image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {args.image_dir}")
    if not args.geojson_dir.is_dir():
        raise FileNotFoundError(f"GeoJSON directory not found: {args.geojson_dir}")

    allowed_stems = set(args.image_stems) if args.image_stems else None
    samples = []
    for geojson_path in sorted(args.geojson_dir.glob("*.geojson")):
        if allowed_stems is not None and geojson_path.stem not in allowed_stems:
            continue
        image_path = args.image_dir / f"{geojson_path.stem}.tif"
        if image_path.is_file():
            samples.append((image_path, geojson_path))
    if args.max_images > 0:
        samples = samples[: args.max_images]

    polygon_rgb = ImageColor.getrgb(args.polygon_color)
    hole_rgb = ImageColor.getrgb(args.hole_color)
    vertex_rgb = ImageColor.getrgb(args.vertex_color)
    polygon_fill_rgba = (*polygon_rgb, max(0, min(args.polygon_fill_alpha, 255)))
    polygon_outline_rgba = (*polygon_rgb, 255)
    hole_fill_rgba = (*hole_rgb, max(0, min(args.hole_fill_alpha, 255)))
    hole_outline_rgba = (*hole_rgb, 255)
    vertex_rgba = (*vertex_rgb, 255)

    tasks = [
        (
            image_path,
            geojson_path,
            args.output_dir,
            args.scale,
            polygon_fill_rgba,
            polygon_outline_rgba,
            hole_fill_rgba,
            hole_outline_rgba,
            vertex_rgba,
            not args.no_vertices,
            args.vertex_radius,
            args.line_width,
        )
        for image_path, geojson_path in samples
    ]
    return tasks


def write_contact_sheet(output_dir: Path, rows: list[dict[str, str | int]], columns: int) -> Path | None:
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

    contact_path = output_dir.parent / "contact_sheet.png"
    sheet.save(contact_path)
    return contact_path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks = build_tasks(args)

    if args.workers <= 1:
        rows = [render_one(*task) for task in tasks]
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            rows = list(executor.map(lambda task: render_one(*task), tasks, chunksize=1))

    print(f"Wrote {len(rows)} visualization(s) to {args.output_dir}")
    if args.contact_sheet:
        contact_path = write_contact_sheet(args.output_dir, rows, columns=args.contact_columns)
        if contact_path is not None:
            print(f"Contact sheet: {contact_path}")


if __name__ == "__main__":
    main()
