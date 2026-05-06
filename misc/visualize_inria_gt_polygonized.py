#!/usr/bin/env python3
"""Visualize Inria gt_polygonized annotations over 40%-resolution imagery."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageColor, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/inria_dataset_aligned"),
        help="Root directory of the aligned Inria dataset.",
    )
    parser.add_argument(
        "--split",
        choices=("train",),
        default="train",
        help="Dataset split to visualize.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Maximum number of images to render. Use 0 or a negative value for all.",
    )
    parser.add_argument(
        "--image-stems",
        nargs="*",
        default=None,
        help="Optional subset of image stems to render.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.6,
        help="Resize ratio applied to the background image before rendering annotations.",
    )
    parser.add_argument(
        "--polygon-color",
        default="#ffd400",
        help="Color for normal polygon regions.",
    )
    parser.add_argument(
        "--hole-color",
        default="#ff2a2a",
        help="Color for hole regions.",
    )
    parser.add_argument(
        "--polygon-fill-alpha",
        type=int,
        default=28,
        help="Polygon fill alpha in [0, 255].",
    )
    parser.add_argument(
        "--hole-fill-alpha",
        type=int,
        default=72,
        help="Hole fill alpha in [0, 255].",
    )
    parser.add_argument(
        "--line-width",
        type=int,
        default=2,
        help="Outline width in pixels.",
    )
    parser.add_argument(
        "--vertex-color",
        default="#1f6dff",
        help="Color for polygon vertices.",
    )
    parser.add_argument(
        "--no-vertices",
        action="store_true",
        help="Disable drawing polygon vertices.",
    )
    parser.add_argument(
        "--vertex-radius",
        type=int,
        default=0,
        help="Vertex marker radius in pixels on the resized image.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, max(1, (os.cpu_count() or 1) // 2)),
        help="Number of worker threads used to render images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write visualizations. Defaults to dataset_root/visualizations/gt_polygonized_<split>_scale40.",
    )
    return parser.parse_args()


def geometry_iter(obj: dict) -> list[dict]:
    if obj.get("type") == "GeometryCollection":
        return obj.get("geometries", [])
    if obj.get("type") == "FeatureCollection":
        return [feature.get("geometry", {}) for feature in obj.get("features", [])]
    return [obj]


def draw_vertex(draw: ImageDraw.ImageDraw, x: float, y: float, radius: int, color: tuple[int, int, int, int]) -> None:
    if radius <= 0:
        draw.point((x, y), fill=color)
        return
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)


def draw_polygon_rings(
    draw: ImageDraw.ImageDraw,
    rings: list[list[list[float]]],
    polygon_fill_rgba: tuple[int, int, int, int],
    polygon_outline_rgba: tuple[int, int, int, int],
    hole_fill_rgba: tuple[int, int, int, int],
    hole_outline_rgba: tuple[int, int, int, int],
    vertex_rgba: tuple[int, int, int, int],
    draw_vertices: bool,
    vertex_radius: int,
    line_width: int,
    scale: float,
) -> tuple[int, int]:
    if not rings:
        return 0, 0

    exterior = [(float(x) * scale, float(y) * scale) for x, y in rings[0]]
    if len(exterior) < 3:
        return 0, 0

    if polygon_fill_rgba[3] > 0:
        draw.polygon(exterior, fill=polygon_fill_rgba, outline=polygon_outline_rgba, width=line_width)
    draw.line(exterior, fill=polygon_outline_rgba, width=line_width)
    if draw_vertices:
        if vertex_radius <= 0:
            draw.point(exterior, fill=vertex_rgba)
        else:
            for x, y in exterior:
                draw_vertex(draw, x, y, vertex_radius, vertex_rgba)

    hole_count = max(0, len(rings) - 1)
    for hole_ring in rings[1:]:
        hole = [(float(x) * scale, float(y) * scale) for x, y in hole_ring]
        if len(hole) < 3:
            continue
        if hole_fill_rgba[3] > 0:
            draw.polygon(hole, fill=hole_fill_rgba, outline=hole_outline_rgba, width=line_width)
        draw.line(hole, fill=hole_outline_rgba, width=line_width)
        if draw_vertices:
            if vertex_radius <= 0:
                draw.point(hole, fill=vertex_rgba)
            else:
                for x, y in hole:
                    draw_vertex(draw, x, y, vertex_radius, vertex_rgba)

    return 1, hole_count


def render_one(
    image_path: Path,
    geojson_path: Path,
    output_dir: Path,
    scale: float,
    polygon_fill_rgba: tuple[int, int, int, int],
    polygon_outline_rgba: tuple[int, int, int, int],
    hole_fill_rgba: tuple[int, int, int, int],
    hole_outline_rgba: tuple[int, int, int, int],
    vertex_rgba: tuple[int, int, int, int],
    draw_vertices: bool,
    vertex_radius: int,
    line_width: int,
) -> dict[str, str | int]:
    image = Image.open(image_path).convert("RGB")
    resized = image.resize(
        (max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale)))),
        resample=Image.Resampling.LANCZOS,
    )
    canvas = resized.convert("RGBA")
    overlay = Image.new("RGBA", resized.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    obj = json.loads(geojson_path.read_text())
    polygon_count = 0
    polygon_with_hole_count = 0
    ring_count = 0

    for geom in geometry_iter(obj):
        geom_type = geom.get("type")
        if geom_type == "Polygon":
            rings = geom.get("coordinates", [])
            count_inc, holes_inc = draw_polygon_rings(
                draw=draw,
                rings=rings,
                polygon_fill_rgba=polygon_fill_rgba,
                polygon_outline_rgba=polygon_outline_rgba,
                hole_fill_rgba=hole_fill_rgba,
                hole_outline_rgba=hole_outline_rgba,
                vertex_rgba=vertex_rgba,
                draw_vertices=draw_vertices,
                vertex_radius=vertex_radius,
                line_width=line_width,
                scale=scale,
            )
            polygon_count += count_inc
            polygon_with_hole_count += int(holes_inc > 0)
            ring_count += len(rings)
        elif geom_type == "MultiPolygon":
            for rings in geom.get("coordinates", []):
                count_inc, holes_inc = draw_polygon_rings(
                    draw=draw,
                    rings=rings,
                    polygon_fill_rgba=polygon_fill_rgba,
                    polygon_outline_rgba=polygon_outline_rgba,
                    hole_fill_rgba=hole_fill_rgba,
                    hole_outline_rgba=hole_outline_rgba,
                    vertex_rgba=vertex_rgba,
                    draw_vertices=draw_vertices,
                    vertex_radius=vertex_radius,
                    line_width=line_width,
                    scale=scale,
                )
                polygon_count += count_inc
                polygon_with_hole_count += int(holes_inc > 0)
                ring_count += len(rings)

    rendered = Image.alpha_composite(canvas, overlay).convert("RGB")
    output_name = f"{image_path.stem}.png"
    rendered.save(output_dir / output_name)
    return {
        "file_name": geojson_path.name,
        "polygon_count": polygon_count,
        "polygon_with_hole_count": polygon_with_hole_count,
        "ring_count": ring_count,
        "output_png": output_name,
    }


def main() -> None:
    args = parse_args()
    image_dir = args.dataset_root / args.split / "images"
    geojson_dir = args.dataset_root / "raw" / args.split / "gt_polygonized"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not geojson_dir.is_dir():
        raise FileNotFoundError(f"gt_polygonized directory not found: {geojson_dir}")

    output_root = args.output_dir or (args.dataset_root / "visualizations" / f"gt_polygonized_{args.split}_scale40")
    output_dir = output_root / "per_file"
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    allowed_stems = set(args.image_stems) if args.image_stems else None
    for geojson_path in sorted(geojson_dir.glob("*.geojson")):
        if allowed_stems is not None and geojson_path.stem not in allowed_stems:
            continue
        image_path = image_dir / f"{geojson_path.stem}.tif"
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
            output_dir,
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

    if args.workers <= 1:
        rows = [render_one(*task) for task in tasks]
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            rows = list(executor.map(lambda task: render_one(*task), tasks, chunksize=1))

    print(f"Wrote {len(rows)} visualization(s) to {output_dir}")

if __name__ == "__main__":
    main()
