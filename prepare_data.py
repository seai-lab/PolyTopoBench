#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

try:
    from shapely.affinity import translate
    from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box, shape
    from shapely.strtree import STRtree
except ModuleNotFoundError:
    translate = None
    GeometryCollection = None
    MultiPolygon = None
    Polygon = None
    box = None
    shape = None
    STRtree = None


Image.MAX_IMAGE_PIXELS = None

INRIA_DATASET = "inria_building"
DEVENTER_DATASET = "deventer_512_valtest_as_val"
PATCH_SIZE = 512
CATEGORY_ID = 100
DEVENTER_TASKS = ("road", "vegetation", "unvegetated")
DEVENTER_MASK_VALUE = {
    "building": 0,
    "road": 1,
    "unvegetated": 2,
    "vegetation": 3,
    "water": 4,
}
METHODS = (
    "unet_poly",
    "maskrcnn_poly",
    "sam2_poly",
    "hisup",
    "acpvnet",
    "ffl",
    "gcp",
    "holitracer",
    "pix2poly",
    "polyworld",
    "roipoly",
)
DEFAULT_METHODS = tuple(method for method in METHODS if method != "acpvnet")
VECTOR_MIRRORS = ("gcp", "pix2poly", "polyworld")
SEG_METHOD_DIRS = {
    "unet_poly": "unet_seg",
    "maskrcnn_poly": "maskrcnn_seg",
    "sam2_poly": "sam2_seg",
}
INRIA_VAL_IMAGES = {
    "austin12.tif",
    "austin14.tif",
    "austin17.tif",
    "austin24.tif",
    "austin30.tif",
    "austin33.tif",
    "austin6.tif",
    "chicago14.tif",
    "chicago21.tif",
    "chicago24.tif",
    "chicago25.tif",
    "chicago29.tif",
    "chicago36.tif",
    "chicago4.tif",
    "kitsap10.tif",
    "kitsap12.tif",
    "kitsap18.tif",
    "kitsap19.tif",
    "kitsap2.tif",
    "kitsap28.tif",
    "kitsap30.tif",
    "tyrol-w18.tif",
    "tyrol-w22.tif",
    "tyrol-w24.tif",
    "tyrol-w25.tif",
    "tyrol-w28.tif",
    "tyrol-w30.tif",
    "tyrol-w4.tif",
    "vienna15.tif",
    "vienna19.tif",
    "vienna2.tif",
    "vienna23.tif",
    "vienna24.tif",
    "vienna5.tif",
    "vienna9.tif",
}
INRIA_NAME_RE = re.compile(r"^(?P<city>[a-zA-Z-]+)(?P<number>\d+)\.tif$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare PolyTopoBench data from raw Inria and Deventer data.")
    parser.add_argument("--inria-root", type=Path, default=None)
    parser.add_argument("--deventer-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--datasets", nargs="+", default=["all"], choices=["all", INRIA_DATASET, DEVENTER_DATASET])
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS), choices=["all", *METHODS])
    parser.add_argument("--deventer-tasks", nargs="+", default=list(DEVENTER_TASKS), choices=list(DEVENTER_TASKS))
    parser.add_argument("--inria-split-csv", type=Path, default=None)
    parser.add_argument("--smoke-train-images", type=int, default=32)
    parser.add_argument("--smoke-val-images", type=int, default=16)
    parser.add_argument("--jpg-quality", type=int, default=95)
    parser.add_argument("--roipoly-num-corners", type=int, default=64)
    parser.add_argument("--acpv-sigma", type=float, default=3.0)
    parser.add_argument("--encode-acpv-latents", action="store_true")
    parser.add_argument("--acpv-autoencoder-config", type=Path, default=None)
    parser.add_argument("--acpv-latent-batch-size", type=int, default=8)
    parser.add_argument("--acpv-latent-num-workers", type=int, default=4)
    parser.add_argument("--acpv-latent-scale-samples", type=int, default=128)
    parser.add_argument("--limit-source-images", type=int, default=0, help="Debug option; 0 uses all source images.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def selected_datasets(values: list[str]) -> set[str]:
    return {INRIA_DATASET, DEVENTER_DATASET} if values == ["all"] else set(values)


def selected_methods(values: list[str]) -> set[str]:
    return set(METHODS) if values == ["all"] else set(values)


def require_shapely() -> None:
    if Polygon is None:
        raise ModuleNotFoundError("prepare_data.py requires shapely. Install the release requirements first.")


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (Path.cwd() / path)


def output_root_from_args(args: argparse.Namespace, release_root: Path) -> Path:
    return release_root / "dataset" / "data_processed" if args.output_root is None else resolve_path(args.output_root)


def inria_root_from_args(args: argparse.Namespace, release_root: Path) -> Path:
    return release_root / "dataset" / "inria_dataset_aligned" if args.inria_root is None else resolve_path(args.inria_root)


def deventer_root_from_args(args: argparse.Namespace, release_root: Path) -> Path:
    return release_root / "dataset" / "deventer_512_valtest_as_val" if args.deventer_root is None else resolve_path(args.deventer_root)


def acpv_config_path(args: argparse.Namespace, release_root: Path) -> Path:
    if args.acpv_autoencoder_config:
        return resolve_path(args.acpv_autoencoder_config)
    return release_root / "models" / "acpvnet" / "source" / "config-files" / "autoencoder_kl_f4.yaml"


def validate_acpv_options(args: argparse.Namespace, release_root: Path, methods: set[str]) -> None:
    if "acpvnet" not in methods:
        return
    if not args.encode_acpv_latents:
        raise SystemExit(
            "ACPV-Net preprocessing requires latent heatmap tensors. "
            "Use --encode-acpv-latents --acpv-autoencoder-config <path>, "
            "or omit acpvnet from --methods."
        )
    config_path = acpv_config_path(args, release_root)
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing ACPV autoencoder config: {config_path}")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")


def prepare_root(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {path}. Re-run with --overwrite.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def copy_tree_files(src: Path, dst: Path) -> None:
    for item in src.rglob("*"):
        if item.is_file():
            link_or_copy(item, dst / item.relative_to(src))


def write_split_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def smoke_subset(coco: dict[str, Any], max_images: int) -> dict[str, Any]:
    images = sorted(coco.get("images", []), key=lambda image: int(image["id"]))[:max_images]
    image_ids = {int(image["id"]) for image in images}
    annotations = [
        dict(annotation, id=new_id)
        for new_id, annotation in enumerate(
            [ann for ann in coco.get("annotations", []) if int(ann["image_id"]) in image_ids],
            start=1,
        )
    ]
    return {
        "info": dict(coco.get("info", {}), smoke_subset=True, max_images=max_images),
        "licenses": coco.get("licenses", []),
        "images": images,
        "annotations": annotations,
        "categories": coco.get("categories", []),
    }


def normalize_ring(segment: list[float]) -> np.ndarray:
    ring = np.asarray(segment, dtype=np.float32).reshape(-1, 2)
    if ring.shape[0] >= 2 and np.allclose(ring[0], ring[-1]):
        ring = ring[:-1]
    if ring.shape[0] < 3:
        return np.zeros((0, 2), dtype=np.float32)
    return ring


def closed_ring_flat(segment: list[float]) -> list[float]:
    ring = normalize_ring(segment)
    if ring.size == 0:
        return []
    closed = np.concatenate([ring, ring[:1]], axis=0)
    return closed.reshape(-1).astype(float).tolist()


def sanitize_segmentation(segmentation: list[list[float]]) -> list[list[float]]:
    result: list[list[float]] = []
    for segment in segmentation:
        closed = closed_ring_flat(segment)
        if closed:
            result.append(closed)
    return result


def polygon_bbox_from_ring(ring: np.ndarray) -> list[float]:
    min_x = float(ring[:, 0].min())
    min_y = float(ring[:, 1].min())
    max_x = float(ring[:, 0].max())
    max_y = float(ring[:, 1].max())
    return [min_x, min_y, max_x - min_x, max_y - min_y]


def anns_by_image(coco: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in coco.get("annotations", []):
        grouped[int(annotation["image_id"])].append(annotation)
    return grouped


def render_mask(width: int, height: int, annotations: list[dict[str, Any]]) -> np.ndarray:
    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    for annotation in annotations:
        segments = annotation.get("segmentation", [])
        if not segments:
            continue
        exterior = [(segments[0][i], segments[0][i + 1]) for i in range(0, len(segments[0]), 2)]
        if len(exterior) >= 3:
            draw.polygon(exterior, fill=1)
        for hole in segments[1:]:
            interior = [(hole[i], hole[i + 1]) for i in range(0, len(hole), 2)]
            if len(interior) >= 3:
                draw.polygon(interior, fill=0)
    return np.asarray(canvas, dtype=np.uint8)


def save_binary_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8), mode="L").save(path)


def category_coco(category_name: str, images: list[dict[str, Any]], annotations: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "info": {"dataset": DEVENTER_DATASET, "category": category_name},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [
            {
                "id": CATEGORY_ID,
                "name": category_name,
                "supercategory": category_name,
                "source_mask_value": DEVENTER_MASK_VALUE.get(category_name),
            }
        ],
    }


def iter_geojson_polygons(obj: dict[str, Any]) -> Iterable[Polygon]:
    obj_type = obj.get("type")
    if obj_type == "FeatureCollection":
        for feature in obj.get("features", []):
            yield from iter_geojson_polygons(feature)
        return
    if obj_type == "Feature":
        yield from iter_geojson_polygons(obj.get("geometry") or {})
        return
    if obj_type == "GeometryCollection":
        for geometry in obj.get("geometries", []):
            yield from iter_geojson_polygons(geometry)
        return
    if obj_type == "Polygon":
        polygon = shape(obj)
        if not polygon.is_empty:
            yield polygon
        return
    if obj_type == "MultiPolygon":
        multipolygon = shape(obj)
        for polygon in multipolygon.geoms:
            if not polygon.is_empty:
                yield polygon
        return
    raise ValueError(f"Unsupported GeoJSON type: {obj_type}")


def iter_polygonal(geometry) -> Iterable[Polygon]:
    if geometry.is_empty:
        return
    if isinstance(geometry, Polygon):
        yield geometry
        return
    if isinstance(geometry, MultiPolygon):
        for item in geometry.geoms:
            if not item.is_empty:
                yield item
        return
    if isinstance(geometry, GeometryCollection):
        for item in geometry.geoms:
            yield from iter_polygonal(item)


def load_tile_polygons(path: Path) -> list[dict[str, Any]]:
    obj = read_json(path)
    records = []
    for index, polygon in enumerate(iter_geojson_polygons(obj)):
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if polygon.is_empty:
            continue
        if isinstance(polygon, MultiPolygon):
            for part_index, part in enumerate(polygon.geoms):
                if not part.is_empty:
                    records.append({"geometry": part, "geometry_index": index, "part_index": part_index})
        elif isinstance(polygon, Polygon):
            records.append({"geometry": polygon, "geometry_index": index, "part_index": 0})
    return records


def ring_flat(coords) -> list[float]:
    arr = np.asarray(coords, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] < 4 or arr.shape[1] != 2:
        return []
    return arr.reshape(-1).astype(float).tolist()


def polygon_to_segmentation(polygon: Polygon) -> list[list[float]]:
    segments = []
    exterior = ring_flat(polygon.exterior.coords)
    if exterior:
        segments.append(exterior)
    for interior in polygon.interiors:
        ring = ring_flat(interior.coords)
        if ring:
            segments.append(ring)
    return segments


def patch_origins(width: int, height: int, patch_size: int = PATCH_SIZE) -> list[tuple[int, int]]:
    xs = list(range(0, max(width - patch_size, 0), patch_size))
    ys = list(range(0, max(height - patch_size, 0), patch_size))
    if not xs or xs[-1] != width - patch_size:
        xs.append(width - patch_size)
    if not ys or ys[-1] != height - patch_size:
        ys.append(height - patch_size)
    return [(int(x), int(y)) for y in ys for x in xs]


def inria_split_rows(images_dir: Path, split_csv: Path | None, limit: int) -> dict[str, list[dict[str, Any]]]:
    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    if split_csv is not None:
        with split_csv.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                image_name = row["image_name"].strip()
                split = row["split"].strip()
                if split not in rows_by_split:
                    raise ValueError(f"Unsupported split in {split_csv}: {split}")
                rows_by_split[split].append({"image_name": image_name, "split": split})
    else:
        for image_path in sorted(images_dir.glob("*.tif")):
            split = "val" if image_path.name in INRIA_VAL_IMAGES else "train"
            rows_by_split[split].append({"image_name": image_path.name, "split": split})

    for split in rows_by_split:
        rows_by_split[split].sort(key=lambda item: item["image_name"])
        if limit > 0:
            rows_by_split[split] = rows_by_split[split][:limit]
        if not rows_by_split[split]:
            raise ValueError(f"No Inria images selected for split '{split}'.")
    return rows_by_split


def build_inria_hisup(
    inria_root: Path,
    output_root: Path,
    split_csv: Path | None,
    smoke_train: int,
    smoke_val: int,
    limit: int,
) -> Path:
    images_dir = inria_root / "train" / "images"
    geojson_dir = inria_root / "raw" / "train" / "gt_polygonized"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing Inria images directory: {images_dir}")
    if not geojson_dir.is_dir():
        raise FileNotFoundError(f"Missing Inria GeoJSON directory: {geojson_dir}")

    hisup_root = output_root / "hisup"
    rows_by_split = inria_split_rows(images_dir, split_csv, limit)
    split_rows_for_csv = []
    for split, rows in rows_by_split.items():
        for row in rows:
            match = INRIA_NAME_RE.match(row["image_name"])
            city = match.group("city") if match else ""
            number = int(match.group("number")) if match else ""
            split_rows_for_csv.append(
                {"city": city, "tile_number": number, "image_name": row["image_name"], "split": split}
            )
    write_split_csv(
        output_root / "image_split_by_city_80_20_hole_balanced.csv",
        sorted(split_rows_for_csv, key=lambda item: (str(item["city"]), str(item["tile_number"]))),
        ["city", "tile_number", "image_name", "split"],
    )

    for split, rows in rows_by_split.items():
        split_root = hisup_root / split
        images_out = split_root / "images"
        images_out.mkdir(parents=True, exist_ok=True)
        coco = {
            "info": {"dataset": INRIA_DATASET, "split": split, "patch_size": PATCH_SIZE},
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": [{"id": CATEGORY_ID, "name": "building", "supercategory": "building"}],
        }
        next_image_id = 1
        next_ann_id = 1
        for row in tqdm(rows, desc=f"inria:{split}", unit="tile"):
            image_name = row["image_name"]
            image_path = images_dir / image_name
            geojson_path = geojson_dir / f"{Path(image_name).stem}.geojson"
            if not image_path.is_file():
                raise FileNotFoundError(f"Missing image: {image_path}")
            if not geojson_path.is_file():
                raise FileNotFoundError(f"Missing GeoJSON: {geojson_path}")

            image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
            height, width = image.shape[:2]
            records = load_tile_polygons(geojson_path)
            geometries = [record["geometry"] for record in records]
            tree = STRtree(geometries) if geometries else None
            stem = Path(image_name).stem

            for x0, y0 in patch_origins(width, height):
                patch_box = box(x0, y0, x0 + PATCH_SIZE, y0 + PATCH_SIZE)
                patch_annotations: list[dict[str, Any]] = []
                if tree is not None:
                    for query in tree.query(patch_box):
                        record = records[int(query)] if isinstance(query, (int, np.integer)) else None
                        if record is None:
                            continue
                        clipped = record["geometry"].intersection(patch_box)
                        for part_index, part in enumerate(iter_polygonal(clipped)):
                            if not part.is_valid:
                                part = part.buffer(0)
                            if part.is_empty:
                                continue
                            local = translate(part, xoff=-x0, yoff=-y0)
                            if not isinstance(local, Polygon) or local.is_empty:
                                continue
                            segments = polygon_to_segmentation(local)
                            if not segments:
                                continue
                            min_x, min_y, max_x, max_y = local.bounds
                            bbox_w = float(max_x - min_x)
                            bbox_h = float(max_y - min_y)
                            if bbox_w <= 5.0 or bbox_h <= 5.0:
                                continue
                            patch_annotations.append(
                                {
                                    "image_id": next_image_id,
                                    "category_id": CATEGORY_ID,
                                    "segmentation": segments,
                                    "area": float(local.area),
                                    "bbox": [float(min_x), float(min_y), bbox_w, bbox_h],
                                    "iscrowd": 0,
                                    "has_hole": len(segments) > 1,
                                    "hole_count": max(0, len(segments) - 1),
                                    "source_image_name": image_name,
                                    "source_geojson": geojson_path.name,
                                    "source_geometry_index": record["geometry_index"],
                                    "source_part_index": record["part_index"],
                                    "clipped_part_index": part_index,
                                    "patch_origin": [x0, y0],
                                }
                            )
                if split == "train" and not patch_annotations:
                    continue

                patch_name = f"{stem}-x{x0:04d}-y{y0:04d}.tif"
                Image.fromarray(image[y0 : y0 + PATCH_SIZE, x0 : x0 + PATCH_SIZE]).save(images_out / patch_name)
                coco["images"].append(
                    {
                        "id": next_image_id,
                        "file_name": patch_name,
                        "width": PATCH_SIZE,
                        "height": PATCH_SIZE,
                        "source_image_name": image_name,
                        "patch_origin": [x0, y0],
                    }
                )
                patch_annotations.sort(key=lambda ann: (ann["bbox"][1], ann["bbox"][0]))
                for annotation in patch_annotations:
                    annotation["id"] = next_ann_id
                    coco["annotations"].append(annotation)
                    next_ann_id += 1
                next_image_id += 1
        write_json(split_root / "annotation.json", coco)
        write_json(split_root / "annotation-smoke.json", smoke_subset(coco, smoke_train if split == "train" else smoke_val))
    return hisup_root


def mirror_vector_root(source_root: Path, dest_root: Path) -> None:
    copy_tree_files(source_root, dest_root)


def crop_inria_mask(gt_dir: Path, image_meta: dict[str, Any]) -> np.ndarray:
    source_name = image_meta["source_image_name"]
    x0, y0 = image_meta["patch_origin"]
    mask_path = gt_dir / source_name
    if not mask_path.is_file():
        raise FileNotFoundError(f"Missing Inria mask: {mask_path}")
    mask = np.asarray(Image.open(mask_path), dtype=np.uint8)
    if mask.ndim == 3:
        mask = mask[..., 0]
    patch = mask[y0 : y0 + PATCH_SIZE, x0 : x0 + PATCH_SIZE]
    return (patch > 0).astype(np.uint8)


def build_seg_mirror_from_coco(
    canonical_root: Path,
    output_root: Path,
    mask_func,
    include_train: bool = True,
    include_val: bool = True,
) -> None:
    splits = []
    if include_train:
        splits.append("train")
    if include_val:
        splits.append("val")
    for split in splits:
        source_split = canonical_root / split
        coco = read_json(source_split / "annotation.json")
        split_root = output_root / split
        copy_tree_files(source_split / "images", split_root / "images")
        write_json(split_root / "annotation.json", coco)
        grouped = anns_by_image(coco)
        for image_meta in tqdm(coco["images"], desc=f"{output_root.name}:{split}:masks", unit="img"):
            mask = mask_func(image_meta, grouped.get(int(image_meta["id"]), []))
            save_binary_mask(mask, split_root / "masks" / f"{Path(image_meta['file_name']).stem}.png")


def generate_heatmap(vertices: np.ndarray, shape_hw: tuple[int, int], sigma: float) -> np.ndarray:
    heatmap = np.zeros(shape_hw, dtype=np.float32)
    if vertices.size == 0:
        return heatmap
    radius = max(1, int(math.ceil(3.0 * sigma)))
    coords = np.arange(-radius, radius + 1, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(coords, coords)
    kernel = np.exp(-((grid_x**2 + grid_y**2) / (2 * sigma**2))).astype(np.float32)
    height, width = shape_hw
    for vx, vy in vertices.astype(np.int32):
        left = max(0, int(vx) - radius)
        right = min(width, int(vx) + radius + 1)
        top = max(0, int(vy) - radius)
        bottom = min(height, int(vy) + radius + 1)
        kl = left - (int(vx) - radius)
        kr = kl + (right - left)
        kt = top - (int(vy) - radius)
        kb = kt + (bottom - top)
        heatmap[top:bottom, left:right] = np.maximum(heatmap[top:bottom, left:right], kernel[kt:kb, kl:kr])
    if heatmap.max() > 0:
        heatmap /= float(heatmap.max())
    return heatmap


def vertices_from_annotations(annotations: list[dict[str, Any]], width: int, height: int) -> np.ndarray:
    vertices = []
    for annotation in annotations:
        for segment in annotation.get("segmentation", []):
            ring = normalize_ring(segment)
            if ring.size == 0:
                continue
            ring[:, 0] = np.clip(ring[:, 0], 0, width - 1)
            ring[:, 1] = np.clip(ring[:, 1], 0, height - 1)
            vertices.append(np.rint(ring).astype(np.int32))
    if not vertices:
        return np.zeros((0, 2), dtype=np.int32)
    return np.concatenate(vertices, axis=0)


def build_acpv_from_coco(canonical_root: Path, output_root: Path, mask_func, sigma: float) -> None:
    for split in ("train", "val"):
        source_split = canonical_root / split
        coco = read_json(source_split / "annotation.json")
        split_root = output_root / split
        copy_tree_files(source_split / "images", split_root / "images")
        write_json(split_root / "annotation.json", coco)
        write_json(split_root / "annotation-smoke.json", smoke_subset(coco, 32 if split == "train" else 16))
        grouped = anns_by_image(coco)
        heatmap_dir = split_root / "vertex_heatmaps_sigma-3_augmented" / "rot0"
        heatmap_dir.mkdir(parents=True, exist_ok=True)
        for image_meta in tqdm(coco["images"], desc=f"acpvnet:{split}", unit="img"):
            image_id = int(image_meta["id"])
            stem = Path(image_meta["file_name"]).stem
            annotations = grouped.get(image_id, [])
            mask = mask_func(image_meta, annotations)
            save_binary_mask(mask, split_root / "masks" / f"{stem}.png")
            vertices = vertices_from_annotations(annotations, int(image_meta["width"]), int(image_meta["height"]))
            np.save(heatmap_dir / f"{stem}.npy", generate_heatmap(vertices, (int(image_meta["height"]), int(image_meta["width"])), sigma))


def encode_acpv_latents(
    release_root: Path,
    acpv_root: Path,
    config_path: Path | None,
    batch_size: int,
    num_workers: int,
    scale_samples: int,
) -> None:
    encoder = release_root / "models" / "acpvnet" / "source" / "latent_encoder_accelerated.py"
    if not encoder.is_file():
        raise FileNotFoundError(f"Missing ACPV latent encoder: {encoder}")
    if config_path is None or not config_path.is_file():
        raise FileNotFoundError(f"Missing ACPV autoencoder config: {config_path}")
    for heatmap_dir in acpv_root.glob("**/train/vertex_heatmaps_sigma-3_augmented/rot0"):
        latent_dir = heatmap_dir.parents[1] / "heatmap_augmented_latent_kl-4" / "rot0"
        command = [
            sys.executable,
            str(encoder),
            "--config",
            str(config_path),
            "--input",
            str(heatmap_dir),
            "--output_dir",
            str(latent_dir),
            "--type",
            "heatmap",
            "--batch-size",
            str(batch_size),
            "--num-workers",
            str(num_workers),
            "--scale-samples",
            str(scale_samples),
        ]
        subprocess.run(command, check=True, cwd=encoder.parent)


def build_holitracer_from_coco(
    canonical_root: Path,
    output_root: Path,
    mask_func,
    jpg_quality: int,
) -> None:
    for split in ("train", "val"):
        source_split = canonical_root / split
        source_images = source_split / "images"
        coco = read_json(source_split / "annotation.json")
        split_root = output_root / split
        img_dir = split_root / "img"
        img_tif_dir = split_root / "img_tif"
        mask_dir = split_root / "mask"
        img_dir.mkdir(parents=True, exist_ok=True)
        img_tif_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        images = []
        grouped = anns_by_image(coco)
        for image_meta in tqdm(coco["images"], desc=f"holitracer:{split}", unit="img"):
            source = source_images / image_meta["file_name"]
            stem = Path(image_meta["file_name"]).stem
            jpg_name = f"{stem}.jpg"
            Image.open(source).convert("RGB").save(img_dir / jpg_name, quality=jpg_quality)
            link_or_copy(source, img_tif_dir / image_meta["file_name"])
            mask = mask_func(image_meta, grouped.get(int(image_meta["id"]), []))
            save_binary_mask(mask, mask_dir / f"{stem}.png")
            copied = dict(image_meta)
            copied["file_name"] = jpg_name
            images.append(copied)
        annotations = [dict(annotation, category_id=1) for annotation in coco["annotations"]]
        holi = {
            "info": dict(coco.get("info", {}), adapter="holitracer"),
            "licenses": coco.get("licenses", []),
            "images": images,
            "annotations": annotations,
            "categories": [
                {
                    "id": 1,
                    "name": coco["categories"][0]["name"],
                    "supercategory": coco["categories"][0].get("supercategory", coco["categories"][0]["name"]),
                }
            ],
        }
        write_json(split_root / "coco_label_with_inter.json", holi)
        write_json(split_root / "coco_label_with_inter_smoke.json", smoke_subset(holi, 32 if split == "train" else 16))


def remove_redundant_vertices(points: np.ndarray, epsilon: float = 0.1) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if points.shape[0] < 3:
        return points
    keep = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1) > epsilon
    return points[keep]


def is_clockwise(points: list[list[float]]) -> bool:
    value = 0.0
    for p1, p2 in zip(points, points[1:] + points[:1]):
        value += (p2[0] - p1[0]) * (p2[1] + p1[1])
    return value > 0.0


def resort_corners(corners: np.ndarray) -> np.ndarray:
    corners = np.asarray(corners).reshape(-1, 2)
    start = np.argmin(corners[:, 0] ** 2 + corners[:, 1] ** 2)
    ordered = np.concatenate([corners[start:], corners[:start]])
    if not is_clockwise(ordered[:, :2].tolist()):
        ordered[1:] = np.flip(ordered[1:], axis=0)
    return ordered.reshape(-1)


def resort_corners_and_labels(corners: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    corners = np.asarray(corners).reshape(-1, 2)
    labels = np.asarray(labels).reshape(-1)
    start = np.argmin(corners[:, 0] ** 2 + corners[:, 1] ** 2)
    ordered = np.concatenate([corners[start:], corners[:start]])
    ordered_labels = np.concatenate([labels[start:], labels[:start]])
    if not is_clockwise(ordered[:, :2].tolist()):
        ordered[1:] = np.flip(ordered[1:], axis=0)
        ordered_labels[1:] = np.flip(ordered_labels[1:], axis=0)
    return ordered.reshape(-1), ordered_labels


def pad_polygon(contour: np.ndarray, num_corners: int) -> tuple[np.ndarray, np.ndarray]:
    if contour.shape[0] == 0:
        return np.zeros((num_corners, 2), dtype=np.int32), np.zeros((num_corners,), dtype=np.int32)
    if contour.shape[0] >= num_corners:
        return contour[:num_corners], np.ones((num_corners,), dtype=np.int32)
    pad = np.tile(contour[-1], (num_corners - contour.shape[0], 1))
    labels = np.zeros((num_corners,), dtype=np.int32)
    labels[: contour.shape[0]] = 1
    return np.vstack([contour, pad]), labels


def uniform_sampling_index(points: np.ndarray, num_corners: int, image_size: int) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    polygon = np.rint(points).astype(np.int32).reshape(-1, 1, 2)
    canvas = np.zeros((image_size, image_size), dtype=np.uint8)
    cv2.polylines(canvas, [polygon], True, 255, 1)
    cv2.fillPoly(canvas, [polygon], 255)
    contours, _ = cv2.findContours(canvas, cv2.RETR_LIST, method=cv2.CHAIN_APPROX_NONE)
    if not contours:
        return pad_polygon(np.zeros((0, 2), dtype=np.int32), num_corners)
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
    if contour.shape[0] < num_corners:
        sampled, labels = pad_polygon(contour, num_corners)
        return sampled.reshape(-1), labels
    indices = np.linspace(0, contour.shape[0], num=num_corners, endpoint=False).round().astype(np.int32)
    indices = np.clip(indices, 0, contour.shape[0] - 1)
    sampled = contour[indices]
    labels = np.zeros((num_corners,), dtype=np.int32)
    polygon_points = polygon.reshape(-1, 2)
    for point in polygon_points:
        distances = np.linalg.norm(sampled - point, axis=1)
        idx = int(np.argmin(distances))
        sampled[idx] = point
        labels[idx] = 1
    return sampled.reshape(-1), labels


def roipoly_bbox(points: np.ndarray, image_size: int) -> list[float]:
    points = points.reshape(-1, 2)
    min_x = float(points[:, 0].min())
    min_y = float(points[:, 1].min())
    max_x = float(points[:, 0].max())
    max_y = float(points[:, 1].max())
    width = max_x - min_x
    height = max_y - min_y
    min_x = max(min_x - width * 0.1, 0.0)
    min_y = max(min_y - height * 0.1, 0.0)
    max_x = min(max_x + width * 0.1, image_size - 1e-4)
    max_y = min(max_y + height * 0.1, image_size - 1e-4)
    return [min_x, min_y, max_x - min_x, max_y - min_y]


def build_roipoly(canonical_root: Path, output_root: Path, num_corners: int, smoke_train: int, smoke_val: int) -> None:
    require_shapely()
    from skimage.measure import approximate_polygon

    for split in ("train", "val"):
        source_split = canonical_root / split
        coco = read_json(source_split / "annotation.json")
        split_root = output_root / split
        copy_tree_files(source_split / "images", split_root / "images")
        raw = {
            "info": dict(coco.get("info", {}), ring_split=True),
            "licenses": coco.get("licenses", []),
            "images": coco["images"],
            "annotations": [],
            "categories": [{"id": CATEGORY_ID, "name": "polygon", "supercategory": "polygon"}],
        }
        next_id = 1
        for annotation in coco["annotations"]:
            for ring_index, segment in enumerate(annotation.get("segmentation", [])):
                ring = normalize_ring(segment)
                if ring.size == 0:
                    continue
                raw["annotations"].append(
                    {
                        "id": next_id,
                        "image_id": int(annotation["image_id"]),
                        "category_id": CATEGORY_ID,
                        "segmentation": [ring.reshape(-1).astype(float).tolist()],
                        "area": float(annotation.get("area", 0.0)),
                        "bbox": polygon_bbox_from_ring(ring),
                        "iscrowd": 0,
                        "is_hole": bool(ring_index > 0),
                        "ring_role": "hole" if ring_index > 0 else "exterior",
                    }
                )
                next_id += 1
        write_json(split_root / "annotation_raw.json", raw)
        write_json(split_root / "annotation_raw_smoke.json", smoke_subset(raw, smoke_train if split == "train" else smoke_val))

        processed = dict(raw)
        processed["annotations"] = []
        next_id = 1
        for annotation in raw["annotations"]:
            ring = normalize_ring(annotation["segmentation"][0])
            if ring.shape[0] < 3:
                continue
            ring = np.clip(ring, 0, PATCH_SIZE - 1)
            ring = remove_redundant_vertices(ring)
            if ring.shape[0] < 3 or Polygon(ring).area < 4:
                continue
            ring = approximate_polygon(ring, tolerance=0.01)
            if ring.shape[0] < 3 or Polygon(ring).area < 4:
                continue
            copied = dict(annotation)
            copied["id"] = next_id
            if split == "train":
                sampled, labels = uniform_sampling_index(ring, num_corners, PATCH_SIZE)
                sampled, labels = resort_corners_and_labels(sampled, labels)
                copied["segmentation"] = [[int(round(value)) for value in sampled.tolist()]]
                copied["cor_cls_poly"] = [int(value) for value in labels.tolist()]
            else:
                copied["segmentation"] = [resort_corners(ring).astype(float).tolist()]
            copied["bbox"] = roipoly_bbox(ring, PATCH_SIZE)
            processed["annotations"].append(copied)
            next_id += 1
        write_json(split_root / "annotation_roipoly.json", processed)
        write_json(split_root / "annotation_roipoly_smoke.json", smoke_subset(processed, smoke_train if split == "train" else smoke_val))


def geojson_for_annotations(annotations: list[dict[str, Any]]) -> dict[str, Any]:
    geometries = []
    for annotation in annotations:
        rings = []
        for segment in annotation.get("segmentation", []):
            closed = closed_ring_flat(segment)
            coords = [[closed[i], closed[i + 1]] for i in range(0, len(closed), 2)]
            if len(coords) >= 4:
                rings.append(coords)
        if rings:
            geometries.append({"type": "Polygon", "coordinates": rings})
    return {"type": "GeometryCollection", "geometries": geometries}


def build_inria_ffl(inria_root: Path, output_root: Path, split_csv_rows: list[dict[str, Any]]) -> None:
    ffl_root = output_root / "ffl" / "inria_building"
    raw_train = ffl_root / "raw" / "train"
    images_dir = raw_train / "images"
    geojson_dir = raw_train / "gt_polygonized"
    images_dir.mkdir(parents=True, exist_ok=True)
    geojson_dir.mkdir(parents=True, exist_ok=True)
    for image_path in sorted((inria_root / "train" / "images").glob("*.tif")):
        link_or_copy(image_path, images_dir / image_path.name)
        geojson_path = inria_root / "raw" / "train" / "gt_polygonized" / f"{image_path.stem}.geojson"
        if geojson_path.is_file():
            link_or_copy(geojson_path, geojson_dir / geojson_path.name)
    rows = [{"image_name": row["image_name"], "split": row["split"]} for row in split_csv_rows]
    write_split_csv(ffl_root / "train" / "image_split_official_train_val.csv", rows, ["image_name", "split"])
    smoke = [row for row in rows if row["split"] == "train"][:8] + [row for row in rows if row["split"] == "val"][:4]
    write_split_csv(ffl_root / "train" / "image_split_smoke_train_val.csv", smoke, ["image_name", "split"])


def load_deventer_split(root: Path, split: str, category: str) -> dict[str, Any]:
    path = root / split / "annotations" / f"{category}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing Deventer annotation: {path}")
    return read_json(path)


def build_deventer_coco(root: Path, split: str, category: str, limit: int = 0) -> dict[str, Any]:
    payload = load_deventer_split(root, split, category)
    images = sorted(
        [
            {
                "id": int(image["id"]),
                "file_name": str(image["file_name"]),
                "width": int(image["width"]),
                "height": int(image["height"]),
            }
            for image in payload.get("images", [])
        ],
        key=lambda image: str(image["file_name"]),
    )
    if limit > 0:
        images = images[:limit]
    selected_old_ids = {int(image["id"]) for image in images}
    file_name_by_old_id = {int(image["id"]): str(image["file_name"]) for image in payload.get("images", [])}
    id_by_file = {image["file_name"]: int(image["id"]) for image in images}
    annotations = []
    next_id = 1
    for ann in payload.get("annotations", []):
        if int(ann["image_id"]) not in selected_old_ids:
            continue
        file_name = file_name_by_old_id[int(ann["image_id"])]
        segmentation = sanitize_segmentation(ann.get("segmentation", []))
        if not segmentation:
            continue
        annotations.append(
            {
                "id": next_id,
                "image_id": id_by_file[file_name],
                "category_id": CATEGORY_ID,
                "segmentation": segmentation,
                "area": float(ann.get("area", 0.0)),
                "bbox": [float(value) for value in ann.get("bbox", [0, 0, 0, 0])],
                "iscrowd": int(ann.get("iscrowd", 0)),
                "source_category_name": category,
                "source_annotation_id": int(ann["id"]),
            }
        )
        next_id += 1
    return category_coco(category, images, annotations)


def build_deventer_hisup(root: Path, output_root: Path, category: str, smoke_train: int, smoke_val: int, limit: int) -> Path:
    hisup_root = output_root / "hisup" / category
    for split in ("train", "val"):
        coco = build_deventer_coco(root, split, category, limit)
        split_root = hisup_root / split
        copy_deventer_images(root, split, split_root / "images", coco["images"])
        write_json(split_root / "annotation.json", coco)
        write_json(split_root / "annotation-smoke.json", smoke_subset(coco, smoke_train if split == "train" else smoke_val))
    return hisup_root


def copy_deventer_images(root: Path, split: str, dest: Path, images: list[dict[str, Any]]) -> None:
    for image in images:
        source = root / split / "images" / image["file_name"]
        if not source.is_file():
            raise FileNotFoundError(f"Missing Deventer image: {source}")
        link_or_copy(source, dest / image["file_name"])


def deventer_mask_func(root: Path, split: str, category: str):
    mask_value = DEVENTER_MASK_VALUE[category]

    def _mask(image_meta: dict[str, Any], _annotations: list[dict[str, Any]]) -> np.ndarray:
        mask_path = root / split / "masks" / image_meta["file_name"]
        if not mask_path.is_file():
            raise FileNotFoundError(f"Missing Deventer mask: {mask_path}")
        raw = np.asarray(Image.open(mask_path), dtype=np.uint8)
        if raw.ndim == 3:
            raw = raw[..., 0]
        return (raw == mask_value).astype(np.uint8)

    return _mask


def build_deventer_seg_mirror(root: Path, hisup_root: Path, output_root: Path, category: str, include_train: bool = True) -> None:
    for split in (("train", "val") if include_train else ("val",)):
        source_split = hisup_root / split
        coco = read_json(source_split / "annotation.json")
        split_root = output_root / split
        copy_tree_files(source_split / "images", split_root / "images")
        write_json(split_root / "annotation.json", coco)
        mask_func = deventer_mask_func(root, split, category)
        for image_meta in tqdm(coco["images"], desc=f"{output_root.name}:{category}:{split}:masks", unit="img"):
            save_binary_mask(mask_func(image_meta, []), split_root / "masks" / f"{Path(image_meta['file_name']).stem}.png")


def build_deventer_acpv(root: Path, hisup_root: Path, output_root: Path, category: str, sigma: float) -> None:
    for split in ("train", "val"):
        source_split = hisup_root / split
        coco = read_json(source_split / "annotation.json")
        split_root = output_root / split
        copy_tree_files(source_split / "images", split_root / "images")
        write_json(split_root / "annotation.json", coco)
        write_json(split_root / "annotation-smoke.json", smoke_subset(coco, 32 if split == "train" else 16))
        grouped = anns_by_image(coco)
        mask_func = deventer_mask_func(root, split, category)
        heatmap_dir = split_root / "vertex_heatmaps_sigma-3_augmented" / "rot0"
        heatmap_dir.mkdir(parents=True, exist_ok=True)
        for image_meta in tqdm(coco["images"], desc=f"acpvnet:{category}:{split}", unit="img"):
            stem = Path(image_meta["file_name"]).stem
            save_binary_mask(mask_func(image_meta, []), split_root / "masks" / f"{stem}.png")
            annotations = grouped.get(int(image_meta["id"]), [])
            vertices = vertices_from_annotations(annotations, int(image_meta["width"]), int(image_meta["height"]))
            np.save(heatmap_dir / f"{stem}.npy", generate_heatmap(vertices, (int(image_meta["height"]), int(image_meta["width"])), sigma))


def build_deventer_ffl(root: Path, output_root: Path, category: str, limit: int) -> None:
    ffl_root = output_root / "ffl" / category
    raw_train = ffl_root / "raw" / "train"
    images_out = raw_train / "images"
    geojson_out = raw_train / "gt_polygonized"
    images_out.mkdir(parents=True, exist_ok=True)
    geojson_out.mkdir(parents=True, exist_ok=True)
    rows = []
    next_image_id = 1
    for source_split in ("train", "val"):
        coco = build_deventer_coco(root, source_split, category, limit)
        grouped = anns_by_image(coco)
        for image in coco["images"]:
            source_file_name = image["file_name"]
            output_file_name = f"{source_split}_{source_file_name}"
            output_stem = Path(output_file_name).stem
            link_or_copy(root / source_split / "images" / source_file_name, images_out / output_file_name)
            write_json(geojson_out / f"{output_stem}.geojson", geojson_for_annotations(grouped.get(int(image["id"]), [])))
            rows.append(
                {
                    "image_name": output_file_name,
                    "split": source_split,
                    "image_id": next_image_id,
                    "source_split": source_split,
                    "source_image_name": source_file_name,
                    "source_image_id": int(image["id"]),
                }
            )
            next_image_id += 1
    rows.sort(key=lambda row: (row["split"], row["image_name"]))
    fields = ["image_name", "split", "image_id", "source_split", "source_image_name", "source_image_id"]
    write_split_csv(ffl_root / "train" / "image_split_official_train_val.csv", rows, fields)
    smoke = [row for row in rows if row["split"] == "train"][:8] + [row for row in rows if row["split"] == "val"][:4]
    write_split_csv(ffl_root / "train" / "image_split_smoke_train_val.csv", smoke, fields)


def build_release_inria(args: argparse.Namespace, release_root: Path, methods: set[str]) -> None:
    require_shapely()
    dataset_root = output_root_from_args(args, release_root) / INRIA_DATASET
    prepare_root(dataset_root, args.overwrite)
    inria_root = inria_root_from_args(args, release_root)
    split_csv = resolve_path(args.inria_split_csv) if args.inria_split_csv else None
    hisup_root = build_inria_hisup(
        inria_root,
        dataset_root,
        split_csv,
        args.smoke_train_images,
        args.smoke_val_images,
        args.limit_source_images,
    )
    split_rows = []
    with (dataset_root / "image_split_by_city_80_20_hole_balanced.csv").open(newline="", encoding="utf-8") as handle:
        split_rows = list(csv.DictReader(handle))

    for mirror in VECTOR_MIRRORS:
        if mirror in {"gcp", "pix2poly", "polyworld"} and (
            mirror in methods or (mirror == "gcp" and "gcp" in methods) or (mirror == "polyworld" and "polyworld" in methods)
        ):
            mirror_vector_root(hisup_root, dataset_root / mirror)

    gt_dir = inria_root / "train" / "gt"
    inria_mask = lambda image_meta, annotations: crop_inria_mask(gt_dir, image_meta)
    if "unet_poly" in methods:
        build_seg_mirror_from_coco(hisup_root, dataset_root / "unet_seg", inria_mask)
    if "maskrcnn_poly" in methods:
        build_seg_mirror_from_coco(hisup_root, dataset_root / "maskrcnn_seg", inria_mask)
    if "sam2_poly" in methods:
        build_seg_mirror_from_coco(hisup_root, dataset_root / "sam2_seg", inria_mask, include_train=False, include_val=True)
    if "acpvnet" in methods:
        build_acpv_from_coco(hisup_root, dataset_root / "acpvnet", inria_mask, args.acpv_sigma)
        if args.encode_acpv_latents:
            encode_acpv_latents(
                release_root,
                dataset_root / "acpvnet",
                acpv_config_path(args, release_root),
                args.acpv_latent_batch_size,
                args.acpv_latent_num_workers,
                args.acpv_latent_scale_samples,
            )
    if "holitracer" in methods:
        build_holitracer_from_coco(hisup_root, dataset_root / "holitracer", inria_mask, args.jpg_quality)
    if "roipoly" in methods:
        build_roipoly(hisup_root, dataset_root / "roipoly", args.roipoly_num_corners, args.smoke_train_images, args.smoke_val_images)
    if "ffl" in methods:
        build_inria_ffl(inria_root, dataset_root, split_rows)


def build_release_deventer(args: argparse.Namespace, release_root: Path, methods: set[str]) -> None:
    dataset_root = output_root_from_args(args, release_root) / DEVENTER_DATASET
    prepare_root(dataset_root, args.overwrite)
    raw_root = deventer_root_from_args(args, release_root)
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Missing Deventer root: {raw_root}")
    if not (raw_root / "train").is_dir() or not (raw_root / "val").is_dir():
        raise FileNotFoundError(f"Deventer root must contain train/ and val/: {raw_root}")

    for task in args.deventer_tasks:
        hisup_root = build_deventer_hisup(
            raw_root,
            dataset_root,
            task,
            args.smoke_train_images,
            args.smoke_val_images,
            args.limit_source_images,
        )
        for mirror in VECTOR_MIRRORS:
            if mirror in methods:
                mirror_vector_root(hisup_root, dataset_root / mirror / task)
        if "unet_poly" in methods:
            build_deventer_seg_mirror(raw_root, hisup_root, dataset_root / "unet_seg" / task, task)
        if "maskrcnn_poly" in methods:
            build_deventer_seg_mirror(raw_root, hisup_root, dataset_root / "maskrcnn_seg" / task, task)
        if "sam2_poly" in methods:
            mirror_vector_root(hisup_root / "val", dataset_root / "sam2_seg" / task / "val")
        if "acpvnet" in methods:
            build_deventer_acpv(raw_root, hisup_root, dataset_root / "acpvnet" / task, task, args.acpv_sigma)
        if "holitracer" in methods:
            mask_by_split = {}
            for split in ("train", "val"):
                mask_by_split[split] = deventer_mask_func(raw_root, split, task)

            def mask_func(image_meta: dict[str, Any], annotations: list[dict[str, Any]]) -> np.ndarray:
                split = "train" if (hisup_root / "train" / "images" / image_meta["file_name"]).exists() else "val"
                return mask_by_split[split](image_meta, annotations)

            build_holitracer_from_coco(hisup_root, dataset_root / "holitracer" / task, mask_func, args.jpg_quality)
        if "roipoly" in methods:
            build_roipoly(hisup_root, dataset_root / "roipoly" / task, args.roipoly_num_corners, args.smoke_train_images, args.smoke_val_images)
        if "ffl" in methods:
            build_deventer_ffl(raw_root, dataset_root, task, args.limit_source_images)
    if "acpvnet" in methods and args.encode_acpv_latents:
        encode_acpv_latents(
            release_root,
            dataset_root / "acpvnet",
            acpv_config_path(args, release_root),
            args.acpv_latent_batch_size,
            args.acpv_latent_num_workers,
            args.acpv_latent_scale_samples,
        )


def main() -> None:
    args = parse_args()
    release_root = Path(__file__).resolve().parent
    datasets = selected_datasets(args.datasets)
    methods = selected_methods(args.methods)
    validate_acpv_options(args, release_root, methods)
    if INRIA_DATASET in datasets:
        build_release_inria(args, release_root, methods)
    if DEVENTER_DATASET in datasets:
        build_release_deventer(args, release_root, methods)
    print(f"Prepared data under {output_root_from_args(args, release_root)}")


if __name__ == "__main__":
    main()
