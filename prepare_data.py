#!/usr/bin/env python3
"""Build baseline-specific data mirrors from the PolyTopoBench release.

Input is the release downloaded from https://huggingface.co/datasets/PingL/PolyTopoBench
(``tasks.json``, ``inria/``, ``deventer/`` and, for FFL on Inria only, ``raw/inria/``).
Every mirror is derived from the released task annotations; binary masks are rendered
from the released polygons.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

try:
    from shapely.geometry import Polygon
except ModuleNotFoundError:
    Polygon = None


INRIA_DATASET = "inria_building"
DEVENTER_DATASET = "deventer_512_valtest_as_val"
HF_REPO = "PingL/PolyTopoBench"
PATCH_SIZE = 512
CATEGORY_ID = 100
SPLITS = ("train", "val")
DEVENTER_TASKS = ("road", "vegetation", "unvegetated")
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
MIRROR_DIRS = {
    "unet_poly": "unet_seg",
    "maskrcnn_poly": "maskrcnn_seg",
    "sam2_poly": "sam2_seg",
    "hisup": "hisup",
    "acpvnet": "acpvnet",
    "ffl": "ffl",
    "gcp": "gcp",
    "holitracer": "holitracer",
    "pix2poly": "pix2poly",
    "polyworld": "polyworld",
    "roipoly": "roipoly",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare baseline-ready PolyTopoBench data from the released dataset.")
    parser.add_argument("--data-root", type=Path, default=None, help="Downloaded release (contains tasks.json). Default: dataset/")
    parser.add_argument("--output-root", type=Path, default=None, help="Default: dataset/data_processed/")
    parser.add_argument("--datasets", nargs="+", default=["all"], choices=["all", INRIA_DATASET, DEVENTER_DATASET])
    parser.add_argument("--methods", nargs="+", default=None, choices=["all", *METHODS],
                        help="Baselines to prepare. Default: all except acpvnet.")
    parser.add_argument("--deventer-tasks", nargs="+", default=list(DEVENTER_TASKS), choices=list(DEVENTER_TASKS))
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
    parser.add_argument("--skip-checksums", action="store_true", help="Do not verify annotation checksums from tasks.json.")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild mirrors that already exist.")
    return parser.parse_args()


def selected_datasets(values: list[str]) -> set[str]:
    return {INRIA_DATASET, DEVENTER_DATASET} if values == ["all"] else set(values)


def selected_methods(values: list[str] | None) -> set[str]:
    if values is None:
        return set(DEFAULT_METHODS)
    return set(METHODS) if values == ["all"] else set(values)


def require_shapely() -> None:
    if Polygon is None:
        raise ModuleNotFoundError("prepare_data.py requires shapely. Install the release requirements first.")


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (Path.cwd() / path)


def data_root_from_args(args: argparse.Namespace, release_root: Path) -> Path:
    return release_root / "dataset" if args.data_root is None else resolve_path(args.data_root)


def output_root_from_args(args: argparse.Namespace, release_root: Path) -> Path:
    return release_root / "dataset" / "data_processed" if args.output_root is None else resolve_path(args.output_root)


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
    checkpoint = release_root / "models" / "acpvnet" / "source" / "models" / "first_stage_models" / "kl-f4" / "model.ckpt"
    if args.acpv_autoencoder_config is None and not checkpoint.is_file():
        raise SystemExit(
            f"Missing the LDM kl-f4 autoencoder weights at {checkpoint}. Download them with\n"
            f"  curl -L -o kl-f4.zip https://ommer-lab.com/files/latent-diffusion/kl-f4.zip && unzip kl-f4.zip -d {checkpoint.parent}"
        )


def release_tasks(datasets: set[str], deventer_tasks: list[str]) -> list[str]:
    tasks = []
    if INRIA_DATASET in datasets:
        tasks.append("inria/building")
    if DEVENTER_DATASET in datasets:
        tasks.extend(f"deventer/{task}" for task in deventer_tasks)
    return tasks


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_release(data_root: Path, tasks: list[str], verify_checksums: bool) -> dict[str, Any]:
    manifest_path = data_root / "tasks.json"
    if not manifest_path.is_file():
        raise SystemExit(
            f"Missing {manifest_path}. Download the release first, e.g.\n"
            f"  hf download {HF_REPO} --repo-type dataset --local-dir {data_root} "
            f"--include 'tasks.json' 'inria/*' 'deventer/*'"
        )
    manifest = read_json(manifest_path)["tasks"]
    for task in tasks:
        for split, info in manifest[task]["splits"].items():
            annotations = data_root / info["annotations"]
            image_dir = data_root / info["image_dir"]
            if not annotations.is_file() or not image_dir.is_dir():
                raise SystemExit(f"Incomplete download: missing {annotations if not annotations.is_file() else image_dir}")
            if verify_checksums and sha256(annotations) != info["annotations_sha256"]:
                raise SystemExit(f"Checksum mismatch for {annotations}; download it again.")
            found = sum(1 for path in image_dir.rglob("*") if path.is_file())
            if found < info["images"]:
                raise SystemExit(f"Incomplete download: {image_dir} has {found} of {info['images']} images.")
    return manifest


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")


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


def build_mirror(path: Path, overwrite: bool, builder: Callable[[Path], None]) -> None:
    """Run ``builder`` into a staging folder and move it to ``path`` once it succeeds."""
    if path.exists() and not overwrite:
        print(f"Skipping {path} (already prepared; pass --overwrite to rebuild)", flush=True)
        return
    staging = path.parent / f".{path.name}.partial"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    builder(staging)
    if path.exists():
        shutil.rmtree(path)
    staging.rename(path)


def polygon_mask(image_meta: dict[str, Any], annotations: list[dict[str, Any]]) -> np.ndarray:
    return render_mask(int(image_meta["width"]), int(image_meta["height"]), annotations)


def build_canonical(data_root: Path, info: dict[str, Any], output_root: Path, smoke_train: int, smoke_val: int) -> None:
    """Mirror released task annotations into the flat hisup layout that every baseline reads."""
    for split, split_info in info["splits"].items():
        coco = read_json(data_root / split_info["annotations"])
        image_dir = data_root / split_info["image_dir"]
        split_root = output_root / split
        for image in tqdm(coco["images"], desc=f"hisup:{split}", unit="img"):
            source = image_dir / image["file_name"]
            image["file_name"] = Path(image["file_name"]).name
            link_or_copy(source, split_root / "images" / image["file_name"])
        write_json(split_root / "annotation.json", coco)
        write_json(split_root / "annotation-smoke.json", smoke_subset(coco, smoke_train if split == "train" else smoke_val))


def mirror_vector_root(source_root: Path, dest_root: Path) -> None:
    copy_tree_files(source_root, dest_root)


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


def build_inria_ffl(raw_root: Path, ffl_mirror: Path, split_rows: list[dict[str, Any]]) -> None:
    ffl_root = ffl_mirror / "inria_building"
    raw_train = ffl_root / "raw" / "train"
    images_dir = raw_train / "images"
    geojson_dir = raw_train / "gt_polygonized"
    images_dir.mkdir(parents=True, exist_ok=True)
    geojson_dir.mkdir(parents=True, exist_ok=True)
    for row in split_rows:
        link_or_copy(raw_root / "train" / "images" / row["image_name"], images_dir / row["image_name"])
        geojson_path = raw_root / "raw" / "train" / "gt_polygonized" / f"{Path(row['image_name']).stem}.geojson"
        link_or_copy(geojson_path, geojson_dir / geojson_path.name)
    rows = [{"image_name": row["image_name"], "split": row["split"]} for row in split_rows]
    write_split_csv(ffl_root / "train" / "image_split_official_train_val.csv", rows, ["image_name", "split"])
    smoke = [row for row in rows if row["split"] == "train"][:8] + [row for row in rows if row["split"] == "val"][:4]
    write_split_csv(ffl_root / "train" / "image_split_smoke_train_val.csv", smoke, ["image_name", "split"])


def build_deventer_ffl(canonical_root: Path, ffl_root: Path) -> None:
    raw_train = ffl_root / "raw" / "train"
    images_out = raw_train / "images"
    geojson_out = raw_train / "gt_polygonized"
    images_out.mkdir(parents=True, exist_ok=True)
    geojson_out.mkdir(parents=True, exist_ok=True)
    rows = []
    next_image_id = 1
    for source_split in SPLITS:
        coco = read_json(canonical_root / source_split / "annotation.json")
        grouped = anns_by_image(coco)
        for image in coco["images"]:
            source_file_name = image["file_name"]
            output_file_name = f"{source_split}_{source_file_name}"
            output_stem = Path(output_file_name).stem
            link_or_copy(canonical_root / source_split / "images" / source_file_name, images_out / output_file_name)
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


def inria_ffl_split_rows(data_root: Path) -> list[dict[str, Any]]:
    splits: dict[str, str] = {}
    with (data_root / "inria" / "splits.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            splits.setdefault(row["source_tile"], row["split"])
    return [{"image_name": name, "split": split} for name, split in sorted(splits.items())]


def build_method_mirrors(
    args: argparse.Namespace,
    release_root: Path,
    methods: set[str],
    canonical_root: Path,
    mirror_path: Callable[[str], Path],
) -> None:
    """Build every selected mirror (except FFL) from one canonical hisup root."""
    builders: dict[str, Callable[[Path], None]] = {
        "gcp": lambda out: mirror_vector_root(canonical_root, out),
        "pix2poly": lambda out: mirror_vector_root(canonical_root, out),
        "polyworld": lambda out: mirror_vector_root(canonical_root, out),
        "unet_poly": lambda out: build_seg_mirror_from_coco(canonical_root, out, polygon_mask),
        "maskrcnn_poly": lambda out: build_seg_mirror_from_coco(canonical_root, out, polygon_mask),
        "sam2_poly": lambda out: build_seg_mirror_from_coco(canonical_root, out, polygon_mask, include_train=False),
        "holitracer": lambda out: build_holitracer_from_coco(canonical_root, out, polygon_mask, args.jpg_quality),
        "roipoly": lambda out: build_roipoly(canonical_root, out, args.roipoly_num_corners, args.smoke_train_images, args.smoke_val_images),
    }

    def build_acpv(out: Path) -> None:
        build_acpv_from_coco(canonical_root, out, polygon_mask, args.acpv_sigma)
        if args.encode_acpv_latents:
            encode_acpv_latents(
                release_root,
                out,
                acpv_config_path(args, release_root),
                args.acpv_latent_batch_size,
                args.acpv_latent_num_workers,
                args.acpv_latent_scale_samples,
            )

    builders["acpvnet"] = build_acpv
    for method in METHODS:
        if method in methods and method in builders:
            build_mirror(mirror_path(method), args.overwrite, builders[method])


def prepare_inria(args: argparse.Namespace, release_root: Path, methods: set[str], manifest: dict[str, Any]) -> None:
    data_root = data_root_from_args(args, release_root)
    dataset_root = output_root_from_args(args, release_root) / INRIA_DATASET
    canonical_root = dataset_root / "hisup"
    build_mirror(
        canonical_root,
        args.overwrite,
        lambda out: build_canonical(data_root, manifest["inria/building"], out, args.smoke_train_images, args.smoke_val_images),
    )
    build_method_mirrors(args, release_root, methods, canonical_root, lambda method: dataset_root / MIRROR_DIRS[method])
    if "ffl" in methods:
        raw_root = data_root / "raw" / "inria"
        split_rows = inria_ffl_split_rows(data_root)
        missing = [row["image_name"] for row in split_rows if not (raw_root / "train" / "images" / row["image_name"]).is_file()]
        if missing:
            message = (
                f"FFL on Inria needs the full tiles under {raw_root} ({len(missing)} missing). Download them with\n"
                f"  hf download {HF_REPO} --repo-type dataset --local-dir {data_root} --include 'raw/inria/*'"
            )
            if args.methods is None:
                print(f"Skipping ffl for {INRIA_DATASET}: {message}", flush=True)
                return
            raise SystemExit(message)
        build_mirror(dataset_root / "ffl", args.overwrite, lambda out: build_inria_ffl(raw_root, out, split_rows))


def prepare_deventer(args: argparse.Namespace, release_root: Path, methods: set[str], manifest: dict[str, Any]) -> None:
    data_root = data_root_from_args(args, release_root)
    dataset_root = output_root_from_args(args, release_root) / DEVENTER_DATASET
    for task in args.deventer_tasks:
        canonical_root = dataset_root / "hisup" / task
        build_mirror(
            canonical_root,
            args.overwrite,
            lambda out: build_canonical(data_root, manifest[f"deventer/{task}"], out, args.smoke_train_images, args.smoke_val_images),
        )
        build_method_mirrors(args, release_root, methods, canonical_root, lambda method: dataset_root / MIRROR_DIRS[method] / task)
        if "ffl" in methods:
            build_mirror(dataset_root / "ffl" / task, args.overwrite, lambda out: build_deventer_ffl(canonical_root, out))


def main() -> None:
    args = parse_args()
    release_root = Path(__file__).resolve().parent
    datasets = selected_datasets(args.datasets)
    methods = selected_methods(args.methods)
    validate_acpv_options(args, release_root, methods)
    if "roipoly" in methods:
        require_shapely()
    data_root = data_root_from_args(args, release_root)
    manifest = check_release(data_root, release_tasks(datasets, args.deventer_tasks), not args.skip_checksums)
    if INRIA_DATASET in datasets:
        prepare_inria(args, release_root, methods, manifest)
    if DEVENTER_DATASET in datasets:
        prepare_deventer(args, release_root, methods, manifest)
    print(f"Prepared data under {output_root_from_args(args, release_root)}")


if __name__ == "__main__":
    main()
