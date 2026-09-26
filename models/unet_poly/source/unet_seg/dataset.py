#!/usr/bin/env python3
"""Dataset classes for the unet_seg mirror.

Reads `(image, merged binary mask)` pairs from:
    <mirror>/<split>/images/*.{tif,png}
    <mirror>/<split>/masks/<stem>.png   (produced by prepare_data.py)
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

Image.MAX_IMAGE_PIXELS = None


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _load_image_rgb(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def _load_binary_mask(path: Path) -> np.ndarray:
    m = np.asarray(Image.open(path), dtype=np.uint8)
    if m.ndim == 3:
        m = m[..., 0]
    return (m > 0).astype(np.uint8)


def _normalize(img: np.ndarray) -> np.ndarray:
    arr = img.astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN[None, None, :]) / IMAGENET_STD[None, None, :]
    return arr


def _random_augment(img: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if random.random() < 0.5:
        img = np.ascontiguousarray(img[:, ::-1])
        mask = np.ascontiguousarray(mask[:, ::-1])
    if random.random() < 0.5:
        img = np.ascontiguousarray(img[::-1, :])
        mask = np.ascontiguousarray(mask[::-1, :])
    k = random.randint(0, 3)
    if k:
        img = np.ascontiguousarray(np.rot90(img, k=k))
        mask = np.ascontiguousarray(np.rot90(mask, k=k))
    return img, mask


class UnetSegImageMaskDataset(Dataset):
    """Loads (image, binary mask) pairs from a split; optional aug."""

    def __init__(self, mirror_root: Path, split: str, augment: bool = False):
        self.mirror_root = Path(mirror_root)
        self.split_dir = self.mirror_root / split
        ann = self.split_dir / "annotation.json"
        with ann.open("r") as f:
            coco = json.load(f)
        self.records = []
        for img in coco["images"]:
            fn = img["file_name"]
            mask_path = self.split_dir / "masks" / (Path(fn).stem + ".png")
            img_path = self.split_dir / "images" / fn
            if mask_path.exists() and img_path.exists():
                self.records.append((img_path, mask_path))
        if not self.records:
            raise RuntimeError(f"No records in {self.split_dir}")
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.records[idx]
        img = _load_image_rgb(img_path)
        mask = _load_binary_mask(mask_path)
        if self.augment:
            img, mask = _random_augment(img, mask)
        img_norm = _normalize(img).transpose(2, 0, 1)
        return {
            "image": torch.from_numpy(img_norm.copy()).float(),
            "mask": torch.from_numpy(mask.astype(np.int64).copy()).long(),
        }


class UnetSegValDataset(Dataset):
    """Inference-only loader that exposes image_id and file_name."""

    def __init__(self, mirror_root: Path):
        self.mirror_root = Path(mirror_root)
        self.split_dir = self.mirror_root / "val"
        ann = self.split_dir / "annotation.json"
        with ann.open("r") as f:
            coco = json.load(f)
        self.images = coco["images"]
        self.images_dir = self.split_dir / "images"

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        meta = self.images[idx]
        img_path = self.images_dir / meta["file_name"]
        img = _load_image_rgb(img_path)
        img_norm = _normalize(img).transpose(2, 0, 1)
        return {
            "image": torch.from_numpy(img_norm.copy()).float(),
            "image_id": int(meta["id"]),
            "file_name": meta["file_name"],
            "height": int(meta["height"]),
            "width": int(meta["width"]),
        }
