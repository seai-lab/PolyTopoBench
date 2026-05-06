#!/usr/bin/env python3
"""torchvision-Mask-R-CNN-ready dataset backed by the maskrcnn_seg mirror.

Reads canonical HiSup `annotation.json`, remaps `category_id=100 → label=1`,
and rasterizes each instance via the hole-aware rasterizer shared with the
other seg+postprocess baselines.
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mask_to_hisup import hisup_annotation_to_mask

Image.MAX_IMAGE_PIXELS = None


def _load_image_rgb(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def _random_hflip(img: np.ndarray, boxes: np.ndarray, masks: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if img.shape[0] == 0:
        return img, boxes, masks
    img = np.ascontiguousarray(img[:, ::-1])
    masks = np.ascontiguousarray(masks[:, :, ::-1])
    W = img.shape[1]
    new_boxes = boxes.copy()
    new_boxes[:, 0] = W - boxes[:, 2]
    new_boxes[:, 2] = W - boxes[:, 0]
    return img, new_boxes, masks


class MaskRcnnSegDataset(Dataset):
    """Returns `(image_tensor, target_dict)` tuples expected by torchvision Mask R-CNN.

    image_tensor: FloatTensor (3, H, W) in `[0, 1]`.
    target_dict: {
        "boxes":   FloatTensor (N, 4) xyxy,
        "labels":  Int64Tensor (N,) — all 1s (remapped from category_id=100),
        "masks":   UInt8Tensor (N, H, W),
        "image_id": Int64 scalar Tensor,
        "area":    FloatTensor (N,),
        "iscrowd": Int64 (N,),
    }
    """

    def __init__(self, mirror_root: Path, split: str = "train", augment: bool = True,
                 min_area_px: float = 4.0):
        self.mirror_root = Path(mirror_root)
        self.split_dir = self.mirror_root / split
        ann_json = self.split_dir / "annotation.json"
        with ann_json.open("r") as f:
            coco = json.load(f)
        self.images = coco["images"]
        self.images_dir = self.split_dir / "images"
        self.image_id_to_ann = defaultdict(list)
        for ann in coco["annotations"]:
            self.image_id_to_ann[int(ann["image_id"])].append(ann)
        self.augment = augment
        self.min_area_px = min_area_px

    def __len__(self) -> int:
        return len(self.images)

    def _build_target(self, image_id: int, height: int, width: int) -> dict:
        anns = self.image_id_to_ann.get(image_id, [])
        boxes = []
        masks = []
        for ann in anns:
            seg = ann.get("segmentation", [])
            if not seg:
                continue
            mask = hisup_annotation_to_mask(seg, height=height, width=width)
            if mask.sum() < self.min_area_px:
                continue
            ys, xs = np.where(mask > 0)
            if ys.size == 0:
                continue
            y0, y1 = ys.min(), ys.max()
            x0, x1 = xs.min(), xs.max()
            if (x1 - x0) < 1 or (y1 - y0) < 1:
                continue
            boxes.append([float(x0), float(y0), float(x1 + 1), float(y1 + 1)])
            masks.append(mask)
        if not boxes:
            return None
        boxes = np.asarray(boxes, dtype=np.float32)
        masks = np.stack(masks, axis=0)
        return {"boxes": boxes, "masks": masks}

    def __getitem__(self, idx: int):
        meta = self.images[idx]
        image_id = int(meta["id"])
        height = int(meta["height"])
        width = int(meta["width"])
        img_path = self.images_dir / meta["file_name"]
        img = _load_image_rgb(img_path)

        target_raw = self._build_target(image_id, height, width)
        if target_raw is None:
            boxes = np.zeros((0, 4), dtype=np.float32)
            masks = np.zeros((0, height, width), dtype=np.uint8)
        else:
            boxes = target_raw["boxes"]
            masks = target_raw["masks"]

        if self.augment and random.random() < 0.5 and boxes.shape[0] > 0:
            img, boxes, masks = _random_hflip(img, boxes, masks)

        img_tensor = torch.from_numpy(img.copy()).float().permute(2, 0, 1) / 255.0
        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)
        boxes_t = torch.from_numpy(boxes).float()
        masks_t = torch.from_numpy(masks.astype(np.uint8).copy()).to(torch.uint8)
        area = (boxes_t[:, 2] - boxes_t[:, 0]) * (boxes_t[:, 3] - boxes_t[:, 1]) if boxes_t.numel() else torch.zeros(0)
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        target = {
            "boxes": boxes_t,
            "labels": labels,
            "masks": masks_t,
            "image_id": torch.tensor([image_id], dtype=torch.int64),
            "area": area.float(),
            "iscrowd": iscrowd,
        }
        return img_tensor, target


class MaskRcnnValImageDataset(Dataset):
    """Inference loader returning raw images + image_id, for post-inference scoring."""

    def __init__(self, mirror_root: Path):
        self.mirror_root = Path(mirror_root)
        self.split_dir = self.mirror_root / "val"
        with (self.split_dir / "annotation.json").open("r") as f:
            coco = json.load(f)
        self.images = coco["images"]
        self.images_dir = self.split_dir / "images"

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        meta = self.images[idx]
        img_path = self.images_dir / meta["file_name"]
        img = _load_image_rgb(img_path)
        img_tensor = torch.from_numpy(img.copy()).float().permute(2, 0, 1) / 255.0
        return {
            "image": img_tensor,
            "image_id": int(meta["id"]),
            "file_name": meta["file_name"],
            "height": int(meta["height"]),
            "width": int(meta["width"]),
        }


def collate_train(batch):
    images = [b[0] for b in batch]
    targets = [b[1] for b in batch]
    return images, targets


def collate_val(batch):
    return {
        "image": [b["image"] for b in batch],
        "image_id": [b["image_id"] for b in batch],
        "file_name": [b["file_name"] for b in batch],
        "height": [b["height"] for b in batch],
        "width": [b["width"] for b in batch],
    }
