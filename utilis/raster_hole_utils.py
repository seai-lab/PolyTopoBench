#!/usr/bin/env python3
"""Core raster-hole detection helpers for COCO polygon annotations."""

from __future__ import annotations

from typing import Any

import numpy as np


def segmentation_to_closed_walk(segmentation: list[float]) -> list[tuple[float, float]]:
    coords = [(float(segmentation[i]), float(segmentation[i + 1])) for i in range(0, len(segmentation), 2)]
    if not coords:
        return []
    deduped = [coords[0]]
    for point in coords[1:]:
        if point != deduped[-1]:
            deduped.append(point)
    if len(deduped) < 3:
        return []
    if deduped[0] != deduped[-1]:
        deduped.append(deduped[0])
    return deduped


def even_odd_mask_from_ring(
    walk: list[tuple[float, float]],
    width: int,
    height: int,
) -> tuple[np.ndarray, tuple[int, int]]:
    if len(walk) < 4:
        return np.zeros((0, 0), dtype=bool), (0, 0)

    coords = np.asarray(walk, dtype=float)
    min_x = max(int(np.floor(coords[:, 0].min())), 0)
    max_x = min(int(np.ceil(coords[:, 0].max())), width)
    min_y = max(int(np.floor(coords[:, 1].min())), 0)
    max_y = min(int(np.ceil(coords[:, 1].max())), height)
    if min_x >= max_x or min_y >= max_y:
        return np.zeros((0, 0), dtype=bool), (min_x, min_y)

    x_centers = np.arange(min_x, max_x, dtype=float) + 0.5
    y_centers = np.arange(min_y, max_y, dtype=float) + 0.5
    xx, yy = np.meshgrid(x_centers, y_centers)
    inside = np.zeros(xx.shape, dtype=bool)

    x0 = coords[:-1, 0]
    y0 = coords[:-1, 1]
    x1 = coords[1:, 0]
    y1 = coords[1:, 1]
    for xa, ya, xb, yb in zip(x0, y0, x1, y1):
        if ya == yb:
            continue
        crosses = ((ya > yy) != (yb > yy))
        intersect_x = (xb - xa) * (yy - ya) / (yb - ya) + xa
        inside ^= crosses & (xx < intersect_x)

    return inside, (min_x, min_y)


def annotation_mask(segmentation: list[list[float]], width: int, height: int) -> tuple[np.ndarray, tuple[int, int]]:
    local_masks: list[tuple[np.ndarray, tuple[int, int]]] = []
    max_x = 0
    max_y = 0
    min_x = width
    min_y = height
    for segment in segmentation:
        walk = segmentation_to_closed_walk(segment)
        if len(walk) < 4:
            continue
        submask, (offset_x, offset_y) = even_odd_mask_from_ring(walk, width=width, height=height)
        if submask.size == 0:
            continue
        local_masks.append((submask, (offset_x, offset_y)))
        min_x = min(min_x, offset_x)
        min_y = min(min_y, offset_y)
        max_x = max(max_x, offset_x + submask.shape[1])
        max_y = max(max_y, offset_y + submask.shape[0])

    if not local_masks:
        return np.zeros((0, 0), dtype=bool), (0, 0)

    full = np.zeros((max_y - min_y, max_x - min_x), dtype=bool)
    for submask, (offset_x, offset_y) in local_masks:
        row0 = offset_y - min_y
        col0 = offset_x - min_x
        full[row0 : row0 + submask.shape[0], col0 : col0 + submask.shape[1]] |= submask
    return full, (min_x, min_y)


def detect_holes(mask: np.ndarray, min_hole_area: int) -> tuple[int, int]:
    if mask.size == 0 or not mask.any():
        return 0, 0

    background = ~mask
    visited = np.zeros_like(background, dtype=bool)
    hole_count = 0
    hole_pixels = 0

    for row in range(background.shape[0]):
        for col in range(background.shape[1]):
            if not background[row, col] or visited[row, col]:
                continue
            stack = [(row, col)]
            visited[row, col] = True
            component_size = 0
            touches_border = False

            while stack:
                cur_row, cur_col = stack.pop()
                component_size += 1
                if (
                    cur_row == 0
                    or cur_row == background.shape[0] - 1
                    or cur_col == 0
                    or cur_col == background.shape[1] - 1
                ):
                    touches_border = True

                for next_row, next_col in (
                    (cur_row - 1, cur_col),
                    (cur_row + 1, cur_col),
                    (cur_row, cur_col - 1),
                    (cur_row, cur_col + 1),
                ):
                    if (
                        0 <= next_row < background.shape[0]
                        and 0 <= next_col < background.shape[1]
                        and background[next_row, next_col]
                        and not visited[next_row, next_col]
                    ):
                        visited[next_row, next_col] = True
                        stack.append((next_row, next_col))

            if touches_border or component_size < min_hole_area:
                continue

            hole_count += 1
            hole_pixels += component_size

    return hole_count, hole_pixels


def detect_annotation_holes(
    annotation: dict[str, Any],
    image_meta: dict[str, Any],
    min_hole_area: int,
) -> dict[str, Any] | None:
    segmentation = annotation.get("segmentation", [])
    if not isinstance(segmentation, list) or not segmentation:
        return None

    mask, (offset_x, offset_y) = annotation_mask(
        segmentation=segmentation,
        width=int(image_meta["width"]),
        height=int(image_meta["height"]),
    )
    hole_count, hole_pixels = detect_holes(mask, min_hole_area=min_hole_area)
    if hole_count == 0:
        return None

    return {
        "polygon_id": int(annotation["id"]),
        "annotation_id": int(annotation["id"]),
        "image_id": int(annotation["image_id"]),
        "category_id": int(annotation.get("category_id", -1)),
        "file_name": image_meta.get("file_name"),
        "hole_count": int(hole_count),
        "hole_pixels": int(hole_pixels),
        "bbox": annotation.get("bbox"),
        "area": annotation.get("area"),
        "iscrowd": int(annotation.get("iscrowd", 0)),
        "segmentation_parts": len(segmentation),
        "mask_width": int(mask.shape[1]),
        "mask_height": int(mask.shape[0]),
        "offset_x": int(offset_x),
        "offset_y": int(offset_y),
    }
