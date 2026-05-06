#!/usr/bin/env python3
"""Shared hole-aware rasterization and HiSup ring extraction utilities.

Used by the three seg+postprocess baselines (`unet_seg`, `maskrcnn_seg`,
`sam2_seg`) so they share exactly one polygonization policy.

Policy:
- Foreground polygons take `.exterior` only; any interiors shapely may
  attach are discarded.
- Holes come solely from background connected components that are fully
  enclosed (do not touch the image border) and meet `min_hole_area`.
- Holes are assigned to the smallest foreground polygon that fully
  contains them.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import rasterio.features
import shapely.geometry
from shapely.geometry import Polygon


def _flatten_ring(ring_xy: np.ndarray) -> list[float]:
    """Convert (N, 2) numpy coords into the flat HiSup list `[x0, y0, x1, y1, ...]`.

    Trailing closing vertex is dropped if it matches the first vertex — canonical
    HiSup GT segmentations do not repeat the start point.
    """
    coords = np.asarray(ring_xy, dtype=float)
    if coords.size == 0:
        return []
    if coords.shape[0] >= 2 and np.allclose(coords[0], coords[-1]):
        coords = coords[:-1]
    return coords.reshape(-1).tolist()


def _coords_from_flat(flat: Iterable[float]) -> list[tuple[float, float]]:
    vals = [float(v) for v in flat]
    if len(vals) < 6 or len(vals) % 2 != 0:
        return []
    pts = [(vals[i], vals[i + 1]) for i in range(0, len(vals), 2)]
    if pts[0] != pts[-1]:
        pts.append(pts[0])
    return pts


def hisup_annotation_to_mask(
    segmentation: list[list[float]],
    height: int,
    width: int,
) -> np.ndarray:
    """Rasterize a HiSup nested-ring segmentation into a `(H, W) uint8` mask.

    `segmentation` must be `[[ext_flat], [hole0_flat], ...]` where each inner
    list is `[x0, y0, x1, y1, ...]`. Holes are explicitly subtracted — this is
    the correctness fix vs. `pycocotools.annToMask` / `raster_hole_utils` which
    take the union across rings for nested inputs.
    """
    if not segmentation:
        return np.zeros((height, width), dtype=np.uint8)

    ext_pts = _coords_from_flat(segmentation[0])
    if len(ext_pts) < 4:
        return np.zeros((height, width), dtype=np.uint8)

    hole_pts_list: list[list[tuple[float, float]]] = []
    for ring in segmentation[1:]:
        pts = _coords_from_flat(ring)
        if len(pts) >= 4:
            hole_pts_list.append(pts)

    try:
        polygon = Polygon(ext_pts, holes=hole_pts_list)
    except Exception:
        return np.zeros((height, width), dtype=np.uint8)

    if polygon.is_empty or not polygon.is_valid:
        try:
            polygon = polygon.buffer(0)
        except Exception:
            return np.zeros((height, width), dtype=np.uint8)
        if polygon.is_empty:
            return np.zeros((height, width), dtype=np.uint8)

    mask = rasterio.features.rasterize(
        [(polygon, 1)],
        out_shape=(height, width),
        dtype=np.uint8,
        all_touched=False,
    )
    return mask


def _shapes_polygons(
    mask: np.ndarray,
    target_value: int,
    connectivity: int,
) -> list[Polygon]:
    """Polygonize pixels equal to `target_value` into shapely Polygons."""
    assert mask.ndim == 2, "mask must be 2D"
    mask_u8 = (mask == target_value).astype(np.uint8)
    if not mask_u8.any():
        return []
    polys: list[Polygon] = []
    for geom_geojson, val in rasterio.features.shapes(
        mask_u8,
        mask=mask_u8.astype(bool),
        connectivity=connectivity,
    ):
        if int(val) != 1:
            continue
        shape_geom = shapely.geometry.shape(geom_geojson)
        if shape_geom.geom_type == "Polygon":
            polys.append(shape_geom)
        elif shape_geom.geom_type == "MultiPolygon":
            polys.extend(list(shape_geom.geoms))
    return polys


def _touches_border(polygon: Polygon, height: int, width: int) -> bool:
    xmin, ymin, xmax, ymax = polygon.bounds
    eps = 1e-6
    return (
        xmin <= eps
        or ymin <= eps
        or xmax >= (width - eps)
        or ymax >= (height - eps)
    )


def mask_to_hisup_rings(
    mask: np.ndarray,
    *,
    simplify_tol: float = 1.0,
    min_area: float = 16.0,
    min_hole_area: float = 16.0,
    connectivity: int = 4,
    score_map: np.ndarray | None = None,
) -> list[dict]:
    """Convert a binary foreground mask into HiSup nested ring records.

    Parameters
    ----------
    mask : (H, W) ndarray
        1 = foreground, 0 = background.
    simplify_tol : float
        Douglas-Peucker tolerance applied to each final polygon.
    min_area : float
        Drop foreground polygons whose area (post-simplify) falls below this.
    min_hole_area : float
        Minimum pixel area for a background hole to be kept.
    connectivity : int
        4 or 8; passed directly to `rasterio.features.shapes`. Kept identical
        across all three seg+postprocess baselines for fairness.
    score_map : (H, W) ndarray, optional
        Per-pixel soft score in `[0, 1]`. If provided, each record's `score`
        is the mean score inside that foreground polygon (excluding holes).
        If omitted, `score` is set to `None`.

    Returns
    -------
    list[dict]
        Each dict has:
            - ``rings``: `list[list[float]]`, `[[ext_flat], [hole0_flat], ...]`
            - ``bbox``:  `[x, y, w, h]` xywh floats
            - ``area``:  polygon area (post-simplify, with holes subtracted)
            - ``score``: float or None
    """
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {mask.shape}")
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    height, width = mask.shape
    fg_polygons = _shapes_polygons(mask, target_value=1, connectivity=connectivity)
    if not fg_polygons:
        return []

    fg_exteriors: list[Polygon] = []
    for poly in fg_polygons:
        ext_only = Polygon(poly.exterior.coords)
        if ext_only.is_empty or not ext_only.is_valid:
            try:
                ext_only = ext_only.buffer(0)
            except Exception:
                continue
            if ext_only.is_empty or not ext_only.is_valid:
                continue
            if ext_only.geom_type == "MultiPolygon":
                ext_only = max(ext_only.geoms, key=lambda g: g.area)
        fg_exteriors.append(ext_only)

    bg_polygons = _shapes_polygons(mask, target_value=0, connectivity=connectivity)
    interior_bg_holes: list[Polygon] = []
    for bg in bg_polygons:
        if _touches_border(bg, height, width):
            continue
        if bg.area < min_hole_area:
            continue
        hole_outer = Polygon(bg.exterior.coords)
        if not hole_outer.is_valid:
            try:
                hole_outer = hole_outer.buffer(0)
            except Exception:
                continue
        if hole_outer.is_empty:
            continue
        if hole_outer.geom_type == "MultiPolygon":
            hole_outer = max(hole_outer.geoms, key=lambda g: g.area)
        interior_bg_holes.append(hole_outer)

    attached_holes: list[list[Polygon]] = [[] for _ in fg_exteriors]
    for hole in sorted(interior_bg_holes, key=lambda p: -p.area):
        containers = [
            (idx, fg.area)
            for idx, fg in enumerate(fg_exteriors)
            if fg.covers(hole)
        ]
        if not containers:
            continue
        target_idx = min(containers, key=lambda item: item[1])[0]
        attached_holes[target_idx].append(hole)

    records: list[dict] = []
    for fg_ext, holes in zip(fg_exteriors, attached_holes):
        hole_ring_list = [list(h.exterior.coords) for h in holes]
        try:
            full_poly = Polygon(list(fg_ext.exterior.coords), holes=hole_ring_list)
        except Exception:
            full_poly = fg_ext
        if full_poly.is_empty or not full_poly.is_valid:
            try:
                full_poly = full_poly.buffer(0)
            except Exception:
                continue
        if full_poly.is_empty or full_poly.geom_type not in {"Polygon", "MultiPolygon"}:
            continue

        simplified = full_poly.simplify(simplify_tol, preserve_topology=True)
        if simplified.is_empty:
            continue
        if simplified.geom_type == "MultiPolygon":
            simplified = max(simplified.geoms, key=lambda g: g.area)
        if simplified.geom_type != "Polygon":
            continue

        if simplified.area < min_area:
            continue

        ext_coords = np.asarray(simplified.exterior.coords, dtype=float)
        if ext_coords.shape[0] < 4:
            continue
        rings: list[list[float]] = [_flatten_ring(ext_coords)]
        for interior in simplified.interiors:
            flat = _flatten_ring(np.asarray(interior.coords, dtype=float))
            if len(flat) >= 6:
                rings.append(flat)

        xmin, ymin, xmax, ymax = simplified.bounds
        bbox = [float(xmin), float(ymin), float(xmax - xmin), float(ymax - ymin)]

        score: float | None = None
        if score_map is not None:
            poly_mask = rasterio.features.rasterize(
                [(simplified, 1)],
                out_shape=(height, width),
                dtype=np.uint8,
                all_touched=False,
            ).astype(bool)
            if poly_mask.any():
                score = float(score_map[poly_mask].mean())
            else:
                score = 0.0

        records.append(
            {
                "rings": rings,
                "bbox": bbox,
                "area": float(simplified.area),
                "score": score,
            }
        )

    return records
