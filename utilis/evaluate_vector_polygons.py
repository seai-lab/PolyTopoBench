#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

try:
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial import cKDTree
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: scipy. Run this evaluator from an environment that has "
        "SciPy installed, for example `conda run -n roipoly python ...`."
    ) from exc
try:
    from shapely.errors import GEOSException
    from shapely.geometry import (
        GeometryCollection,
        LineString,
        MultiLineString,
        MultiPolygon,
        Point,
        Polygon,
    )
    from shapely.ops import unary_union
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: shapely. Run this evaluator from an environment that has "
        "Shapely installed, for example `conda run -n roipoly python ...` or "
        "`conda run -n hisup python ...`."
    ) from exc
from tqdm import tqdm

from export_roipoly_predictions_to_geojson import (
    compute_depths,
    infer_parent_indices,
)
try:
    from shapely.validation import make_valid
except ImportError:  # pragma: no cover
    make_valid = None


DEFAULT_AP_THRESHOLDS = [round(0.50 + 0.05 * i, 2) for i in range(10)]
AP_WORKER_CTX = None
PAIRWISE_WORKER_CTX = None
ROLE_WORKER_CTX = None


@dataclass
class BuildingRecord:
    image_id: int
    geometry: Polygon
    exterior: np.ndarray
    holes: list[np.ndarray]
    score: float
    source_indices: list[int]

    @property
    def all_rings(self) -> list[np.ndarray]:
        return [self.exterior, *self.holes]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate patch-level building vector predictions for HiSup- or RoIPoly-style "
            "outputs with vector-native polygon metrics."
        )
    )
    parser.add_argument("--pred", type=Path, required=True, help="Prediction JSON file.")
    parser.add_argument("--gt", type=Path, required=True, help="Ground-truth annotation JSON file.")
    parser.add_argument(
        "--dataset-type",
        choices=["hisup", "roipoly", "auto"],
        default="auto",
        help=(
            "Interpretation of the input pair when GT and prediction use the same schema. "
            "'auto' uses GT schema to choose both loaders."
        ),
    )
    parser.add_argument(
        "--gt-type",
        choices=["hisup", "roipoly", "auto"],
        default=None,
        help="Optional override for the ground-truth loader type.",
    )
    parser.add_argument(
        "--pred-type",
        choices=["hisup", "roipoly", "auto"],
        default=None,
        help="Optional override for the prediction loader type.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write metrics JSON. Defaults next to the prediction file.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.0,
        help="Drop predictions below this score before evaluation.",
    )
    parser.add_argument(
        "--pair-match-iou-threshold",
        type=float,
        default=0.5,
        help="Building-level IoU threshold used for pairwise metrics and topology matching.",
    )
    parser.add_argument(
        "--boundary-radius",
        type=float,
        default=1.0,
        help="Vector buffer radius used for boundary IoU.",
    )
    parser.add_argument(
        "--boundary-match-threshold",
        type=float,
        default=0.5,
        help="Boundary IoU threshold used when deciding ring matches for F1/topology metrics.",
    )
    parser.add_argument(
        "--sample-spacing",
        type=float,
        default=2.0,
        help="Boundary sampling spacing used by HD95 and MTA.",
    )
    parser.add_argument(
        "--mta-min-precision",
        type=float,
        default=0.5,
        help="Minimum precision filter used by the MTA computation.",
    )
    parser.add_argument(
        "--mta-max-stretch",
        type=float,
        default=2.0,
        help="Maximum projection stretch used by the MTA computation.",
    )
    parser.add_argument(
        "--corner-simplify-tol",
        type=float,
        default=1.0,
        help="Tolerance used to simplify rings before corner extraction.",
    )
    parser.add_argument(
        "--corner-thresholds",
        default="2,5",
        help="Comma-separated distance thresholds used for Corner F1.",
    )
    parser.add_argument(
        "--ap-thresholds",
        default=",".join(str(value) for value in DEFAULT_AP_THRESHOLDS),
        help="Comma-separated polygon IoU thresholds used for AP.",
    )
    parser.add_argument(
        "--roipoly-dedup-iou-threshold",
        type=float,
        default=0.6,
        help="RoIPoly ring deduplication IoU threshold before hierarchy reconstruction.",
    )
    parser.add_argument(
        "--roipoly-dedup-overlap-threshold",
        type=float,
        default=0.85,
        help="RoIPoly ring deduplication overlap/min(area) threshold before hierarchy reconstruction.",
    )
    parser.add_argument(
        "--roipoly-containment-threshold",
        type=float,
        default=0.98,
        help="Containment threshold used to infer parent-child nesting for RoIPoly predictions.",
    )
    parser.add_argument(
        "--roipoly-min-area",
        type=float,
        default=10.0,
        help="Drop RoIPoly predicted rings with area below this threshold after sanitization.",
    )
    parser.add_argument(
        "--min-hole-area",
        type=float,
        default=0.0,
        help=(
            "Optionally drop interior rings whose polygon area is below this threshold for both "
            "GT and predictions before computing metrics."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce progress output.",
    )
    parser.add_argument(
        "--image-ids",
        default="",
        help="Optional comma-separated patch image ids to evaluate.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional debug knob: evaluate only the first N sorted image_ids shared across GT/pred unions.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=16,
        help="Number of worker processes used for parallel metric computation.",
    )
    return parser.parse_args()


def log(message: str, quiet: bool) -> None:
    if not quiet:
        print(message)


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def is_closed(coords: np.ndarray, tol: float = 1e-6) -> bool:
    return coords.shape[0] >= 2 and np.allclose(coords[0], coords[-1], atol=tol)


def ring_from_flat_coords(flat_coords: Sequence[float]) -> np.ndarray | None:
    if not flat_coords or len(flat_coords) < 6 or len(flat_coords) % 2 != 0:
        return None
    ring = np.asarray(flat_coords, dtype=np.float64).reshape(-1, 2)
    if is_closed(ring):
        ring = ring[:-1]
    if ring.shape[0] < 3:
        return None
    return ring


def close_ring(ring: np.ndarray) -> np.ndarray:
    if ring.shape[0] == 0:
        return ring
    if is_closed(ring):
        return ring
    return np.vstack([ring, ring[0]])


def polygon_from_rings(exterior: np.ndarray, holes: list[np.ndarray]) -> Polygon | None:
    try:
        polygon = Polygon(close_ring(exterior), [close_ring(hole) for hole in holes])
    except Exception:
        return None
    return sanitize_polygon_geometry(polygon)


def sanitize_polygon_geometry(geometry) -> Polygon | None:
    if geometry is None or geometry.is_empty:
        return None
    if not geometry.is_valid:
        if make_valid is not None:
            geometry = make_valid(geometry)
        else:  # pragma: no cover
            geometry = geometry.buffer(0)
    parts = []
    if isinstance(geometry, Polygon):
        parts = [geometry]
    elif isinstance(geometry, MultiPolygon):
        parts = [part for part in geometry.geoms if not part.is_empty]
    elif isinstance(geometry, GeometryCollection):
        parts = [part for part in geometry.geoms if isinstance(part, Polygon) and not part.is_empty]
    if not parts:
        return None
    polygon = max(parts, key=lambda item: item.area)
    if polygon.area <= 0 or len(polygon.exterior.coords) < 4:
        return None
    return polygon


def polygon_iou(left: Polygon, right: Polygon) -> float:
    inter = left.intersection(right).area
    if inter <= 0:
        return 0.0
    union = left.area + right.area - inter
    return inter / union if union > 0 else 0.0


def ring_polygon_area(ring: np.ndarray) -> float:
    polygon = sanitize_polygon_geometry(Polygon(close_ring(ring)))
    return float(polygon.area) if polygon is not None else 0.0


def overlap_small_ratio(left: Polygon, right: Polygon) -> float:
    inter = left.intersection(right).area
    if inter <= 0:
        return 0.0
    return inter / max(min(left.area, right.area), 1e-8)


def is_strict_containment_pair(left: Polygon, right: Polygon, containment_threshold: float) -> bool:
    if left.area <= 0 or right.area <= 0:
        return False
    parent, child = (left, right) if left.area >= right.area else (right, left)
    anchor = child.representative_point()
    if not parent.covers(anchor):
        return False
    covered_ratio = parent.intersection(child).area / max(child.area, 1e-8)
    return covered_ratio >= containment_threshold


def deduplicate_roipoly_patch_rings(
    items: list[dict],
    iou_threshold: float,
    overlap_threshold: float,
    containment_threshold: float,
) -> list[dict]:
    if len(items) <= 1:
        return items
    order = sorted(
        range(len(items)),
        key=lambda index: (items[index]["score"], items[index]["geometry"].area),
        reverse=True,
    )
    suppressed = set()
    kept = []
    for position, index in enumerate(order):
        if index in suppressed:
            continue
        kept.append(index)
        geometry = items[index]["geometry"]
        for candidate in order[position + 1 :]:
            if candidate in suppressed:
                continue
            candidate_geometry = items[candidate]["geometry"]
            iou = polygon_iou(geometry, candidate_geometry)
            if iou >= iou_threshold:
                suppressed.add(candidate)
                continue
            overlap_small = overlap_small_ratio(geometry, candidate_geometry)
            if overlap_small < overlap_threshold:
                continue
            # Preserve strict containment pairs so nested rings can survive into hole reconstruction.
            if is_strict_containment_pair(
                geometry,
                candidate_geometry,
                containment_threshold=containment_threshold,
            ):
                continue
            suppressed.add(candidate)
    return [items[index] for index in kept]


def multiline_from_rings(rings: list[np.ndarray]):
    lines = [LineString(close_ring(ring)) for ring in rings if ring.shape[0] >= 3]
    if not lines:
        return None
    if len(lines) == 1:
        return lines[0]
    return MultiLineString(lines)


def sanitize_areal_geometry(geometry):
    if geometry is None or geometry.is_empty:
        return None
    if not geometry.is_valid:
        if make_valid is not None:
            geometry = make_valid(geometry)
        else:  # pragma: no cover
            geometry = geometry.buffer(0)
    if geometry.is_empty:
        return None
    if isinstance(geometry, (Polygon, MultiPolygon)):
        return geometry
    if isinstance(geometry, GeometryCollection):
        polygon_parts = []
        for part in geometry.geoms:
            if isinstance(part, Polygon):
                polygon_parts.append(part)
            elif isinstance(part, MultiPolygon):
                polygon_parts.extend([item for item in part.geoms if not item.is_empty])
        if not polygon_parts:
            return None
        merged = unary_union(polygon_parts)
        if not merged.is_valid:
            merged = merged.buffer(0)
        return merged if not merged.is_empty else None
    return None


def boundary_band_iou(gt_rings: list[np.ndarray], pred_rings: list[np.ndarray], radius: float) -> float:
    gt_lines = multiline_from_rings(gt_rings)
    pred_lines = multiline_from_rings(pred_rings)
    if gt_lines is None or pred_lines is None:
        return 0.0
    gt_band = sanitize_areal_geometry(gt_lines.buffer(radius, cap_style=2, join_style=2))
    pred_band = sanitize_areal_geometry(pred_lines.buffer(radius, cap_style=2, join_style=2))
    if gt_band is None or pred_band is None:
        return 0.0
    try:
        inter = gt_band.intersection(pred_band).area
        union = gt_band.union(pred_band).area
    except GEOSException:
        gt_band = sanitize_areal_geometry(gt_band.buffer(0))
        pred_band = sanitize_areal_geometry(pred_band.buffer(0))
        if gt_band is None or pred_band is None:
            return 0.0
        inter = gt_band.intersection(pred_band).area
        union = gt_band.union(pred_band).area
    return inter / union if union > 0 else 0.0


def polis_distance(gt_rings: list[np.ndarray], pred_rings: list[np.ndarray]) -> float:
    gt_boundary = multiline_from_rings(gt_rings)
    pred_boundary = multiline_from_rings(pred_rings)
    if gt_boundary is None or pred_boundary is None:
        return float("nan")

    def mean_distance(source_rings: list[np.ndarray], target_boundary) -> float:
        dists = []
        for ring in source_rings:
            for point in ring:
                dists.append(target_boundary.distance(Point(float(point[0]), float(point[1]))))
        return float(np.mean(dists)) if dists else 0.0

    return 0.5 * (mean_distance(gt_rings, pred_boundary) + mean_distance(pred_rings, gt_boundary))


def sample_ring_points(ring: np.ndarray, spacing: float) -> np.ndarray:
    closed = close_ring(ring)
    if closed.shape[0] < 2:
        return closed
    sampled = []
    for index in range(closed.shape[0] - 1):
        start = closed[index]
        end = closed[index + 1]
        length = float(np.linalg.norm(end - start))
        num = max(1, int(round(length / max(spacing, 1e-6)))) + 1
        xs = np.linspace(start[0], end[0], num=num)
        ys = np.linspace(start[1], end[1], num=num)
        coords = np.stack([xs, ys], axis=1)
        if index > 0:
            coords = coords[1:]
        sampled.append(coords)
    return np.concatenate(sampled, axis=0) if sampled else closed


def sample_boundary_points(rings: list[np.ndarray], spacing: float) -> np.ndarray:
    parts = [sample_ring_points(ring, spacing) for ring in rings if ring.shape[0] >= 3]
    if not parts:
        return np.zeros((0, 2), dtype=np.float64)
    return np.concatenate(parts, axis=0)


def hd95_distance(gt_rings: list[np.ndarray], pred_rings: list[np.ndarray], spacing: float) -> float:
    gt_points = sample_boundary_points(gt_rings, spacing)
    pred_points = sample_boundary_points(pred_rings, spacing)
    if gt_points.shape[0] == 0 or pred_points.shape[0] == 0:
        return float("nan")
    gt_tree = cKDTree(gt_points)
    pred_tree = cKDTree(pred_points)
    d_pred_to_gt = gt_tree.query(pred_points, k=1)[0]
    d_gt_to_pred = pred_tree.query(gt_points, k=1)[0]
    return float(max(np.quantile(d_pred_to_gt, 0.95), np.quantile(d_gt_to_pred, 0.95)))


def sample_geometry(geom, density: float):
    if isinstance(geom, GeometryCollection):
        return GeometryCollection([sample_geometry(item, density) for item in geom.geoms])
    if isinstance(geom, Polygon):
        sampled_exterior = sample_geometry(geom.exterior, density)
        sampled_interiors = [sample_geometry(interior, density) for interior in geom.interiors]
        return Polygon(sampled_exterior, sampled_interiors)
    if isinstance(geom, LineString):
        coords = np.asarray(geom.coords[:], dtype=np.float64)
        lengths = np.linalg.norm(coords[1:] - coords[:-1], axis=1)
        sampled_chunks = []
        for index, length in enumerate(lengths):
            start = coords[index]
            end = coords[index + 1]
            num = max(1, int(round(length / max(density, 1e-6)))) + 1
            xs = np.linspace(start[0], end[0], num=num)
            ys = np.linspace(start[1], end[1], num=num)
            chunk = np.stack([xs, ys], axis=1)
            if index > 0:
                chunk = chunk[1:]
            sampled_chunks.append(chunk)
        sampled = np.concatenate(sampled_chunks, axis=0) if sampled_chunks else coords
        return LineString(sampled)
    raise TypeError(f"Unsupported geometry type for sampling: {type(geom)}")


def nearest_projection(target_contours: list[LineString], point: Point):
    best = None
    best_dist = float("inf")
    for contour in target_contours:
        distance = point.distance(contour)
        if distance < best_dist:
            best_dist = distance
            best = contour
    return best, best_dist


def compute_contour_measure(pred_polygon: Polygon, gt_polygon: Polygon, sampling_spacing: float, max_stretch: float) -> float:
    pred_contours = [pred_polygon.exterior, *pred_polygon.interiors]
    gt_contours = [gt_polygon.exterior, *gt_polygon.interiors]
    sampled_pred_contours = [sample_geometry(contour, sampling_spacing) for contour in pred_contours]

    contour_measures = []
    for contour in sampled_pred_contours:
        coords = np.asarray(contour.coords[:], dtype=np.float64)
        if coords.shape[0] < 2:
            continue
        projected_coords = []
        for x, y in coords:
            point = Point(float(x), float(y))
            best_contour, _ = nearest_projection(gt_contours, point)
            if best_contour is None:
                projected_coords.append([x, y])
                continue
            t = best_contour.project(point)
            projected_point = best_contour.interpolate(t)
            projected_coords.append([projected_point.x, projected_point.y])
        projected_coords = np.asarray(projected_coords, dtype=np.float64)

        edges = coords[1:] - coords[:-1]
        proj_edges = projected_coords[1:] - projected_coords[:-1]
        edge_norms = np.linalg.norm(edges, axis=1)
        proj_edge_norms = np.linalg.norm(proj_edges, axis=1)
        valid = (edge_norms > 0) & (proj_edge_norms > 0)
        if not np.any(valid):
            continue
        edges = edges[valid]
        proj_edges = proj_edges[valid]
        edge_norms = edge_norms[valid]
        proj_edge_norms = proj_edge_norms[valid]

        stretch = edge_norms / proj_edge_norms
        valid = (stretch > 1.0 / max_stretch) & (stretch < max_stretch)
        if not np.any(valid):
            continue
        edges = edges[valid]
        proj_edges = proj_edges[valid]
        edge_norms = edge_norms[valid]
        proj_edge_norms = proj_edge_norms[valid]

        cosine = np.abs(np.sum(edges * proj_edges, axis=1) / (edge_norms * proj_edge_norms))
        if cosine.size == 0:
            continue
        contour_measures.append(float(np.min(np.clip(cosine, -1.0, 1.0))))

    if not contour_measures:
        return float("nan")
    return float(np.degrees(np.arccos(min(contour_measures))))


def mta_distance(pred_polygon: Polygon, gt_polygon: Polygon, min_precision: float, sampling_spacing: float, max_stretch: float) -> float:
    if pred_polygon.area <= 0 or gt_polygon.area <= 0:
        return float("nan")
    precision = pred_polygon.intersection(gt_polygon).area / max(pred_polygon.area, 1e-8)
    if precision < min_precision:
        return float("nan")
    return compute_contour_measure(pred_polygon, gt_polygon, sampling_spacing=sampling_spacing, max_stretch=max_stretch)


def simplify_ring_for_corners(ring: np.ndarray, tolerance: float) -> np.ndarray:
    polygon = Polygon(close_ring(ring))
    polygon = sanitize_polygon_geometry(polygon)
    if polygon is None:
        return ring
    simple = polygon.simplify(tolerance, preserve_topology=True)
    simple = sanitize_polygon_geometry(simple) or polygon
    coords = np.asarray(simple.exterior.coords[:-1], dtype=np.float64)
    return coords if coords.shape[0] >= 3 else ring


def match_points(gt_points: np.ndarray, pred_points: np.ndarray, threshold: float) -> tuple[int, int, int]:
    if gt_points.shape[0] == 0 and pred_points.shape[0] == 0:
        return 0, 0, 0
    if gt_points.shape[0] == 0:
        return 0, pred_points.shape[0], 0
    if pred_points.shape[0] == 0:
        return 0, 0, gt_points.shape[0]
    distances = np.linalg.norm(gt_points[:, None, :] - pred_points[None, :, :], axis=2)
    gt_indices, pred_indices = linear_sum_assignment(distances)
    matched = 0
    for gt_index, pred_index in zip(gt_indices, pred_indices):
        if distances[gt_index, pred_index] <= threshold:
            matched += 1
    return matched, pred_points.shape[0] - matched, gt_points.shape[0] - matched


def match_buildings(
    gt_records: list[BuildingRecord],
    pred_records: list[BuildingRecord],
    iou_threshold: float,
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    if not gt_records and not pred_records:
        return [], [], []
    if not gt_records:
        return [], [], list(range(len(pred_records)))
    if not pred_records:
        return [], list(range(len(gt_records))), []

    ious = np.zeros((len(gt_records), len(pred_records)), dtype=np.float64)
    for gt_index, gt_record in enumerate(gt_records):
        for pred_index, pred_record in enumerate(pred_records):
            ious[gt_index, pred_index] = polygon_iou(gt_record.geometry, pred_record.geometry)

    row_ind, col_ind = linear_sum_assignment(1.0 - ious)
    matches = []
    matched_gt = set()
    matched_pred = set()
    for gt_index, pred_index in zip(row_ind, col_ind):
        iou = ious[gt_index, pred_index]
        if iou >= iou_threshold:
            matches.append((gt_index, pred_index, float(iou)))
            matched_gt.add(gt_index)
            matched_pred.add(pred_index)

    unmatched_gt = [index for index in range(len(gt_records)) if index not in matched_gt]
    unmatched_pred = [index for index in range(len(pred_records)) if index not in matched_pred]
    return matches, unmatched_gt, unmatched_pred


def greedy_match_detections(
    gt_by_image: dict[int, list[BuildingRecord]],
    pred_by_image: dict[int, list[BuildingRecord]],
    iou_threshold: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    predictions = []
    for image_id, preds in pred_by_image.items():
        for pred_index, pred in enumerate(preds):
            predictions.append((float(pred.score), image_id, pred_index, pred))
    predictions.sort(key=lambda item: item[0], reverse=True)

    gt_matched = {
        image_id: np.zeros(len(gt_by_image.get(image_id, [])), dtype=bool)
        for image_id in set(gt_by_image) | set(pred_by_image)
    }
    tp = np.zeros(len(predictions), dtype=np.float64)
    fp = np.zeros(len(predictions), dtype=np.float64)
    total_gt = sum(len(records) for records in gt_by_image.values())

    for pred_position, (_, image_id, _, pred_record) in enumerate(predictions):
        gt_records = gt_by_image.get(image_id, [])
        if not gt_records:
            fp[pred_position] = 1.0
            continue

        ious = np.asarray(
            [polygon_iou(pred_record.geometry, gt_record.geometry) for gt_record in gt_records],
            dtype=np.float64,
        )
        best_gt = int(np.argmax(ious))
        best_iou = float(ious[best_gt])
        if best_iou >= iou_threshold and not gt_matched[image_id][best_gt]:
            tp[pred_position] = 1.0
            gt_matched[image_id][best_gt] = True
        else:
            fp[pred_position] = 1.0

    return tp, fp, total_gt


def compute_average_precision(tp: np.ndarray, fp: np.ndarray, total_gt: int) -> float:
    if total_gt == 0:
        return float("nan")
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / total_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)

    recall_points = np.linspace(0.0, 1.0, 101)
    precision_interp = []
    for threshold in recall_points:
        valid = precision[recall >= threshold]
        precision_interp.append(np.max(valid) if valid.size else 0.0)
    return float(np.mean(precision_interp))


def get_rings_for_role(records: list[BuildingRecord], role: str) -> list[np.ndarray]:
    if role == "exterior":
        return [record.exterior for record in records]
    if role == "hole":
        return [hole for record in records for hole in record.holes]
    raise ValueError(f"Unsupported role: {role}")


def pairwise_metrics_for_image(
    gt_records: list[BuildingRecord],
    pred_records: list[BuildingRecord],
    *,
    pair_match_iou_threshold: float,
    boundary_radius: float,
    boundary_match_threshold: float,
    sample_spacing: float,
    mta_min_precision: float,
    mta_max_stretch: float,
) -> dict:
    matches, unmatched_gt, unmatched_pred = match_buildings(
        gt_records,
        pred_records,
        iou_threshold=pair_match_iou_threshold,
    )
    overall_boundary_ious = []
    overall_polis = []
    overall_hd95 = []
    overall_mta = []
    topology_sum = 0
    topology_count = len(unmatched_gt) + len(unmatched_pred)

    for gt_index, pred_index, _ in matches:
        gt_record = gt_records[gt_index]
        pred_record = pred_records[pred_index]

        overall_boundary_ious.append(
            boundary_band_iou(gt_record.all_rings, pred_record.all_rings, radius=boundary_radius)
        )
        overall_polis.append(polis_distance(gt_record.all_rings, pred_record.all_rings))
        overall_hd95.append(hd95_distance(gt_record.all_rings, pred_record.all_rings, spacing=sample_spacing))
        overall_mta.append(
            mta_distance(
                pred_record.geometry,
                gt_record.geometry,
                min_precision=mta_min_precision,
                sampling_spacing=sample_spacing,
                max_stretch=mta_max_stretch,
            )
        )

        topology_sum += int(
            boundary_band_iou([gt_record.exterior], [pred_record.exterior], radius=boundary_radius)
            >= boundary_match_threshold
            and len(gt_record.holes) == len(pred_record.holes)
            and len(
                role_specific_ring_match(
                    gt_record.holes,
                    pred_record.holes,
                    boundary_radius=boundary_radius,
                    match_threshold=boundary_match_threshold,
                )[0]
            )
            == len(gt_record.holes)
            == len(pred_record.holes)
        )
        topology_count += 1

    return {
        "matched_buildings": len(matches),
        "overall_boundary_ious": overall_boundary_ious,
        "overall_polis": overall_polis,
        "overall_hd95": overall_hd95,
        "overall_mta": overall_mta,
        "topology_sum": topology_sum,
        "topology_count": topology_count,
    }


def role_metrics_for_image(
    gt_records: list[BuildingRecord],
    pred_records: list[BuildingRecord],
    *,
    role: str,
    boundary_radius: float,
    match_threshold: float,
    sample_spacing: float,
    mta_min_precision: float,
    mta_max_stretch: float,
    corner_simplify_tol: float,
    corner_thresholds: list[float],
    compute_corners: bool,
) -> dict:
    gt_rings = get_rings_for_role(gt_records, role)
    pred_rings = get_rings_for_role(pred_records, role)
    matches, unmatched_gt, unmatched_pred = role_specific_ring_match(
        gt_rings,
        pred_rings,
        boundary_radius=boundary_radius,
        match_threshold=match_threshold,
    )
    boundary_ious = []
    polis_values = []
    hd95_values = []
    mta_values = []
    corner_stats = {threshold: {"tp": 0, "fp": 0, "fn": 0} for threshold in corner_thresholds}

    for gt_index, pred_index, _ in matches:
        gt_ring = gt_rings[gt_index]
        pred_ring = pred_rings[pred_index]
        boundary_ious.append(boundary_band_iou([gt_ring], [pred_ring], radius=boundary_radius))
        polis_values.append(polis_distance([gt_ring], [pred_ring]))
        hd95_values.append(hd95_distance([gt_ring], [pred_ring], spacing=sample_spacing))
        gt_poly = sanitize_polygon_geometry(Polygon(close_ring(gt_ring)))
        pred_poly = sanitize_polygon_geometry(Polygon(close_ring(pred_ring)))
        if gt_poly is not None and pred_poly is not None:
            mta_values.append(
                mta_distance(
                    pred_poly,
                    gt_poly,
                    min_precision=mta_min_precision,
                    sampling_spacing=sample_spacing,
                    max_stretch=mta_max_stretch,
                )
            )
        if compute_corners:
            gt_corners = simplify_ring_for_corners(gt_ring, corner_simplify_tol)
            pred_corners = simplify_ring_for_corners(pred_ring, corner_simplify_tol)
            for threshold in corner_thresholds:
                corner_tp, corner_fp, corner_fn = match_points(gt_corners, pred_corners, threshold=threshold)
                corner_stats[threshold]["tp"] += corner_tp
                corner_stats[threshold]["fp"] += corner_fp
                corner_stats[threshold]["fn"] += corner_fn

    if compute_corners:
        for gt_index in unmatched_gt:
            gt_corners = simplify_ring_for_corners(gt_rings[gt_index], corner_simplify_tol)
            for threshold in corner_thresholds:
                corner_stats[threshold]["fn"] += gt_corners.shape[0]
        for pred_index in unmatched_pred:
            pred_corners = simplify_ring_for_corners(pred_rings[pred_index], corner_simplify_tol)
            for threshold in corner_thresholds:
                corner_stats[threshold]["fp"] += pred_corners.shape[0]

    return {
        "tp": len(matches),
        "fp": len(unmatched_pred),
        "fn": len(unmatched_gt),
        "boundary_ious": boundary_ious,
        "polis_values": polis_values,
        "hd95_values": hd95_values,
        "mta_values": mta_values,
        "corner_stats": corner_stats,
    }


def init_ap_worker(gt_by_image, pred_by_image):
    global AP_WORKER_CTX
    AP_WORKER_CTX = {
        "gt_by_image": gt_by_image,
        "pred_by_image": pred_by_image,
    }


def compute_ap_for_threshold_worker(threshold: float) -> tuple[float, float]:
    tp, fp, total_gt = greedy_match_detections(
        AP_WORKER_CTX["gt_by_image"],
        AP_WORKER_CTX["pred_by_image"],
        threshold,
    )
    return threshold, compute_average_precision(tp, fp, total_gt)


def init_pairwise_worker(
    gt_by_image,
    pred_by_image,
    pair_match_iou_threshold,
    boundary_radius,
    boundary_match_threshold,
    sample_spacing,
    mta_min_precision,
    mta_max_stretch,
):
    global PAIRWISE_WORKER_CTX
    PAIRWISE_WORKER_CTX = {
        "gt_by_image": gt_by_image,
        "pred_by_image": pred_by_image,
        "pair_match_iou_threshold": pair_match_iou_threshold,
        "boundary_radius": boundary_radius,
        "boundary_match_threshold": boundary_match_threshold,
        "sample_spacing": sample_spacing,
        "mta_min_precision": mta_min_precision,
        "mta_max_stretch": mta_max_stretch,
    }


def compute_pairwise_for_image_worker(image_id: int) -> dict:
    ctx = PAIRWISE_WORKER_CTX
    return pairwise_metrics_for_image(
        ctx["gt_by_image"].get(image_id, []),
        ctx["pred_by_image"].get(image_id, []),
        pair_match_iou_threshold=ctx["pair_match_iou_threshold"],
        boundary_radius=ctx["boundary_radius"],
        boundary_match_threshold=ctx["boundary_match_threshold"],
        sample_spacing=ctx["sample_spacing"],
        mta_min_precision=ctx["mta_min_precision"],
        mta_max_stretch=ctx["mta_max_stretch"],
    )


def init_role_worker(
    gt_by_image,
    pred_by_image,
    role,
    boundary_radius,
    match_threshold,
    sample_spacing,
    mta_min_precision,
    mta_max_stretch,
    corner_simplify_tol,
    corner_thresholds,
    compute_corners,
):
    global ROLE_WORKER_CTX
    ROLE_WORKER_CTX = {
        "gt_by_image": gt_by_image,
        "pred_by_image": pred_by_image,
        "role": role,
        "boundary_radius": boundary_radius,
        "match_threshold": match_threshold,
        "sample_spacing": sample_spacing,
        "mta_min_precision": mta_min_precision,
        "mta_max_stretch": mta_max_stretch,
        "corner_simplify_tol": corner_simplify_tol,
        "corner_thresholds": corner_thresholds,
        "compute_corners": compute_corners,
    }


def compute_role_for_image_worker(image_id: int) -> dict:
    ctx = ROLE_WORKER_CTX
    return role_metrics_for_image(
        ctx["gt_by_image"].get(image_id, []),
        ctx["pred_by_image"].get(image_id, []),
        role=ctx["role"],
        boundary_radius=ctx["boundary_radius"],
        match_threshold=ctx["match_threshold"],
        sample_spacing=ctx["sample_spacing"],
        mta_min_precision=ctx["mta_min_precision"],
        mta_max_stretch=ctx["mta_max_stretch"],
        corner_simplify_tol=ctx["corner_simplify_tol"],
        corner_thresholds=ctx["corner_thresholds"],
        compute_corners=ctx["compute_corners"],
    )


def role_specific_ring_match(
    gt_rings: list[np.ndarray],
    pred_rings: list[np.ndarray],
    boundary_radius: float,
    match_threshold: float,
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    if not gt_rings and not pred_rings:
        return [], [], []
    if not gt_rings:
        return [], [], list(range(len(pred_rings)))
    if not pred_rings:
        return [], list(range(len(gt_rings))), []

    similarities = np.zeros((len(gt_rings), len(pred_rings)), dtype=np.float64)
    for gt_index, gt_ring in enumerate(gt_rings):
        for pred_index, pred_ring in enumerate(pred_rings):
            similarities[gt_index, pred_index] = boundary_band_iou([gt_ring], [pred_ring], radius=boundary_radius)

    row_ind, col_ind = linear_sum_assignment(1.0 - similarities)
    matches = []
    matched_gt = set()
    matched_pred = set()
    for gt_index, pred_index in zip(row_ind, col_ind):
        score = float(similarities[gt_index, pred_index])
        if score >= match_threshold:
            matches.append((gt_index, pred_index, score))
            matched_gt.add(gt_index)
            matched_pred.add(pred_index)

    unmatched_gt = [index for index in range(len(gt_rings)) if index not in matched_gt]
    unmatched_pred = [index for index in range(len(pred_rings)) if index not in matched_pred]
    return matches, unmatched_gt, unmatched_pred


def evaluate_role_ring_metrics(
    gt_by_image: dict[int, list[BuildingRecord]],
    pred_by_image: dict[int, list[BuildingRecord]],
    role: str,
    boundary_radius: float,
    match_threshold: float,
    sample_spacing: float,
    mta_min_precision: float,
    mta_max_stretch: float,
    corner_simplify_tol: float,
    corner_thresholds: list[float],
    compute_corners: bool,
    num_workers: int,
    quiet: bool,
) -> dict:
    boundary_ious = []
    polis_values = []
    hd95_values = []
    mta_values = []
    tp = fp = fn = 0
    corner_stats = {threshold: {"tp": 0, "fp": 0, "fn": 0} for threshold in corner_thresholds}

    all_image_ids = sorted(set(gt_by_image) | set(pred_by_image))
    iterator = (
        tqdm(all_image_ids, desc="Role metrics", leave=False, disable=quiet)
        if all_image_ids
        else []
    )
    if num_workers > 1 and len(all_image_ids) > 1:
        chunksize = max(1, len(all_image_ids) // max(num_workers * 8, 1))
        with ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=mp.get_context("fork"),
            initializer=init_role_worker,
            initargs=(
                gt_by_image,
                pred_by_image,
                role,
                boundary_radius,
                match_threshold,
                sample_spacing,
                mta_min_precision,
                mta_max_stretch,
                corner_simplify_tol,
                corner_thresholds,
                compute_corners,
            ),
        ) as executor:
            iterator = executor.map(compute_role_for_image_worker, all_image_ids, chunksize=chunksize)
            iterator = tqdm(iterator, total=len(all_image_ids), desc="Role metrics", leave=False, disable=quiet)
            for image_result in iterator:
                tp += image_result["tp"]
                fp += image_result["fp"]
                fn += image_result["fn"]
                boundary_ious.extend(image_result["boundary_ious"])
                polis_values.extend(image_result["polis_values"])
                hd95_values.extend(image_result["hd95_values"])
                mta_values.extend(image_result["mta_values"])
                if compute_corners:
                    for threshold in corner_thresholds:
                        corner_stats[threshold]["tp"] += image_result["corner_stats"][threshold]["tp"]
                        corner_stats[threshold]["fp"] += image_result["corner_stats"][threshold]["fp"]
                        corner_stats[threshold]["fn"] += image_result["corner_stats"][threshold]["fn"]
    else:
        for image_id in iterator:
            image_result = role_metrics_for_image(
                gt_by_image.get(image_id, []),
                pred_by_image.get(image_id, []),
                role=role,
                boundary_radius=boundary_radius,
                match_threshold=match_threshold,
                sample_spacing=sample_spacing,
                mta_min_precision=mta_min_precision,
                mta_max_stretch=mta_max_stretch,
                corner_simplify_tol=corner_simplify_tol,
                corner_thresholds=corner_thresholds,
                compute_corners=compute_corners,
            )
            tp += image_result["tp"]
            fp += image_result["fp"]
            fn += image_result["fn"]
            boundary_ious.extend(image_result["boundary_ious"])
            polis_values.extend(image_result["polis_values"])
            hd95_values.extend(image_result["hd95_values"])
            mta_values.extend(image_result["mta_values"])
            if compute_corners:
                for threshold in corner_thresholds:
                    corner_stats[threshold]["tp"] += image_result["corner_stats"][threshold]["tp"]
                    corner_stats[threshold]["fp"] += image_result["corner_stats"][threshold]["fp"]
                    corner_stats[threshold]["fn"] += image_result["corner_stats"][threshold]["fn"]

    result = {
        **build_metric_summary(tp, fp, fn),
        "aggregation": "direct_ring_matching_by_image",
        "boundary_iou": nanmean(boundary_ious),
        "polis": nanmean(polis_values),
        "hd95": nanmean(hd95_values),
        "mta": nanmean(mta_values),
    }
    if compute_corners:
        result["corner_f1"] = {
            str(int(threshold) if float(threshold).is_integer() else threshold): build_metric_summary(
                stats["tp"], stats["fp"], stats["fn"]
            )
            for threshold, stats in corner_stats.items()
        }
    return result


def nanmean(values: list[float]) -> float:
    valid = [value for value in values if value == value]
    return float(np.mean(valid)) if valid else float("nan")


def build_metric_summary(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def build_hisup_building_from_annotation(annotation: dict, score: float) -> BuildingRecord | None:
    segmentations = annotation.get("segmentation", [])
    if not segmentations:
        return None
    exterior = ring_from_flat_coords(segmentations[0])
    if exterior is None:
        return None
    holes = []
    for segment in segmentations[1:]:
        ring = ring_from_flat_coords(segment)
        if ring is not None:
            holes.append(ring)
    geometry = polygon_from_rings(exterior, holes)
    if geometry is None:
        return None
    exterior = np.asarray(geometry.exterior.coords[:-1], dtype=np.float64)
    holes = [np.asarray(interior.coords[:-1], dtype=np.float64) for interior in geometry.interiors]
    return BuildingRecord(
        image_id=int(annotation["image_id"]),
        geometry=geometry,
        exterior=exterior,
        holes=holes,
        score=float(score),
        source_indices=[int(annotation.get("id", -1))],
    )


def load_hisup_gt(path: Path, selected_image_ids: set[int] | None = None) -> tuple[dict[int, list[BuildingRecord]], dict]:
    data = read_json(path)
    grouped = defaultdict(list)
    for annotation in data["annotations"]:
        image_id = int(annotation["image_id"])
        if selected_image_ids is not None and image_id not in selected_image_ids:
            continue
        record = build_hisup_building_from_annotation(annotation, score=1.0)
        if record is not None:
            grouped[record.image_id].append(record)
    return grouped, {
        "num_images": len(data.get("images", [])),
        "num_annotations": len(data.get("annotations", [])),
        "schema": "hisup_gt",
    }


def load_hisup_predictions(
    path: Path,
    score_threshold: float,
    selected_image_ids: set[int] | None = None,
) -> tuple[dict[int, list[BuildingRecord]], dict]:
    data = read_json(path)
    grouped = defaultdict(list)
    kept = 0
    for pred_index, annotation in enumerate(data):
        image_id = int(annotation["image_id"])
        if selected_image_ids is not None and image_id not in selected_image_ids:
            continue
        score = float(annotation.get("score", 0.0))
        if score < score_threshold:
            continue
        record = build_hisup_building_from_annotation(annotation, score=score)
        if record is None:
            continue
        record.source_indices = [pred_index]
        grouped[record.image_id].append(record)
        kept += 1
    return grouped, {
        "num_predictions": len(data),
        "num_kept_predictions": kept,
        "schema": "hisup_prediction",
    }


def roipoly_group_key(annotation: dict) -> tuple:
    if annotation.get("source_geojson") is None or annotation.get("source_geometry_index") is None:
        raise ValueError(
            "RoIPoly GT annotation is missing `source_geojson` or `source_geometry_index`; "
            "cannot safely reconstruct building-level polygons."
        )
    return (
        int(annotation["image_id"]),
        annotation.get("source_geojson"),
        annotation.get("source_geometry_index"),
    )


def assign_holes_to_exteriors(
    exteriors: list[np.ndarray],
    holes: list[np.ndarray],
    containment_threshold: float,
) -> list[list[np.ndarray]]:
    exterior_polygons = [sanitize_polygon_geometry(Polygon(close_ring(ring))) for ring in exteriors]
    attached = [[] for _ in exteriors]
    for hole_ring in holes:
        hole_geom = sanitize_polygon_geometry(Polygon(close_ring(hole_ring)))
        if hole_geom is None:
            continue
        candidates = []
        anchor = hole_geom.representative_point()
        for exterior_index, exterior_geom in enumerate(exterior_polygons):
            if exterior_geom is None:
                continue
            cover_ratio = exterior_geom.intersection(hole_geom).area / max(hole_geom.area, 1e-8)
            if exterior_geom.covers(anchor) and cover_ratio >= containment_threshold:
                candidates.append((exterior_geom.area, exterior_index))
        if candidates:
            _, target_index = min(candidates, key=lambda item: item[0])
            attached[target_index].append(hole_ring)
    return attached


def load_roipoly_gt(
    path: Path,
    containment_threshold: float,
    selected_image_ids: set[int] | None = None,
) -> tuple[dict[int, list[BuildingRecord]], dict]:
    data = read_json(path)
    grouped_by_building = defaultdict(list)
    for annotation in data["annotations"]:
        image_id = int(annotation["image_id"])
        if selected_image_ids is not None and image_id not in selected_image_ids:
            continue
        grouped_by_building[roipoly_group_key(annotation)].append(annotation)

    grouped_by_image = defaultdict(list)
    for group_items in grouped_by_building.values():
        exteriors = []
        holes = []
        image_id = int(group_items[0]["image_id"])
        source_indices = []
        for annotation in group_items:
            segment = annotation.get("segmentation", [])
            if not segment:
                continue
            ring = ring_from_flat_coords(segment[0])
            if ring is None:
                continue
            source_indices.append(int(annotation.get("id", -1)))
            if bool(annotation.get("is_hole", False)) or annotation.get("ring_role") == "hole":
                holes.append(ring)
            else:
                exteriors.append(ring)
        attached_holes = assign_holes_to_exteriors(
            exteriors,
            holes,
            containment_threshold=containment_threshold,
        )
        for exterior, exterior_holes in zip(exteriors, attached_holes):
            geometry = polygon_from_rings(exterior, exterior_holes)
            if geometry is None:
                continue
            grouped_by_image[image_id].append(
                BuildingRecord(
                    image_id=image_id,
                    geometry=geometry,
                    exterior=np.asarray(geometry.exterior.coords[:-1], dtype=np.float64),
                    holes=[np.asarray(interior.coords[:-1], dtype=np.float64) for interior in geometry.interiors],
                    score=1.0,
                    source_indices=source_indices,
                )
            )

    return grouped_by_image, {
        "num_images": len(data.get("images", [])),
        "num_annotations": len(data.get("annotations", [])),
        "num_grouped_buildings": sum(len(items) for items in grouped_by_image.values()),
        "schema": "roipoly_gt",
    }


def load_roipoly_predictions(
    path: Path,
    score_threshold: float,
    min_area: float,
    dedup_iou_threshold: float,
    dedup_overlap_threshold: float,
    containment_threshold: float,
    selected_image_ids: set[int] | None = None,
) -> tuple[dict[int, list[BuildingRecord]], dict]:
    predictions = read_json(path)
    rings_by_image = defaultdict(list)
    skipped_low_score = 0
    skipped_invalid = 0

    for pred_index, prediction in enumerate(predictions):
        image_id = int(prediction["image_id"])
        if selected_image_ids is not None and image_id not in selected_image_ids:
            continue
        score = float(prediction.get("score", 0.0))
        if score < score_threshold:
            skipped_low_score += 1
            continue
        segmentation = prediction.get("segmentation", [])
        if not segmentation:
            skipped_invalid += 1
            continue
        ring = ring_from_flat_coords(segmentation[0])
        if ring is None:
            skipped_invalid += 1
            continue
        polygon = sanitize_polygon_geometry(Polygon(close_ring(ring)))
        if polygon is None or polygon.area < min_area:
            skipped_invalid += 1
            continue
        rings_by_image[image_id].append(
            {
                "prediction_index": pred_index,
                "geometry": polygon,
                "score": score,
                "ring": np.asarray(polygon.exterior.coords[:-1], dtype=np.float64),
            }
        )

    buildings_by_image = defaultdict(list)
    num_deduped_rings = 0
    for image_id, items in rings_by_image.items():
        deduped = deduplicate_roipoly_patch_rings(
            items,
            iou_threshold=dedup_iou_threshold,
            overlap_threshold=dedup_overlap_threshold,
            containment_threshold=containment_threshold,
        )
        num_deduped_rings += len(deduped)
        parents = infer_parent_indices(deduped, containment_threshold=containment_threshold)
        depths = compute_depths(parents)
        children_by_parent = defaultdict(list)
        for child_index, parent_index in enumerate(parents):
            if parent_index is not None:
                children_by_parent[parent_index].append(child_index)

        for index, item in enumerate(deduped):
            if depths[index] % 2 == 1:
                continue
            hole_indices = [child for child in children_by_parent[index] if depths[child] % 2 == 1]
            hole_rings = [deduped[child]["ring"] for child in hole_indices]
            exterior = item["ring"]
            geometry = polygon_from_rings(exterior, hole_rings)
            if geometry is None:
                geometry = item["geometry"]
            buildings_by_image[image_id].append(
                BuildingRecord(
                    image_id=image_id,
                    geometry=geometry,
                    exterior=np.asarray(geometry.exterior.coords[:-1], dtype=np.float64),
                    holes=[np.asarray(interior.coords[:-1], dtype=np.float64) for interior in geometry.interiors],
                    score=float(item["score"]),
                    source_indices=[int(item["prediction_index"])],
                )
            )

    return buildings_by_image, {
        "num_predictions": len(predictions),
        "num_kept_rings": sum(len(items) for items in rings_by_image.values()),
        "num_deduped_rings": num_deduped_rings,
        "num_grouped_buildings": sum(len(items) for items in buildings_by_image.values()),
        "num_skipped_low_score": skipped_low_score,
        "num_skipped_invalid": skipped_invalid,
        "schema": "roipoly_prediction",
    }


def infer_dataset_type(gt_json: dict) -> str:
    annotations = gt_json.get("annotations", [])
    if not annotations:
        raise ValueError("Ground-truth annotation file contains no annotations.")
    sample = annotations[0]
    if "ring_role" in sample or "is_hole" in sample:
        return "roipoly"
    if "has_hole" in sample or "hole_count" in sample:
        return "hisup"
    # Deventer/HiSup-style COCO exports may omit explicit hole flags for
    # single-class datasets while still using the same polygon segmentation
    # schema consumed by the HiSup loader.
    if "segmentation" in sample and "bbox" in sample:
        return "hisup"
    raise ValueError("Could not infer dataset type from GT annotations.")


def evaluate_pairwise_metrics(
    gt_by_image: dict[int, list[BuildingRecord]],
    pred_by_image: dict[int, list[BuildingRecord]],
    pair_match_iou_threshold: float,
    boundary_radius: float,
    boundary_match_threshold: float,
    sample_spacing: float,
    mta_min_precision: float,
    mta_max_stretch: float,
    corner_simplify_tol: float,
    corner_thresholds: list[float],
    num_workers: int,
    quiet: bool,
) -> dict:
    overall_boundary_ious = []
    overall_polis = []
    overall_hd95 = []
    overall_mta = []
    topology_exact = []

    matched_buildings = 0
    gt_building_total = sum(len(records) for records in gt_by_image.values())
    pred_building_total = sum(len(records) for records in pred_by_image.values())

    all_image_ids = sorted(set(gt_by_image) | set(pred_by_image))
    iterator = (
        tqdm(all_image_ids, desc="Pairwise metrics", leave=False, disable=quiet)
        if all_image_ids
        else []
    )
    if num_workers > 1 and len(all_image_ids) > 1:
        chunksize = max(1, len(all_image_ids) // max(num_workers * 8, 1))
        with ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=mp.get_context("fork"),
            initializer=init_pairwise_worker,
            initargs=(
                gt_by_image,
                pred_by_image,
                pair_match_iou_threshold,
                boundary_radius,
                boundary_match_threshold,
                sample_spacing,
                mta_min_precision,
                mta_max_stretch,
            ),
        ) as executor:
            iterator = executor.map(compute_pairwise_for_image_worker, all_image_ids, chunksize=chunksize)
            iterator = tqdm(iterator, total=len(all_image_ids), desc="Pairwise metrics", leave=False, disable=quiet)
            for image_result in iterator:
                matched_buildings += image_result["matched_buildings"]
                overall_boundary_ious.extend(image_result["overall_boundary_ious"])
                overall_polis.extend(image_result["overall_polis"])
                overall_hd95.extend(image_result["overall_hd95"])
                overall_mta.extend(image_result["overall_mta"])
                topology_exact.extend([0] * (image_result["topology_count"] - image_result["matched_buildings"]))
                topology_exact.extend([1] * image_result["topology_sum"])
                topology_exact.extend([0] * (image_result["matched_buildings"] - image_result["topology_sum"]))
    else:
        for image_id in iterator:
            image_result = pairwise_metrics_for_image(
                gt_by_image.get(image_id, []),
                pred_by_image.get(image_id, []),
                pair_match_iou_threshold=pair_match_iou_threshold,
                boundary_radius=boundary_radius,
                boundary_match_threshold=boundary_match_threshold,
                sample_spacing=sample_spacing,
                mta_min_precision=mta_min_precision,
                mta_max_stretch=mta_max_stretch,
            )
            matched_buildings += image_result["matched_buildings"]
            overall_boundary_ious.extend(image_result["overall_boundary_ious"])
            overall_polis.extend(image_result["overall_polis"])
            overall_hd95.extend(image_result["overall_hd95"])
            overall_mta.extend(image_result["overall_mta"])
            topology_exact.extend([0] * (image_result["topology_count"] - image_result["matched_buildings"]))
            topology_exact.extend([1] * image_result["topology_sum"])
            topology_exact.extend([0] * (image_result["matched_buildings"] - image_result["topology_sum"]))
    exterior_metrics = evaluate_role_ring_metrics(
        gt_by_image=gt_by_image,
        pred_by_image=pred_by_image,
        role="exterior",
        boundary_radius=boundary_radius,
        match_threshold=boundary_match_threshold,
        sample_spacing=sample_spacing,
        mta_min_precision=mta_min_precision,
        mta_max_stretch=mta_max_stretch,
        corner_simplify_tol=corner_simplify_tol,
        corner_thresholds=corner_thresholds,
        compute_corners=True,
        num_workers=num_workers,
        quiet=quiet,
    )
    hole_metrics = evaluate_role_ring_metrics(
        gt_by_image=gt_by_image,
        pred_by_image=pred_by_image,
        role="hole",
        boundary_radius=boundary_radius,
        match_threshold=boundary_match_threshold,
        sample_spacing=sample_spacing,
        mta_min_precision=mta_min_precision,
        mta_max_stretch=mta_max_stretch,
        corner_simplify_tol=corner_simplify_tol,
        corner_thresholds=corner_thresholds,
        compute_corners=False,
        num_workers=num_workers,
        quiet=quiet,
    )

    return {
        "meta": {
            "gt_building_total": gt_building_total,
            "pred_building_total": pred_building_total,
            "matched_building_pairs": matched_buildings,
        },
        "overall": {
            "aggregation": "matched_building_pairs_only",
            "boundary_iou": nanmean(overall_boundary_ious),
            "polis": nanmean(overall_polis),
            "hd95": nanmean(overall_hd95),
            "mta": nanmean(overall_mta),
            "topology_exact_match_rate": float(np.mean(topology_exact)) if topology_exact else float("nan"),
        },
        "exterior": exterior_metrics,
        "hole": hole_metrics,
    }


def compute_hole_count_mae(
    gt_by_image: dict[int, list[BuildingRecord]],
    pred_by_image: dict[int, list[BuildingRecord]],
    pair_match_iou_threshold: float,
) -> float:
    diffs = []
    all_image_ids = set(gt_by_image) | set(pred_by_image)
    for image_id in all_image_ids:
        gt_records = gt_by_image.get(image_id, [])
        pred_records = pred_by_image.get(image_id, [])
        matches, unmatched_gt, unmatched_pred = match_buildings(gt_records, pred_records, pair_match_iou_threshold)
        for gt_index, pred_index, _ in matches:
            diffs.append(abs(len(gt_records[gt_index].holes) - len(pred_records[pred_index].holes)))
    return float(np.mean(diffs)) if diffs else float("nan")


def to_jsonable(value):
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def evaluate_poly_ap(
    gt_by_image: dict[int, list[BuildingRecord]],
    pred_by_image: dict[int, list[BuildingRecord]],
    thresholds: list[float],
    num_workers: int,
    quiet: bool,
) -> dict[str, float]:
    values = {}
    ap_values = []
    if num_workers > 1 and len(thresholds) > 1:
        with ProcessPoolExecutor(
            max_workers=min(num_workers, len(thresholds)),
            mp_context=mp.get_context("fork"),
            initializer=init_ap_worker,
            initargs=(gt_by_image, pred_by_image),
        ) as executor:
            iterator = executor.map(compute_ap_for_threshold_worker, thresholds, chunksize=1)
            iterator = tqdm(iterator, total=len(thresholds), desc="Poly AP", leave=False, disable=quiet)
            for threshold, ap in iterator:
                values[f"AP@{threshold:.2f}"] = ap
                ap_values.append(ap)
    else:
        iterator = tqdm(thresholds, total=len(thresholds), desc="Poly AP", leave=False, disable=quiet)
        for threshold in iterator:
            tp, fp, total_gt = greedy_match_detections(gt_by_image, pred_by_image, threshold)
            ap = compute_average_precision(tp, fp, total_gt)
            values[f"AP@{threshold:.2f}"] = ap
            ap_values.append(ap)
    values["AP"] = float(np.mean(ap_values)) if ap_values else float("nan")
    if "AP@0.50" in values:
        values["AP50"] = values["AP@0.50"]
    if "AP@0.75" in values:
        values["AP75"] = values["AP@0.75"]
    return values


def evaluate_dataset(
    gt_by_image: dict[int, list[BuildingRecord]],
    pred_by_image: dict[int, list[BuildingRecord]],
    args: argparse.Namespace,
) -> dict:
    ap_thresholds = parse_float_list(args.ap_thresholds)
    corner_thresholds = parse_float_list(args.corner_thresholds)

    poly_ap = evaluate_poly_ap(
        gt_by_image,
        pred_by_image,
        thresholds=ap_thresholds,
        num_workers=args.num_workers,
        quiet=args.quiet,
    )
    pair_metrics = evaluate_pairwise_metrics(
        gt_by_image=gt_by_image,
        pred_by_image=pred_by_image,
        pair_match_iou_threshold=args.pair_match_iou_threshold,
        boundary_radius=args.boundary_radius,
        boundary_match_threshold=args.boundary_match_threshold,
        sample_spacing=args.sample_spacing,
        mta_min_precision=args.mta_min_precision,
        mta_max_stretch=args.mta_max_stretch,
        corner_simplify_tol=args.corner_simplify_tol,
        corner_thresholds=corner_thresholds,
        num_workers=args.num_workers,
        quiet=args.quiet,
    )
    pair_metrics["hole"]["count_mae"] = compute_hole_count_mae(
        gt_by_image,
        pred_by_image,
        pair_match_iou_threshold=args.pair_match_iou_threshold,
    )
    pair_metrics["hole"]["count_mae_aggregation"] = "matched_building_pairs_only"

    return {
        "poly_ap": poly_ap,
        **pair_metrics,
    }


def maybe_limit_images(
    gt_by_image: dict[int, list[BuildingRecord]],
    pred_by_image: dict[int, list[BuildingRecord]],
    image_ids: set[int] | None,
    max_images: int | None,
) -> tuple[dict[int, list[BuildingRecord]], dict[int, list[BuildingRecord]], list[int]]:
    selected_ids = sorted(set(gt_by_image) | set(pred_by_image))
    if image_ids:
        selected_lookup = set(image_ids)
        selected_ids = [image_id for image_id in selected_ids if image_id in selected_lookup]
    if max_images is not None and max_images > 0:
        selected_ids = selected_ids[:max_images]
    if not image_ids and (max_images is None or max_images <= 0):
        return gt_by_image, pred_by_image, selected_ids
    selected = set(selected_ids)
    return (
        {image_id: records for image_id, records in gt_by_image.items() if image_id in selected},
        {image_id: records for image_id, records in pred_by_image.items() if image_id in selected},
        selected_ids,
    )


def apply_min_hole_area_filter(
    records_by_image: dict[int, list[BuildingRecord]],
    min_hole_area: float,
) -> tuple[dict[int, list[BuildingRecord]], dict]:
    if min_hole_area <= 0:
        hole_total = sum(len(record.holes) for records in records_by_image.values() for record in records)
        return records_by_image, {
            "min_hole_area": float(min_hole_area),
            "holes_before": int(hole_total),
            "holes_after": int(hole_total),
            "holes_removed": 0,
            "buildings_changed": 0,
        }

    filtered_by_image = {}
    holes_before = 0
    holes_after = 0
    buildings_changed = 0

    for image_id, records in records_by_image.items():
        filtered_records = []
        for record in records:
            holes_before += len(record.holes)
            kept_holes = [hole for hole in record.holes if ring_polygon_area(hole) >= min_hole_area]
            holes_after += len(kept_holes)
            if len(kept_holes) != len(record.holes):
                buildings_changed += 1
            if len(kept_holes) == len(record.holes):
                filtered_records.append(record)
                continue
            geometry = polygon_from_rings(record.exterior, kept_holes)
            if geometry is None:
                geometry = polygon_from_rings(record.exterior, [])
            if geometry is None:
                continue
            filtered_records.append(
                BuildingRecord(
                    image_id=record.image_id,
                    geometry=geometry,
                    exterior=np.asarray(geometry.exterior.coords[:-1], dtype=np.float64),
                    holes=[np.asarray(interior.coords[:-1], dtype=np.float64) for interior in geometry.interiors],
                    score=record.score,
                    source_indices=list(record.source_indices),
                )
            )
        filtered_by_image[image_id] = filtered_records

    return filtered_by_image, {
        "min_hole_area": float(min_hole_area),
        "holes_before": int(holes_before),
        "holes_after": int(holes_after),
        "holes_removed": int(holes_before - holes_after),
        "buildings_changed": int(buildings_changed),
    }


def load_inputs(args: argparse.Namespace):
    gt_json = read_json(args.gt)
    inferred_gt_type = infer_dataset_type(gt_json)
    gt_type = args.gt_type or args.dataset_type
    pred_type = args.pred_type or args.dataset_type
    if gt_type == "auto":
        gt_type = inferred_gt_type
    if pred_type == "auto":
        pred_type = gt_type
    selected_image_ids = set(parse_int_list(args.image_ids)) if args.image_ids.strip() else None

    if gt_type == "hisup":
        gt_by_image, gt_meta = load_hisup_gt(args.gt, selected_image_ids=selected_image_ids)
    elif gt_type == "roipoly":
        gt_by_image, gt_meta = load_roipoly_gt(
            args.gt,
            containment_threshold=args.roipoly_containment_threshold,
            selected_image_ids=selected_image_ids,
        )
    else:  # pragma: no cover
        raise ValueError(f"Unsupported GT type: {gt_type}")

    if pred_type == "hisup":
        pred_by_image, pred_meta = load_hisup_predictions(
            args.pred,
            score_threshold=args.score_threshold,
            selected_image_ids=selected_image_ids,
        )
    elif pred_type == "roipoly":
        pred_by_image, pred_meta = load_roipoly_predictions(
            args.pred,
            score_threshold=args.score_threshold,
            min_area=args.roipoly_min_area,
            dedup_iou_threshold=args.roipoly_dedup_iou_threshold,
            dedup_overlap_threshold=args.roipoly_dedup_overlap_threshold,
            containment_threshold=args.roipoly_containment_threshold,
            selected_image_ids=selected_image_ids,
        )
    else:  # pragma: no cover
        raise ValueError(f"Unsupported prediction type: {pred_type}")
    gt_by_image, pred_by_image, selected_ids = maybe_limit_images(
        gt_by_image,
        pred_by_image,
        selected_image_ids,
        args.max_images,
    )
    gt_by_image, gt_hole_filter = apply_min_hole_area_filter(gt_by_image, args.min_hole_area)
    pred_by_image, pred_hole_filter = apply_min_hole_area_filter(pred_by_image, args.min_hole_area)
    gt_meta = {
        **gt_meta,
        "hole_area_filter": gt_hole_filter,
    }
    pred_meta = {
        **pred_meta,
        "hole_area_filter": pred_hole_filter,
    }
    if selected_image_ids or (args.max_images is not None and args.max_images > 0):
        gt_meta = {
            **gt_meta,
            "filtered_image_count": len(gt_by_image),
            "selected_image_ids": selected_ids,
        }
        pred_meta = {
            **pred_meta,
            "filtered_image_count": len(pred_by_image),
            "selected_image_ids": selected_ids,
        }
    return gt_type, pred_type, gt_by_image, pred_by_image, gt_meta, pred_meta


def make_output_path(args: argparse.Namespace, gt_type: str, pred_type: str) -> Path:
    if args.output is not None:
        return args.output
    stem = args.pred.stem
    type_suffix = gt_type if gt_type == pred_type else f"gt-{gt_type}.pred-{pred_type}"
    hole_suffix = (
        f".holearea{int(args.min_hole_area) if float(args.min_hole_area).is_integer() else args.min_hole_area:g}"
        if args.min_hole_area > 0
        else ""
    )
    suffix = f".first{args.max_images}" if args.max_images is not None and args.max_images > 0 else ""
    return args.pred.with_name(f"{stem}.vector_metrics.{type_suffix}{hole_suffix}{suffix}.json")


def main() -> None:
    args = parse_args()
    gt_type, pred_type, gt_by_image, pred_by_image, gt_meta, pred_meta = load_inputs(args)
    log(
        f"[vector-eval] gt_type={gt_type} pred_type={pred_type} gt_images={len(gt_by_image)} pred_images={len(pred_by_image)}",
        args.quiet,
    )
    metrics = evaluate_dataset(gt_by_image, pred_by_image, args)
    result = {
        "config": {
            "dataset_type": gt_type if gt_type == pred_type else None,
            "gt_type": gt_type,
            "pred_type": pred_type,
            "pred": str(args.pred),
            "gt": str(args.gt),
            "score_threshold": args.score_threshold,
            "pair_match_iou_threshold": args.pair_match_iou_threshold,
            "boundary_radius": args.boundary_radius,
            "boundary_match_threshold": args.boundary_match_threshold,
            "sample_spacing": args.sample_spacing,
            "mta_min_precision": args.mta_min_precision,
            "mta_max_stretch": args.mta_max_stretch,
            "corner_simplify_tol": args.corner_simplify_tol,
            "corner_thresholds": parse_float_list(args.corner_thresholds),
            "ap_thresholds": parse_float_list(args.ap_thresholds),
            "roipoly_dedup_iou_threshold": args.roipoly_dedup_iou_threshold,
            "roipoly_dedup_overlap_threshold": args.roipoly_dedup_overlap_threshold,
            "roipoly_containment_threshold": args.roipoly_containment_threshold,
            "roipoly_min_area": args.roipoly_min_area,
            "min_hole_area": args.min_hole_area,
            "image_ids": parse_int_list(args.image_ids),
            "max_images": args.max_images,
        },
        "ground_truth": gt_meta,
        "predictions": pred_meta,
        "metrics": metrics,
    }
    result = to_jsonable(result)

    output_path = make_output_path(args, gt_type, pred_type)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log(f"[vector-eval] wrote metrics to {output_path}", args.quiet)
    if not args.quiet:
        print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
