from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from pycocotools.coco import COCO
from skimage import io
from skimage.transform import resize
from torch.utils.data import Dataset


def _drop_redundant_ring_points(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points
    deduped = [points[0]]
    for point in points[1:]:
        if not np.array_equal(point, deduped[-1]):
            deduped.append(point)
    deduped = np.asarray(deduped, dtype=np.int64)
    if len(deduped) > 1 and np.array_equal(deduped[0], deduped[-1]):
        deduped = deduped[:-1]
    return deduped


def _ring_area_abs(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    x = points[:, 1].astype(np.float64)
    y = points[:, 0].astype(np.float64)
    return float(0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _sample_ring(points: np.ndarray, keep_vertices: int) -> np.ndarray:
    if keep_vertices >= len(points):
        return points
    indices = np.floor(np.linspace(0, len(points), num=keep_vertices, endpoint=False)).astype(np.int64)
    indices = np.clip(indices, 0, len(points) - 1)
    sampled = points[indices]
    sampled = _drop_redundant_ring_points(sampled)
    return sampled if len(sampled) >= 3 else points[:keep_vertices]


def _select_target_rings(segmentation: list[list[float]], target_rings: str) -> list[list[float]]:
    if target_rings == "all":
        return list(segmentation)
    if target_rings == "exterior":
        return [segmentation[0]] if segmentation else []
    raise ValueError(f"Unsupported target_rings mode: {target_rings}")


def _scale_ring_to_window(flat_ring, original_height: int, original_width: int, window_size: int) -> np.ndarray | None:
    ring = np.asarray(flat_ring, dtype=np.float32).reshape(-1, 2)
    if len(ring) < 3:
        return None
    cols = ring[:, 0] * (window_size / float(original_width))
    rows = ring[:, 1] * (window_size / float(original_height))
    points = np.stack([rows, cols], axis=1)
    points = np.rint(points).astype(np.int64)
    points[:, 0] = np.clip(points[:, 0], 0, window_size - 1)
    points[:, 1] = np.clip(points[:, 1], 0, window_size - 1)
    points = _drop_redundant_ring_points(points)
    return points if len(points) >= 3 else None


@dataclass
class RingEncodingStats:
    original_ring_count: int
    kept_ring_count: int
    dropped_ring_count: int
    original_vertex_count: int
    kept_vertex_count: int


def _allocate_ring_budgets(rings: list[np.ndarray], max_points: int, min_ring_vertices: int = 3) -> list[np.ndarray]:
    if not rings:
        return []

    max_rings = max(1, max_points // min_ring_vertices)
    ranked = sorted(
        rings,
        key=lambda ring: (_ring_area_abs(ring), len(ring)),
        reverse=True,
    )[:max_rings]

    budgets = [min(len(ring), min_ring_vertices) for ring in ranked]
    extra_need = [max(0, len(ring) - budget) for ring, budget in zip(ranked, budgets)]
    remaining = max_points - sum(budgets)
    order = sorted(
        range(len(ranked)),
        key=lambda index: (extra_need[index], _ring_area_abs(ranked[index]), len(ranked[index])),
        reverse=True,
    )

    while remaining > 0 and any(need > 0 for need in extra_need):
        progressed = False
        for index in order:
            if extra_need[index] <= 0 or remaining <= 0:
                continue
            budgets[index] += 1
            extra_need[index] -= 1
            remaining -= 1
            progressed = True
        if not progressed:
            break

    simplified = []
    for ring, budget in zip(ranked, budgets):
        sampled = _sample_ring(ring, budget)
        sampled = _drop_redundant_ring_points(sampled)
        if len(sampled) >= 3:
            simplified.append(sampled)
    return simplified


def _build_gaussian_kernel(radius: int, sigma: float) -> np.ndarray:
    if radius <= 0:
        return np.ones((1, 1), dtype=np.float32)
    sigma = float(sigma) if sigma > 0 else max(1.0, radius / 2.0)
    offsets = np.arange(-radius, radius + 1, dtype=np.float32)
    grid_y, grid_x = np.meshgrid(offsets, offsets, indexing="ij")
    kernel = np.exp(-(grid_x ** 2 + grid_y ** 2) / (2.0 * sigma * sigma)).astype(np.float32)
    kernel /= kernel.max()
    return kernel


def _stamp_gaussian(heatmap: np.ndarray, rows: np.ndarray, cols: np.ndarray, kernel: np.ndarray) -> None:
    if kernel.shape == (1, 1):
        heatmap[0, rows, cols] = np.maximum(heatmap[0, rows, cols], 1.0)
        return
    radius = kernel.shape[0] // 2
    H, W = heatmap.shape[1], heatmap.shape[2]
    for r, c in zip(rows, cols):
        r0, r1 = r - radius, r + radius + 1
        c0, c1 = c - radius, c + radius + 1
        k_r0 = max(0, -r0); k_r1 = kernel.shape[0] - max(0, r1 - H)
        k_c0 = max(0, -c0); k_c1 = kernel.shape[1] - max(0, c1 - W)
        r0 = max(0, r0); r1 = min(H, r1)
        c0 = max(0, c0); c1 = min(W, c1)
        if r1 <= r0 or c1 <= c0:
            continue
        sub = kernel[k_r0:k_r1, k_c0:k_c1]
        heatmap[0, r0:r1, c0:c1] = np.maximum(heatmap[0, r0:r1, c0:c1], sub)


def encode_multiring_targets(
    segmentations: list[list[list[float]]],
    original_height: int,
    original_width: int,
    window_size: int = 320,
    max_points: int = 256,
    target_rings: str = "all",
    heatmap_radius: int = 0,
    heatmap_sigma: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, RingEncodingStats]:
    scaled_rings = []
    original_ring_count = 0
    original_vertex_count = 0
    for segmentation in segmentations:
        for ring in _select_target_rings(segmentation, target_rings):
            original_ring_count += 1
            original_vertex_count += max(0, len(ring) // 2 - 1)
            scaled = _scale_ring_to_window(ring, original_height, original_width, window_size)
            if scaled is not None:
                scaled_rings.append(scaled)

    kept_rings = _allocate_ring_budgets(scaled_rings, max_points=max_points)
    kept_vertex_count = int(sum(len(ring) for ring in kept_rings))

    heatmap = np.zeros((1, window_size, window_size), dtype=np.float32)
    graph = np.zeros((max_points, 2), dtype=np.int64)
    perm_target = np.zeros((max_points, max_points), dtype=np.float32)
    perm_mask = np.zeros((max_points, max_points), dtype=bool)
    valid_mask = np.zeros((max_points,), dtype=bool)

    np.fill_diagonal(perm_target, 1.0)
    np.fill_diagonal(perm_mask, True)

    kernel = _build_gaussian_kernel(heatmap_radius, heatmap_sigma)

    cursor = 0
    for ring in kept_rings:
        ring_len = len(ring)
        if ring_len < 3 or cursor + ring_len > max_points:
            continue
        ring_indices = np.arange(cursor, cursor + ring_len)
        graph[ring_indices] = ring
        valid_mask[ring_indices] = True
        _stamp_gaussian(heatmap, ring[:, 0], ring[:, 1], kernel)

        perm_target[ring_indices, ring_indices] = 0.0
        perm_mask[np.ix_(ring_indices, ring_indices)] = True
        for offset, source_index in enumerate(ring_indices):
            target_index = ring_indices[(offset + 1) % ring_len]
            perm_target[source_index, target_index] = 1.0
        cursor += ring_len

    if cursor > 0:
        # Supervise all valid-to-valid edges, not only within-ring blocks, so
        # cross-ring connections are explicitly penalized as negatives.
        perm_mask[:cursor, :cursor] = True

    stats = RingEncodingStats(
        original_ring_count=original_ring_count,
        kept_ring_count=len(kept_rings),
        dropped_ring_count=max(0, original_ring_count - len(kept_rings)),
        original_vertex_count=original_vertex_count,
        kept_vertex_count=int(valid_mask.sum()),
    )
    return (
        torch.from_numpy(heatmap),
        torch.from_numpy(graph),
        torch.from_numpy(perm_target),
        torch.from_numpy(perm_mask),
        torch.from_numpy(valid_mask),
        stats,
    )


_D4_TRANSFORMS = ("identity", "rot90", "rot180", "rot270", "flip_h", "flip_v", "flip_diag", "flip_anti")


def _apply_d4_image(image: np.ndarray, mode: str) -> np.ndarray:
    if mode == "identity":
        return image
    if mode == "rot90":
        return np.rot90(image, k=1).copy()
    if mode == "rot180":
        return np.rot90(image, k=2).copy()
    if mode == "rot270":
        return np.rot90(image, k=3).copy()
    if mode == "flip_h":
        return np.ascontiguousarray(image[:, ::-1, :])
    if mode == "flip_v":
        return np.ascontiguousarray(image[::-1, :, :])
    if mode == "flip_diag":
        return np.ascontiguousarray(np.transpose(image, (1, 0, 2)))
    if mode == "flip_anti":
        transposed = np.transpose(image, (1, 0, 2))
        return np.ascontiguousarray(transposed[::-1, ::-1, :])
    raise ValueError(f"Unknown D4 mode: {mode}")


def _apply_d4_ring(points: np.ndarray, mode: str, h: int, w: int) -> np.ndarray:
    """points in (row, col) integer coords inside window of size (h, w)."""
    if mode == "identity":
        return points
    r = points[:, 0]
    c = points[:, 1]
    if mode == "rot90":
        # np.rot90(image, k=1): (r, c) -> (w-1-c, r); new shape (w, h)
        new = np.stack([w - 1 - c, r], axis=1)
    elif mode == "rot180":
        new = np.stack([h - 1 - r, w - 1 - c], axis=1)
    elif mode == "rot270":
        new = np.stack([c, h - 1 - r], axis=1)
    elif mode == "flip_h":
        new = np.stack([r, w - 1 - c], axis=1)
    elif mode == "flip_v":
        new = np.stack([h - 1 - r, c], axis=1)
    elif mode == "flip_diag":
        new = np.stack([c, r], axis=1)
    elif mode == "flip_anti":
        new = np.stack([w - 1 - c, h - 1 - r], axis=1)
    else:
        raise ValueError(f"Unknown D4 mode: {mode}")
    return new.astype(np.int64)


class HiSupMultiringTrainDataset(Dataset):
    def __init__(
        self,
        images_directory,
        annotations_path,
        window_size: int = 320,
        max_points: int = 256,
        image_limit: int | None = None,
        target_rings: str = "all",
        heatmap_radius: int = 0,
        heatmap_sigma: float = 0.0,
        use_d4_aug: bool = False,
    ):
        self.images_directory = Path(images_directory)
        self.annotations_path = Path(annotations_path)
        self.window_size = int(window_size)
        self.max_points = int(max_points)
        if target_rings not in {"all", "exterior"}:
            raise ValueError(f"Unsupported target_rings mode: {target_rings}")
        self.target_rings = target_rings
        self.heatmap_radius = int(heatmap_radius)
        self.heatmap_sigma = float(heatmap_sigma)
        self.use_d4_aug = bool(use_d4_aug)

        self.coco = COCO(str(self.annotations_path))
        self.image_ids = sorted(self.coco.getImgIds())
        if image_limit is not None:
            self.image_ids = self.image_ids[:image_limit]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = self.images_directory / image_info["file_name"]
        image = io.imread(str(image_path))
        original_height, original_width = image.shape[:2]
        image = resize(
            image,
            (self.window_size, self.window_size, 3),
            anti_aliasing=True,
            preserve_range=True,
        )
        image_np = np.asarray(image, dtype=np.float32)

        annotation_ids = self.coco.getAnnIds(imgIds=[image_id])
        annotations = self.coco.loadAnns(annotation_ids)
        segmentations = [annotation.get("segmentation", []) for annotation in annotations]
        heatmap, graph, perm_target, perm_mask, valid_vertex_mask, stats = encode_multiring_targets(
            segmentations=segmentations,
            original_height=original_height,
            original_width=original_width,
            window_size=self.window_size,
            max_points=self.max_points,
            target_rings=self.target_rings,
            heatmap_radius=self.heatmap_radius,
            heatmap_sigma=self.heatmap_sigma,
        )

        if self.use_d4_aug:
            mode = np.random.choice(_D4_TRANSFORMS)
            if mode != "identity":
                image_np = _apply_d4_image(image_np, mode)
                hm_np = heatmap.numpy() if isinstance(heatmap, torch.Tensor) else heatmap
                # heatmap is (1, H, W); transform spatial dims
                hm_stacked = hm_np[0]
                if mode == "rot90":
                    hm_stacked = np.rot90(hm_stacked, k=1).copy()
                elif mode == "rot180":
                    hm_stacked = np.rot90(hm_stacked, k=2).copy()
                elif mode == "rot270":
                    hm_stacked = np.rot90(hm_stacked, k=3).copy()
                elif mode == "flip_h":
                    hm_stacked = np.ascontiguousarray(hm_stacked[:, ::-1])
                elif mode == "flip_v":
                    hm_stacked = np.ascontiguousarray(hm_stacked[::-1, :])
                elif mode == "flip_diag":
                    hm_stacked = np.ascontiguousarray(hm_stacked.T)
                elif mode == "flip_anti":
                    hm_stacked = np.ascontiguousarray(hm_stacked.T[::-1, ::-1])
                heatmap = torch.from_numpy(hm_stacked[None, :, :])

                # Transform graph coords only for the valid rows.
                graph_np = graph.numpy() if isinstance(graph, torch.Tensor) else graph
                valid_np = valid_vertex_mask.numpy() if isinstance(valid_vertex_mask, torch.Tensor) else valid_vertex_mask
                if valid_np.any():
                    valid_pts = graph_np[valid_np]
                    new_pts = _apply_d4_ring(valid_pts, mode, self.window_size, self.window_size)
                    new_pts[:, 0] = np.clip(new_pts[:, 0], 0, self.window_size - 1)
                    new_pts[:, 1] = np.clip(new_pts[:, 1], 0, self.window_size - 1)
                    graph_np = graph_np.copy()
                    graph_np[valid_np] = new_pts
                    graph = torch.from_numpy(graph_np)

        image = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
        sample = {
            "image": image,
            "heatmap": heatmap,
            "graph": graph,
            "perm_target": perm_target,
            "perm_mask": perm_mask,
            "valid_vertex_mask": valid_vertex_mask,
            "image_id": torch.tensor(image_id, dtype=torch.int64),
            "original_size": torch.tensor([original_height, original_width], dtype=torch.float32),
            "file_name": image_info["file_name"],
            "stats_original_ring_count": torch.tensor(stats.original_ring_count, dtype=torch.int64),
            "stats_kept_ring_count": torch.tensor(stats.kept_ring_count, dtype=torch.int64),
            "stats_dropped_ring_count": torch.tensor(stats.dropped_ring_count, dtype=torch.int64),
            "stats_original_vertex_count": torch.tensor(stats.original_vertex_count, dtype=torch.int64),
            "stats_kept_vertex_count": torch.tensor(stats.kept_vertex_count, dtype=torch.int64),
        }
        return sample
