import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from dataloader_polytopobench import CocoInferenceDataset
from models.backbone import DetectionBranch, NonMaxSuppression, R2U_Net
from models.matching import OptimalMatching


REPO_ROOT = Path(__file__).resolve().parent


def bounding_box_from_points(points):
    points = np.array(points).flatten()
    even_locations = np.arange(points.shape[0] / 2) * 2
    odd_locations = even_locations + 1
    x_coord = np.take(points, even_locations.tolist())
    y_coord = np.take(points, odd_locations.tolist())
    bbox = [x_coord.min(), y_coord.min(), x_coord.max() - x_coord.min(), y_coord.max() - y_coord.min()]
    bbox = [float(b) for b in bbox]
    return bbox


def scale_polygon_to_original_size(polygon, original_size, window_size):
    height = float(original_size[0])
    width = float(original_size[1])

    polygon = polygon.detach().cpu().float().clone()
    polygon[:, 1] *= width / window_size
    polygon[:, 0] *= height / window_size
    polygon = torch.fliplr(polygon)
    return polygon.reshape(-1).tolist()


def single_annotation(image_id, poly, category_id):
    result = {}
    result["image_id"] = int(image_id)
    result["category_id"] = int(category_id)
    result["score"] = 1.0
    result["segmentation"] = poly
    result["bbox"] = bounding_box_from_points(result["segmentation"])
    return result


def load_weights(module, weight_path, device):
    if not weight_path.is_file():
        raise FileNotFoundError(
            f"Missing weight file: {weight_path}. "
            "Use weights exported from the PolyTopoBench PolyWorld training run."
        )
    state_dict = torch.load(weight_path, map_location=device)
    module.load_state_dict(state_dict)


def resolve_weight_paths(weights_dir, backbone_weights, seg_head_weights, matching_weights):
    if weights_dir is not None:
        weights_dir = Path(weights_dir)
        backbone_weights = backbone_weights or (weights_dir / "polyworld_backbone")
        seg_head_weights = seg_head_weights or (weights_dir / "polyworld_seg_head")
        matching_weights = matching_weights or (weights_dir / "polyworld_matching")
    else:
        backbone_weights = backbone_weights or (REPO_ROOT / "trained_weights" / "polyworld_backbone")
        seg_head_weights = seg_head_weights or (REPO_ROOT / "trained_weights" / "polyworld_seg_head")
        matching_weights = matching_weights or (REPO_ROOT / "trained_weights" / "polyworld_matching")
    return Path(backbone_weights), Path(seg_head_weights), Path(matching_weights)


def prediction(
    batch_size,
    images_directory,
    annotations_path,
    output_json,
    window_size=320,
    num_workers=None,
    device_name=None,
    weights_dir=None,
    backbone_weights=None,
    seg_head_weights=None,
    matching_weights=None,
    n_peaks=256,
    nms_border_margin=0,
    nms_score_threshold=0.0,
):
    if device_name is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    model = R2U_Net().to(device).eval()
    head_ver = DetectionBranch().to(device).eval()
    suppression = NonMaxSuppression(
        n_peaks=n_peaks,
        border_margin=nms_border_margin,
        score_threshold=nms_score_threshold,
    ).to(device)
    matching = OptimalMatching().to(device).eval()
    backbone_weights, seg_head_weights, matching_weights = resolve_weight_paths(
        weights_dir=weights_dir,
        backbone_weights=backbone_weights,
        seg_head_weights=seg_head_weights,
        matching_weights=matching_weights,
    )

    print(f"Loading PolyWorld weights on {device}")
    load_weights(model, backbone_weights, device)
    load_weights(head_ver, seg_head_weights, device)
    load_weights(matching, matching_weights, device)

    dataset = CocoInferenceDataset(
        images_directory=images_directory,
        annotations_path=annotations_path,
        window_size=window_size,
    )

    if num_workers is None:
        cpu_count = os.cpu_count() or 0
        num_workers = min(batch_size, cpu_count)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    iterator = tqdm(dataloader)

    category_ids = dataset.coco.getCatIds()
    category_id = category_ids[0] if category_ids else 100

    speed = []
    predictions = []
    with torch.no_grad():
        for sample_batched in iterator:
            rgb = sample_batched["image"].to(device=device, dtype=torch.float32, non_blocking=True)
            idx = sample_batched["image_idx"]
            original_sizes = sample_batched["original_size"]

            t0 = time.time()

            features = model(rgb)
            occupancy_grid = head_ver(features)
            _, graph_pressed = suppression(occupancy_grid)
            polygons = matching.predict(rgb, features, graph_pressed, out="torch")

            speed.append(time.time() - t0)

            for image_offset, polygon_group in enumerate(polygons):
                for polygon in polygon_group:
                    scaled_polygon = scale_polygon_to_original_size(
                        polygon=polygon,
                        original_size=original_sizes[image_offset],
                        window_size=window_size,
                    )
                    predictions.append(single_annotation(idx[image_offset], [scaled_polygon], category_id))

    print("Average model speed: ", np.mean(speed) / batch_size, " [s / image]")

    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(predictions), encoding="utf-8")
    print(f"Saved predictions to {output_json}")


def build_argparser():
    parser = argparse.ArgumentParser(description="Run PolyWorld inference on a COCO-style patch dataset.")
    parser.add_argument(
        "--images-directory",
        type=Path,
        default=Path("data_processed/inria_building/hisup/val/images"),
    )
    parser.add_argument(
        "--annotations-path",
        type=Path,
        default=Path("data_processed/inria_building/hisup/val/annotation.json"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPO_ROOT / "predictions_inria_val.json",
    )
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--window-size", type=int, default=320)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None, help="Torch device, e.g. cpu, cuda, cuda:0.")
    parser.add_argument("--weights-dir", type=Path, default=None, help="Directory containing exported PolyWorld weights.")
    parser.add_argument("--backbone-weights", type=Path, default=None, help="Override backbone checkpoint path.")
    parser.add_argument("--seg-head-weights", type=Path, default=None, help="Override detection head checkpoint path.")
    parser.add_argument("--matching-weights", type=Path, default=None, help="Override matching checkpoint path.")
    parser.add_argument("--n-peaks", type=int, default=256,
                        help="Upper-bound vertex count per image returned by NMS.")
    parser.add_argument("--nms-border-margin", type=int, default=0,
                        help="Border margin (pixels) where NMS peaks are suppressed.")
    parser.add_argument("--nms-score-threshold", type=float, default=0.0,
                        help="Sigmoid score threshold below which NMS peaks are suppressed.")
    return parser


if __name__ == "__main__":
    args = build_argparser().parse_args()
    prediction(
        batch_size=args.batch_size,
        images_directory=args.images_directory,
        annotations_path=args.annotations_path,
        output_json=args.output_json,
        window_size=args.window_size,
        num_workers=args.num_workers,
        device_name=args.device,
        weights_dir=args.weights_dir,
        backbone_weights=args.backbone_weights,
        seg_head_weights=args.seg_head_weights,
        matching_weights=args.matching_weights,
        n_peaks=args.n_peaks,
        nms_border_margin=args.nms_border_margin,
        nms_score_threshold=args.nms_score_threshold,
    )
