"""
Generate RoIPoly polygon predictions from Sparse R-CNN proposal boxes.

Each region of interest (RoI) feature corresponds to one polygon prediction. The release
runner supplies a proposal JSON converted from Sparse R-CNN detections, matching the
paper-reported setting.
"""
import json
import time
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import torch
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.logger import setup_logger
from detectron2.data import build_detection_test_loader
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json
from roipoly import RoIPolyDatasetMapper, add_roipoly_config
import shapely.geometry
from detectron2.config import get_cfg
import argparse
import os


def setup_cfg(args):
    """Set up the configuration for the Detectron2 model."""
    cfg = get_cfg()
    add_roipoly_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.CORNER_THRESHOLD = args.corner_threshold
    cfg.OUTPUT_DIRPATH = args.output
    cfg.ITER = args.iter
    cfg.IMAGE_ID = args.image_id
    cfg.DATASET_NAME = args.dataset_name
    cfg.PRED_CATEGORY_ID = args.category_id
    cfg.MAX_IMAGES = args.max_images
    cfg.TEST_BATCH_SIZE = args.batch_size
    cfg.PRED_JSON = args.pred_json
    cfg.freeze()
    return cfg


def get_parser():
    """Set up argument parser for command line options."""
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument("--config-file", default="configs/roipoly.res50.100pro.3x_inference_acc_test.yaml", metavar="FILE", help="Path to config file")
    parser.add_argument("--output", help="Directory to save output visualizations")
    parser.add_argument("--gt-path", help="Directory of the annotation file")
    parser.add_argument("--corner-threshold", type=float, default=0.4, help="Threshold for vertex predictions to be classified as corners")
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--iter", type=int, default=2842656, help="Checkpoint iteration number")
    parser.add_argument("--image-id", type=int, default=1, help="ID of the input image")
    parser.add_argument("--dataset-name", default="inria_val", help="Dataset name used for registration")
    parser.add_argument("--category-id", type=int, default=100, help="Category id written to prediction JSON")
    parser.add_argument("--max-images", type=int, default=0, help="Stop after this many images. 0 means evaluate all images.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size used by the test dataloader.")
    parser.add_argument("--pred-json", default="", help="Exact prediction JSON path. Defaults to <output>/predictions_<iter>.json.")
    parser.add_argument("--test-json", "--train-json", dest="test_json", help="Path to the COCO-format annotation file")
    parser.add_argument("--test-path", "--train-path", dest="test_path", help="Path to the images directory")
    parser.add_argument("--opts", default=[], nargs=argparse.REMAINDER, help="Modify config options using the command-line 'KEY VALUE' pairs")
    return parser


def single_annotation(image_id, poly, bbox, score, category_id, gt_is_hole=None):
    """Create a single annotation result dictionary."""
    annotation = {
        "image_id": int(image_id),
        "category_id": int(category_id),
        "score": score,
        "segmentation": poly,
        "bbox": [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]  # Convert (x1, y1, x2, y2) to (x1, y1, width, height)
    }
    if gt_is_hole is not None:
        annotation["gt_is_hole"] = bool(gt_is_hole)
        annotation["gt_ring_role"] = "hole" if bool(gt_is_hole) else "exterior"
    return annotation


def prediction(cfg):
    """Perform predictions on the test dataset."""
    mapper = RoIPolyDatasetMapper(cfg, is_train=False)
    try:
        dataloader = build_detection_test_loader(
            DatasetCatalog.get(cfg.DATASET_NAME),
            mapper=mapper,
            batch_size=cfg.TEST_BATCH_SIZE,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
        )
    except TypeError:
        # detectron2 <0.7 does not accept batch_size; it evaluates one image at a time.
        dataloader = build_detection_test_loader(
            DatasetCatalog.get(cfg.DATASET_NAME),
            mapper=mapper,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
        )
    test_iterator = tqdm(dataloader)

    # Load the RoIPoly model
    model = build_model(cfg)
    model.eval()
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    speed = []  # List to keep track of model speed
    predictions = []  # List to store prediction results

    # Iterate over the test dataset
    processed_images = 0
    for batched_inputs in test_iterator:
        if cfg.MAX_IMAGES and processed_images >= cfg.MAX_IMAGES:
            break

        # Perform inference with the model
        with torch.no_grad():
            t0 = time.time()
            outputs = model(batched_inputs)
            t1 = time.time()

        batch_size = len(batched_inputs)
        speed.extend([(t1 - t0) / batch_size] * batch_size)

        for batch_index, per_input in enumerate(batched_inputs):
            if cfg.MAX_IMAGES and processed_images >= cfg.MAX_IMAGES:
                break

            instances = per_input["instances"]
            proposal_boxes = instances.gt_boxes.tensor
            num_proposal_boxes = proposal_boxes.shape[0]
            pred_logits = outputs["pred_logits"][batch_index]  # [num_proposals, num_corners]
            pred_coords = outputs["pred_coords"][batch_index]  # [num_proposals, num_corners, 2]
            fg_mask = torch.sigmoid(pred_logits) > cfg.CORNER_THRESHOLD  # [num_proposals, num_corners]

            polys_image = []  # List to store valid polygons
            scores_image = []  # List to store average scores per polygon
            boxes_image = []  # List to store bounding boxes for valid polygons
            gt_hole_flags_image = []  # GT semantic role aligned with kept polygons

            for j in range(num_proposal_boxes):
                fg_mask_per_poly = fg_mask[j]  # [num_corners]
                valid_coords_per_poly = pred_coords[j][fg_mask_per_poly]  # [num_valid_corners_per_poly, 2]
                valid_scores_per_poly = torch.sigmoid(pred_logits[j])[fg_mask_per_poly]  # [num_valid_corners_per_poly]
                gt_is_hole = None
                if instances.has("gt_is_hole"):
                    gt_is_hole = bool(instances.gt_is_hole[j].item())

                if len(valid_coords_per_poly) > 0:
                    coords = valid_coords_per_poly.cpu().numpy()  # [num_valid_corners_per_poly, 2]
                    if len(coords) >= 3 and shapely.geometry.Polygon(coords).area >= 10:
                        polys_image.append(coords)
                        scores_image.append(
                            torch.mean(valid_scores_per_poly).cpu().numpy())  # Average score per polygon
                        boxes_image.append(np.array([np.min(coords[:, 0]), np.min(coords[:, 1]),
                                                     np.max(coords[:, 0]), np.max(coords[:, 1])]))
                        gt_hole_flags_image.append(gt_is_hole)

            # Store predictions for the current image
            image_id = per_input["image_id"]
            for j, poly in enumerate(polys_image):
                box = list(np.array(boxes_image[j]).flatten().astype(np.float64))
                score = float(scores_image[j])
                poly = list(np.array(poly).flatten().astype(np.float64))

                # polygons with 2 vertices will be mistaken as bounding boxes and cause error in eval_coco.py
                if len(poly) > 4:
                    predictions.append(
                        single_annotation(
                            image_id,
                            [poly],
                            box,
                            score,
                            cfg.PRED_CATEGORY_ID,
                            gt_is_hole=gt_hole_flags_image[j],
                        )
                    )

            processed_images += 1

    # Print average model speed
    print("Average model speed: ", np.mean(speed), " [s / image]")

    # Save predictions to a JSON file
    output_path = cfg.PRED_JSON or os.path.join(cfg.OUTPUT_DIRPATH, f"predictions_{int(cfg.ITER)}.json")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as fp:
        json.dump(predictions, fp)


def register_my_dataset(dataset_name="inria_val",
                        TEST_JSON="../../../../data_processed/inria_building/roipoly/val/annotation_roipoly.json",
                        TEST_PATH="../../../../data_processed/inria_building/roipoly/val/images"):
    """Register your own COCO-format dataset.

       usage::
       from detectron2.data import DatasetCatalog, MetadataCatalog
       from detectron2.data.datasets.coco import load_coco_json

    :param TRAIN_JSON: the file path of the annotation
    :param TRAIN_PATH: the folder path of the images
    """
    extra_annotation_keys = [
        "is_hole",
        "ring_role",
        "source_annotation_id",
        "source_image_id",
        "source_image_name",
        "source_geojson",
        "source_geometry_index",
        "source_ring_index",
        "source_parent_hole_count",
        "patch_origin",
    ]
    DatasetCatalog.register(
        dataset_name,
        lambda: load_coco_json(TEST_JSON, TEST_PATH, dataset_name, extra_annotation_keys),
    )
    MetadataCatalog.get(dataset_name).set(json_file=TEST_JSON, image_root=TEST_PATH)


if __name__ == '__main__':
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()

    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)  # default config, args.config_file, args.opts
    register_my_dataset(dataset_name=args.dataset_name, TEST_JSON=args.test_json,
                        TEST_PATH=args.test_path)
    prediction(cfg)
