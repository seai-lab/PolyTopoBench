"""
Convert detector predictions into a COCO-format bbox-only annotation file.

This is used for feeding Sparse R-CNN detector proposals back into RoIPoly.
"""
import json
from collections import defaultdict
from tqdm import tqdm
import argparse


def predictions_to_coco(
    json_path,
    annotation_path,
    save_path,
    building_type,
    min_area=96 ** 2 * 1.44,
    score_threshold=0.0,
    topk_per_image=0,
):
    """
    Convert object detector predictions to COCO-format annotations and filter based on building size.

    This function ensures that the output COCO-format annotations contain all the images from the
    given test annotation file, and filters bounding boxes based on their size (small/medium or large).

    Args:
        json_path (str): Path to the predictions JSON file from the object detector.
        annotation_path (str): Path to the COCO-format annotation JSON used for testing.
        save_path (str): Path to save the newly generated COCO-format annotation file.
        building_type (str): Specify whether to keep "small_medium", "large", or "all" buildings.
        min_area (float, optional): Threshold to filter buildings by bounding box area. Defaults to 96**2 * 1.44.
        score_threshold (float, optional): Drop predictions below this score.
        topk_per_image (int, optional): Keep at most this many predictions per image after score filtering.

    Returns:
        None
    """
    print(f"Loading predictions from {json_path}...")
    with open(json_path, 'r') as f:
        predictions = json.load(f)
    print("Predictions loaded.")

    # Load the annotation data used for testing (to ensure all images are included)
    print(f"Loading annotation from {annotation_path}...")
    with open(annotation_path, 'r') as f:
        annotation_data = json.load(f)

    # Prepare COCO-style labels dictionary for the output
    labels = {
        "info": {
            "contributor": "PolyTopoBench",
            "about": "Detector proposals for RoIPoly",
            "description": "PolyTopoBench proposal annotations",
            "version": "1.0",
        },
        "categories": [{"id": 100, "name": "building", "supercategory": "building"}],
        "images": [],
        "annotations": []
    }

    filtered_predictions = []
    for pred in tqdm(predictions, desc="Filtering predictions"):
        score = float(pred.get("score", 0.0))
        if score < score_threshold:
            continue

        w, h = pred["bbox"][2], pred["bbox"][3]
        area = h * w

        if building_type == "small_medium" and area >= min_area:
            continue
        if building_type == "large" and area < min_area:
            continue
        if building_type not in {"small_medium", "large", "all"}:
            raise ValueError(f"Unsupported building_type: {building_type}")

        filtered_predictions.append(pred)

    if topk_per_image > 0:
        grouped_predictions = defaultdict(list)
        for pred in filtered_predictions:
            grouped_predictions[pred["image_id"]].append(pred)

        filtered_predictions = []
        for image_id, preds in grouped_predictions.items():
            preds = sorted(preds, key=lambda item: float(item.get("score", 0.0)), reverse=True)
            filtered_predictions.extend(preds[:topk_per_image])

    for i, pred in enumerate(tqdm(filtered_predictions, desc="Writing COCO annotations")):
        w, h = pred["bbox"][2], pred["bbox"][3]
        labels["annotations"].append(
            {
                "id": i,
                "image_id": pred["image_id"],
                "area": h * w,
                "category_id": 100,
                "iscrowd": 0,
                "bbox": pred["bbox"],
            }
        )

    # Ensure all images in the test set are included
    labels["images"] = annotation_data['images']

    # Save the filtered annotations to the output path
    print(f"Saving filtered annotations to {save_path}...")
    with open(save_path, 'w') as f:
        json.dump(labels, f)

    print(f"Filtered annotations saved successfully to {save_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert object detector predictions to COCO-format annotations.")
    parser.add_argument('--json_path', type=str, required=True, help='Path to the predictions JSON file')
    parser.add_argument('--annotation_path', type=str, required=True, help='Path to the test annotation JSON file')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save the COCO-format annotation file')
    parser.add_argument('--type', type=str, required=True, choices=["small_medium", "large", "all"],
                        help='Specify to keep "small_medium", "large", or "all" buildings')
    parser.add_argument('--min_area', type=float, default=96 ** 2 * 1.44, help='Threshold to filter buildings by area')
    parser.add_argument('--score_threshold', type=float, default=0.0, help='Drop predictions below this score')
    parser.add_argument('--topk_per_image', type=int, default=0, help='Keep at most this many predictions per image')

    args = parser.parse_args()

    predictions_to_coco(
        args.json_path,
        args.annotation_path,
        args.save_path,
        args.type,
        args.min_area,
        score_threshold=args.score_threshold,
        topk_per_image=args.topk_per_image,
    )
