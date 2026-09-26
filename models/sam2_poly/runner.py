from __future__ import annotations

import json
from pathlib import Path

from utilis.common import CommandStep, RunContext, require_paths


DEFAULT_ENV = "polytopobench"
BBOX_FILE = "bbox_predictions_for_sam2.json"


def _bbox_json(ctx: RunContext) -> Path:
    """SAM2 box prompts: --bbox-json, else --set MASKRCNN_BBOX_JSON=..., else the default Mask R-CNN run."""
    value = ctx.param("MASKRCNN_BBOX_JSON", "")
    if ctx.bbox_json is not None or value:
        path = Path(ctx.bbox_json or value)
        return path if path.is_absolute() else ctx.release_root / path
    # Default Mask R-CNN run names (see utilis.common.default_run_name). A non-smoke SAM2 run
    # never falls back to smoke boxes (those only cover a handful of val images).
    suffixes = ["smoke", "train_eval", "infer", "eval"] if ctx.smoke else ["train_eval", "infer", "eval"]
    run_root = ctx.output_root / "maskrcnn_poly" / ctx.dataset / ctx.task
    candidates = [run_root / f"maskrcnn_poly_{ctx.task}_{suffix}" / BBOX_FILE for suffix in suffixes]
    return next((path for path in candidates if path.is_file()), candidates[0])


def _sam2_image_ids(bbox_json: Path, data_root: Path, score_threshold: float, count: int) -> list[int]:
    # run_sam2_inference.py --max-images N prompts the first N sorted val ids that have a kept
    # box. Score every val id up to the last prompted one, so box-less images in that range
    # still count as misses (as they do in a full run).
    with bbox_json.open("r") as f:
        boxes = json.load(f)
    with (data_root / "val" / "annotation.json").open("r") as f:
        val_ids = sorted(int(meta["id"]) for meta in json.load(f)["images"])
    prompted = sorted({int(b["image_id"]) for b in boxes if float(b.get("score", 0.0)) >= score_threshold} & set(val_ids))
    if not prompted:
        return []
    last = prompted[:count][-1]
    return [image_id for image_id in val_ids if image_id <= last]


def prepare(ctx: RunContext) -> None:
    if ctx.mode == "train":
        return
    bbox_json = _bbox_json(ctx)
    if not bbox_json.is_file():
        raise SystemExit(
            f"sam2_poly needs Mask R-CNN box prompts but {bbox_json} does not exist. Run "
            f"`main.py --method maskrcnn_poly --dataset {ctx.dataset} --task {ctx.task} --mode train_eval` first, "
            "or pass --bbox-json <maskrcnn run>/bbox_predictions_for_sam2.json."
        )


def build_commands(ctx: RunContext) -> list[CommandStep]:
    if ctx.mode == "train":
        print("sam2_poly is inference-only (frozen SAM2 prompted by Mask R-CNN boxes); nothing to train.", flush=True)
        return []

    env_name = ctx.env_name or DEFAULT_ENV
    source = ctx.source_root()
    data_root = ctx.data_dir("sam2_seg")
    output_dir = ctx.output_dir()
    pred_path = output_dir / "predictions.json"
    bbox_json = _bbox_json(ctx)
    require_paths([source / "run_sam2_inference.py", data_root / "val"])

    score_threshold = ctx.param("SCORE_THRESHOLD", 0.5)
    max_infer_images = int(ctx.param("MAX_INFER_IMAGES", 4 if ctx.smoke else 0))
    cwd = ctx.release_root
    steps = [
        CommandStep(
            "sam2_poly_infer",
            cwd=cwd,
            env=ctx.env(),
            argv=ctx.python(env_name, source / "run_sam2_inference.py", [
                "--mirror-root", ctx.path(data_root, cwd),
                "--bbox-json", ctx.path(bbox_json, cwd),
                "--output", ctx.path(pred_path, cwd),
                "--summary-out", ctx.path(output_dir / "sam2_summary.json", cwd),
                "--score-threshold", score_threshold,
                "--connectivity", ctx.param("CONNECTIVITY", 4),
                "--simplify-tol", ctx.param("SIMPLIFY_TOL", 1.0),
                "--min-area", ctx.param("MIN_AREA", 16),
                "--min-hole-area", ctx.param("MIN_HOLE_AREA", 16),
                "--sam2-model-id", ctx.param("SAM2_MODEL_ID", "facebook/sam2-hiera-small"),
                "--max-images", str(max_infer_images),
            ], cwd),
        )
    ]
    if ctx.run_val:
        eval_step = ctx.evaluate_step(name="sam2_poly_eval", pred=pred_path, env_name=env_name, cwd=cwd)
        if max_infer_images > 0 and bbox_json.is_file():
            # Only the first N prompted val images were inferred: score exactly those.
            ids = _sam2_image_ids(bbox_json, data_root, float(score_threshold), max_infer_images)
            eval_step.argv += ["--image-ids", ",".join(map(str, ids))]
        steps.append(eval_step)
    return steps
