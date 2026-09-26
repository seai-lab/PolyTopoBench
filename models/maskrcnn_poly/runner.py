from __future__ import annotations

import json
from pathlib import Path

from utilis.common import CommandStep, RunContext, require_paths


DEFAULT_ENV = "polytopobench"


def _checkpoint(ctx: RunContext) -> Path:
    value = ctx.param("CHECKPOINT", "")
    if not value:
        return ctx.output_dir() / "checkpoints" / "best.pt"
    path = Path(value)
    return path if path.is_absolute() else ctx.release_root / path


def _first_val_image_ids(data_root: Path, count: int) -> list[int]:
    # infer.py walks val/annotation.json["images"] in order (shuffle=False).
    with (data_root / "val" / "annotation.json").open("r") as f:
        images = json.load(f)["images"]
    return [int(meta["id"]) for meta in images[:count]]


def prepare(ctx: RunContext) -> None:
    if ctx.mode in {"infer", "eval"} and ctx.run_val and not _checkpoint(ctx).is_file():
        raise SystemExit(
            f"maskrcnn_poly: no checkpoint at {_checkpoint(ctx)}. Pass --run-name of the training run "
            f"(e.g. maskrcnn_poly_{ctx.task}_train_eval) or --set CHECKPOINT=<path/to/best.pt>."
        )


def build_commands(ctx: RunContext) -> list[CommandStep]:
    env_name = ctx.env_name or DEFAULT_ENV
    source = ctx.source_root() / "maskrcnn_seg"
    data_root = ctx.data_dir("maskrcnn_seg")
    output_dir = ctx.output_dir()
    pred_path = output_dir / "predictions.json"
    bbox_path = output_dir / "bbox_predictions_for_sam2.json"
    ckpt_path = _checkpoint(ctx)
    require_paths([source / "train.py", source / "infer.py", data_root / "train", data_root / "val"])

    max_epochs = ctx.param("MAX_EPOCHS", 1 if ctx.smoke else 100)
    batch_size = ctx.param("BATCH_SIZE", 2 if ctx.smoke else 6)
    num_workers = ctx.param("NUM_WORKERS", 2 if ctx.smoke else 4)
    max_infer_images = int(ctx.param("MAX_INFER_IMAGES", 4 if ctx.smoke else 0))
    cwd = ctx.release_root
    steps: list[CommandStep] = []

    if ctx.mode in {"train", "train_eval"}:
        steps.append(CommandStep(
            "maskrcnn_poly_train",
            cwd=cwd,
            env=ctx.env(),
            argv=ctx.python(env_name, source / "train.py", [
                "--mirror-root", ctx.path(data_root, cwd),
                "--output-dir", ctx.path(output_dir, cwd),
                "--max-epochs", max_epochs,
                "--batch-size", batch_size,
                "--lr", ctx.param("LR", "0.005"),
                "--num-workers", num_workers,
                "--seed", str(ctx.seed),
                "--max-train-steps", ctx.param("MAX_TRAIN_STEPS", 2 if ctx.smoke else 0),
            ], cwd),
        ))

    if ctx.run_val and ctx.mode in {"infer", "eval", "train_eval"}:
        steps.append(CommandStep(
            "maskrcnn_poly_infer",
            cwd=cwd,
            env=ctx.env(),
            argv=ctx.python(env_name, source / "infer.py", [
                "--mirror-root", ctx.path(data_root, cwd),
                "--checkpoint", ctx.path(ckpt_path, cwd),
                "--predictions-out", ctx.path(pred_path, cwd),
                "--bbox-out", ctx.path(bbox_path, cwd),
                "--score-threshold", ctx.param("SCORE_THRESHOLD", 0.05),
                "--connectivity", ctx.param("CONNECTIVITY", 4),
                "--simplify-tol", ctx.param("SIMPLIFY_TOL", 1.0),
                "--min-area", ctx.param("MIN_AREA", 16),
                "--min-hole-area", ctx.param("MIN_HOLE_AREA", 16),
                "--max-images", str(max_infer_images),
            ], cwd),
        ))
        eval_step = ctx.evaluate_step(name="maskrcnn_poly_eval", pred=pred_path, env_name=env_name, cwd=cwd)
        if max_infer_images > 0:
            # Only the first N val images were inferred: score exactly those, not the whole split.
            ids = _first_val_image_ids(data_root, max_infer_images)
            eval_step.argv += ["--image-ids", ",".join(map(str, ids))]
        steps.append(eval_step)
    return steps
