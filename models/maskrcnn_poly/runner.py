from __future__ import annotations

from utilis.common import CommandStep, RunContext, require_paths


DEFAULT_ENV = "polytopobench"


def build_commands(ctx: RunContext) -> list[CommandStep]:
    env_name = ctx.env_name or DEFAULT_ENV
    source = ctx.source_root() / "maskrcnn_seg"
    data_root = ctx.data_dir("maskrcnn_seg")
    output_dir = ctx.output_dir()
    pred_path = output_dir / "predictions.json"
    bbox_path = output_dir / "bbox_predictions_for_sam2.json"
    ckpt_path = output_dir / "checkpoints" / "best.pt"
    require_paths([source / "train.py", source / "infer.py", data_root / "train", data_root / "val"])

    max_epochs = ctx.param("MAX_EPOCHS", 1 if ctx.smoke else 100)
    batch_size = ctx.param("BATCH_SIZE", 2 if ctx.smoke else 6)
    num_workers = ctx.param("NUM_WORKERS", 2 if ctx.smoke else 4)
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
                "--max-images", ctx.param("MAX_INFER_IMAGES", 4 if ctx.smoke else 0),
            ], cwd),
        ))
        steps.append(ctx.evaluate_step(name="maskrcnn_poly_eval", pred=pred_path, env_name=env_name, cwd=cwd))
    return steps
