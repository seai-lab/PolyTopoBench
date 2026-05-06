from __future__ import annotations

from utilis.common import CommandStep, RunContext, require_paths


DEFAULT_ENV = "polytopobench"


def build_commands(ctx: RunContext) -> list[CommandStep]:
    env_name = ctx.env_name or DEFAULT_ENV
    source = ctx.source_root()
    data_root = ctx.data_dir("hisup") if ctx.is_inria else ctx.data_dir("polyworld", fallback="hisup")
    output_dir = ctx.output_dir()
    ann_name = "annotation-smoke.json" if ctx.smoke and (data_root / "train" / "annotation-smoke.json").exists() else "annotation.json"
    require_paths([source / "train_inria_multiring.py", data_root / "train" / "images", data_root / "train" / ann_name, data_root / "val" / ann_name])

    cwd = source
    args = [
        "--train-images-directory", ctx.path(data_root / "train" / "images", cwd),
        "--train-annotations-path", ctx.path(data_root / "train" / ann_name, cwd),
        "--val-images-directory", ctx.path(data_root / "val" / "images", cwd),
        "--val-annotations-path", ctx.path(data_root / "val" / ann_name, cwd),
        "--output-dir", ctx.path(output_dir, cwd),
        "--window-size", ctx.param("WINDOW_SIZE", 320),
        "--max-points", ctx.param("MAX_POINTS", 256),
        "--target-rings", ctx.param("TARGET_RINGS", "all"),
        "--batch-size", ctx.param("BATCH_SIZE", 4 if ctx.smoke else 16),
        "--num-workers", ctx.param("NUM_WORKERS", 4 if ctx.smoke else 8),
        "--epochs", ctx.param("EPOCHS", 1 if ctx.smoke else 100),
        "--lr", ctx.param("LR", "1e-4"),
        "--weight-decay", ctx.param("WEIGHT_DECAY", "1e-4"),
    ]
    if ctx.smoke:
        args += [
            "--train-image-limit", ctx.param("TRAIN_IMAGE_LIMIT", 32),
            "--val-image-limit", ctx.param("VAL_IMAGE_LIMIT", 16),
            "--max-train-steps-per-epoch", ctx.param("MAX_TRAIN_STEPS_PER_EPOCH", 2),
            "--max-val-batches", ctx.param("MAX_VAL_BATCHES", 1),
        ]
    return [
        CommandStep(
            "polyworld_train",
            cwd=cwd,
            env=ctx.env(PYTHONNOUSERSITE="1"),
            argv=ctx.python(env_name, source / "train_inria_multiring.py", args, cwd),
        )
    ] if ctx.mode in {"train", "train_eval"} else []
