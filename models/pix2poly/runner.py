from __future__ import annotations

from utilis.common import CommandStep, RunContext, require_paths


DEFAULT_ENV = "polytopobench"


def build_commands(ctx: RunContext) -> list[CommandStep]:
    env_name = ctx.env_name or DEFAULT_ENV
    source = ctx.source_root()
    data_root = ctx.data_dir("pix2poly")
    runs_dir = ctx.output_root / "pix2poly" / ctx.dataset / ctx.task
    output_dir = runs_dir / ctx.run_name
    require_paths([source / "train_ddp.py", data_root / "train", data_root / "val"])

    cwd = source
    common_env = ctx.env(
        PIX2POLY_DATASET=ctx.param("PIX2POLY_DATASET", f"{ctx.dataset}_{ctx.task}"),
        PIX2POLY_TRAIN_DATASET_DIR=ctx.path(data_root / "train", cwd),
        PIX2POLY_VAL_DATASET_DIR=ctx.path(data_root / "val", cwd),
        PIX2POLY_TEST_IMAGES_DIR=ctx.path(data_root / "val" / "images", cwd),
        PIX2POLY_IMG_SIZE=ctx.param("PIX2POLY_IMG_SIZE", 512),
        PIX2POLY_INPUT_SIZE=ctx.param("PIX2POLY_INPUT_SIZE", 512),
        PIX2POLY_N_VERTICES=ctx.param("PIX2POLY_N_VERTICES", 224),
        PIX2POLY_AFFINE_P=ctx.param("PIX2POLY_AFFINE_P", 0),
        PIX2POLY_RANDOM_ROTATE90_P=ctx.param("PIX2POLY_RANDOM_ROTATE90_P", 0),
        PIX2POLY_PRETRAINED_ENCODER=ctx.param("PIX2POLY_PRETRAINED_ENCODER", 0),
        PIX2POLY_USE_AMP=ctx.param("PIX2POLY_USE_AMP", 1),
        PIX2POLY_ALLOW_TF32=ctx.param("PIX2POLY_ALLOW_TF32", 1),
        PIX2POLY_LR=ctx.param("PIX2POLY_LR", "2e-4"),
        PIX2POLY_BATCH_SIZE=ctx.param("PIX2POLY_BATCH_SIZE", 2 if ctx.smoke else 28),
        PIX2POLY_NUM_EPOCHS=ctx.param("PIX2POLY_NUM_EPOCHS", 1 if ctx.smoke else 200),
        PIX2POLY_NUM_WORKERS=ctx.param("PIX2POLY_NUM_WORKERS", 0 if ctx.smoke else 8),
        PIX2POLY_VAL_EVERY=ctx.param("PIX2POLY_VAL_EVERY", 1 if ctx.smoke else 5),
        PIX2POLY_SAVE_EVERY=ctx.param("PIX2POLY_SAVE_EVERY", 1 if ctx.smoke else 10),
        PIX2POLY_DEBUG_MAX_TRAIN_STEPS=ctx.param("PIX2POLY_DEBUG_MAX_TRAIN_STEPS", 2 if ctx.smoke else 0),
        PIX2POLY_DEBUG_MAX_VAL_STEPS=ctx.param("PIX2POLY_DEBUG_MAX_VAL_STEPS", 1 if ctx.smoke else 0),
        PIX2POLY_RUNS_DIR=ctx.path(runs_dir, cwd),
        PIX2POLY_EXPERIMENT_NAME=ctx.run_name,
        PIX2POLY_SAVE_BEST=ctx.param("PIX2POLY_SAVE_BEST", "true"),
        PIX2POLY_SAVE_LATEST=ctx.param("PIX2POLY_SAVE_LATEST", "true"),
        PIX2POLY_PRED_MAX_BATCHES=ctx.param("PIX2POLY_PRED_MAX_BATCHES", 1 if ctx.smoke else 0),
        PIX2POLY_PRED_BATCH_SIZE=ctx.param("PIX2POLY_PRED_BATCH_SIZE", 2 if ctx.smoke else 32),
        PIX2POLY_PRED_NUM_WORKERS=ctx.param("PIX2POLY_PRED_NUM_WORKERS", 0 if ctx.smoke else 8),
    )
    steps: list[CommandStep] = []
    if ctx.mode in {"train", "train_eval"}:
        steps.append(CommandStep(
            "pix2poly_train",
            cwd=cwd,
            env=common_env,
            argv=ctx.conda(env_name, ["torchrun", "--standalone", "--nproc_per_node=1", "train_ddp.py"]),
        ))
    if ctx.run_val and ctx.mode in {"infer", "eval", "train_eval"}:
        args = ["-d", common_env["PIX2POLY_DATASET"], "-e", ctx.path(output_dir, cwd), "-c", "best_valid_metric", "-o", "val_best_valid_metric"]
        if common_env["PIX2POLY_PRED_MAX_BATCHES"] != "0":
            args += ["--max-batches", common_env["PIX2POLY_PRED_MAX_BATCHES"]]
        steps.append(CommandStep(
            "pix2poly_predict",
            cwd=cwd,
            env=common_env,
            argv=ctx.python(env_name, source / "predict_inria_coco_val_set.py", args, cwd),
        ))
    return steps
