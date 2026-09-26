from __future__ import annotations

from utilis.common import CommandStep, RunContext, require_paths


DEFAULT_ENV = "polytopobench"


def build_commands(ctx: RunContext) -> list[CommandStep]:
    env_name = ctx.env_name or DEFAULT_ENV
    source = ctx.source_root()
    data_root = ctx.data_dir("hisup")
    output_dir = ctx.output_dir()
    config = source / "config-files" / ("for_hisup_hrnet48.yaml" if ctx.is_inria else "deventer_512_hrnet48_singleclass.yaml")
    require_paths([source / "scripts" / "train.py", source / "scripts" / "test.py", config, data_root / "train", data_root / "val"])

    ims_per_batch = ctx.param("IMS_PER_BATCH", 2 if ctx.smoke else 8)
    max_epoch = ctx.param("MAX_EPOCH", 1 if ctx.smoke else 100)
    checkpoint_period = ctx.param("CHECKPOINT_PERIOD", 1 if ctx.smoke else 10)
    num_workers = ctx.param("NUM_WORKERS", 8)
    # Smoke runs evaluate on the 16-image val/annotation-smoke.json subset (same image ids as the
    # canonical GT); full runs always use the complete canonical val/annotation.json.
    eval_ann = data_root / "val" / "annotation.json"
    smoke_ann = data_root / "val" / "annotation-smoke.json"
    if ctx.smoke and ctx.param("SMOKE_EVAL_SUBSET", 1) != "0" and smoke_ann.is_file():
        eval_ann = smoke_ann
    cwd = source
    common_env = ctx.env(
        HISUP_EVAL_ANN_FILE=ctx.path(eval_ann, cwd),
        PYTHONPATH=f"{ctx.path(source, cwd)}:{ctx.param('PYTHONPATH', '')}",
        HISUP_DATA_ROOT=ctx.path(data_root, cwd),
        POLYTOPOBENCH_DATA_PROCESSED_ROOT=ctx.path(ctx.data_processed_root, cwd),
        POLYTOPOBENCH_MAX_TRAIN_STEPS=ctx.param("MAX_TRAIN_STEPS", 2 if ctx.smoke else 0),
    )
    steps: list[CommandStep] = []
    train_opts = [
        "--config-file", ctx.path(config, cwd),
        "--seed", str(ctx.seed),
        "OUTPUT_DIR", ctx.path(output_dir, cwd),
        "DATASETS.TRAIN", "('custom_hisup_train',)",
        "DATASETS.TEST", "('custom_hisup_val',)",
        "DATASETS.IMAGE.HEIGHT", "512",
        "DATASETS.IMAGE.WIDTH", "512",
        "DATASETS.ORIGIN.HEIGHT", "512",
        "DATASETS.ORIGIN.WIDTH", "512",
        "DATASETS.TARGET.HEIGHT", ctx.param("TARGET_SIZE", 128),
        "DATASETS.TARGET.WIDTH", ctx.param("TARGET_SIZE", 128),
        "SOLVER.IMS_PER_BATCH", ims_per_batch,
        "SOLVER.BASE_LR", ctx.param("BASE_LR", "1e-4"),
        "SOLVER.MAX_EPOCH", max_epoch,
        "SOLVER.CHECKPOINT_PERIOD", checkpoint_period,
        "DATALOADER.NUM_WORKERS", num_workers,
    ]
    if ctx.mode in {"train", "train_eval"}:
        steps.append(CommandStep(
            "hisup_train",
            cwd=cwd,
            env=common_env,
            argv=ctx.python(env_name, source / "scripts" / "train.py", train_opts, cwd),
        ))
    if ctx.run_val and ctx.mode in {"infer", "eval", "train_eval"}:
        steps.append(CommandStep(
            "hisup_eval",
            cwd=cwd,
            env=common_env,
            argv=ctx.python(env_name, source / "scripts" / "test.py", [
                "--config-file", ctx.path(config, cwd),
                "--eval-type", ctx.param("EVAL_TYPE", "coco_iou"),
                "OUTPUT_DIR", ctx.path(output_dir, cwd),
                "DATASETS.TEST", "('custom_hisup_val',)",
                "DATASETS.IMAGE.HEIGHT", "512",
                "DATASETS.IMAGE.WIDTH", "512",
                "DATASETS.ORIGIN.HEIGHT", "512",
                "DATASETS.ORIGIN.WIDTH", "512",
                "DATASETS.TARGET.HEIGHT", ctx.param("TARGET_SIZE", 128),
                "DATASETS.TARGET.WIDTH", ctx.param("TARGET_SIZE", 128),
                "DATALOADER.NUM_WORKERS", num_workers,
            ], cwd),
        ))
        # test.py writes <OUTPUT_DIR>/custom_hisup_val.json with segmentation=[exterior, hole1, ...]
        # and image ids read from the canonical annotation file -> unified evaluator, pred-type hisup.
        steps.append(ctx.evaluate_step(
            name="hisup_unified_eval",
            pred=output_dir / "custom_hisup_val.json",
            gt=eval_ann,
            env_name=env_name,
            pred_type="hisup",
            gt_type="hisup",
        ))
    return steps
