from __future__ import annotations

from utilis.common import CommandStep, RunContext, require_paths


DEFAULT_ENV = "polytopobench"


def _ann_name(ctx: RunContext, data_root) -> str:
    smoke_ann = data_root / "train" / "annotation-smoke.json"
    return "annotation-smoke.json" if ctx.smoke and smoke_ann.exists() else "annotation.json"


def build_commands(ctx: RunContext) -> list[CommandStep]:
    env_name = ctx.env_name or DEFAULT_ENV
    source = ctx.source_root()
    data_root = ctx.data_dir("hisup") if ctx.is_inria else ctx.data_dir("gcp", fallback="hisup")
    output_dir = ctx.output_dir()
    ann_name = _ann_name(ctx, data_root)
    require_paths([source / "tools" / "train.py", data_root / "train" / ann_name, data_root / "val" / ann_name])

    if ctx.is_inria:
        stage1_cfg = source / "configs" / "inria_hisup" / "mask2former_r50_query-300_24e_inria-hisup.py"
        stage2_cfg = source / "configs" / "inria_hisup" / "gcp_r50_query-300_24e_inria-hisup.py"
    else:
        stage1_cfg = source / "configs" / "deventer_512" / "mask2former_r50_query-300_24e_deventer-512.py"
        stage2_cfg = source / "configs" / "deventer_512" / "gcp_r50_query-300_24e_deventer-512.py"
    require_paths([stage1_cfg, stage2_cfg])

    cwd = source
    stage1_epochs = ctx.param("STAGE1_EPOCHS", 1 if ctx.smoke else 24)
    stage2_epochs = ctx.param("STAGE2_EPOCHS", 1 if ctx.smoke else 24)
    stage1_dir = output_dir / "stage1_mask2former"
    stage2_dir = output_dir / "stage2_gcp"
    stage1_ckpt = stage1_dir / f"epoch_{stage1_epochs}.pth"

    common_env = ctx.env(
        PYTHONPATH=f"{ctx.path(source, cwd)}:{ctx.param('PYTHONPATH', '')}",
        GCP_DEVENTER_DATA_ROOT=ctx.path(data_root, cwd),
        GCP_DEVENTER_SINGLE_CLASS=ctx.task,
        GCP_DEVENTER_TRAIN_ANN=f"train/{ann_name}",
        GCP_DEVENTER_VAL_ANN=f"val/{ann_name}",
        GCP_DEVENTER_TEST_ANN=f"test/{ann_name}",
    )

    stage1_opts = [
        f"train_dataloader.dataset.data_root={ctx.path(data_root, cwd)}",
        f"val_dataloader.dataset.data_root={ctx.path(data_root, cwd)}",
        f"test_dataloader.dataset.data_root={ctx.path(data_root, cwd)}",
        f"train_dataloader.dataset.ann_file=train/{ann_name}",
        f"val_dataloader.dataset.ann_file=val/{ann_name}",
        f"test_dataloader.dataset.ann_file=val/{ann_name}",
        f"val_evaluator.0.ann_file={ctx.path(data_root / 'val' / ann_name, cwd)}",
        f"test_evaluator.0.ann_file={ctx.path(data_root / 'val' / ann_name, cwd)}",
        f"max_epochs={stage1_epochs}",
        f"train_cfg.max_epochs={stage1_epochs}",
        f"train_cfg.val_interval={ctx.param('EVAL_INTERVAL', 1)}",
        f"train_dataloader.batch_size={ctx.param('STAGE1_BATCH_SIZE', 8 if ctx.smoke else 24)}",
        f"train_dataloader.num_workers={ctx.param('STAGE1_NUM_WORKERS', 8)}",
        f"val_dataloader.num_workers={ctx.param('STAGE1_NUM_WORKERS', 8)}",
        f"test_dataloader.num_workers={ctx.param('STAGE1_NUM_WORKERS', 8)}",
        f"optim_wrapper.optimizer.lr={ctx.param('STAGE1_LR', '1e-4')}",
        f"default_hooks.checkpoint.interval={ctx.param('CKPT_INTERVAL', 1)}",
        "default_hooks.checkpoint.save_last=True",
    ]
    stage2_opts = [
        f"train_dataloader.dataset.data_root={ctx.path(data_root, cwd)}",
        f"val_dataloader.dataset.data_root={ctx.path(data_root, cwd)}",
        f"test_dataloader.dataset.data_root={ctx.path(data_root, cwd)}",
        f"train_dataloader.dataset.ann_file=train/{ann_name}",
        f"val_dataloader.dataset.ann_file=val/{ann_name}",
        f"test_dataloader.dataset.ann_file=val/{ann_name}",
        f"val_evaluator.0.ann_file={ctx.path(data_root / 'val' / ann_name, cwd)}",
        f"test_evaluator.0.ann_file={ctx.path(data_root / 'val' / ann_name, cwd)}",
        f"load_from={ctx.path(stage1_ckpt, cwd)}",
        f"model.panoptic_head.type={ctx.param('STAGE2_HEAD_TYPE', 'PolygonizerHead')}",
        f"model.test_mode={ctx.param('STAGE2_TEST_MODE', 'normal')}",
        f"max_epochs={stage2_epochs}",
        f"train_cfg.max_epochs={stage2_epochs}",
        f"train_cfg.val_interval={ctx.param('EVAL_INTERVAL', 1)}",
        f"train_dataloader.batch_size={ctx.param('STAGE2_BATCH_SIZE', 8 if ctx.smoke else 24)}",
        f"train_dataloader.num_workers={ctx.param('STAGE2_NUM_WORKERS', 8)}",
        f"val_dataloader.num_workers={ctx.param('STAGE2_NUM_WORKERS', 8)}",
        f"test_dataloader.num_workers={ctx.param('STAGE2_NUM_WORKERS', 8)}",
        f"optim_wrapper.optimizer.lr={ctx.param('STAGE2_LR', '1e-4')}",
        f"default_hooks.checkpoint.interval={ctx.param('CKPT_INTERVAL', 1)}",
        "default_hooks.checkpoint.save_last=True",
    ]
    steps: list[CommandStep] = []
    if ctx.mode in {"train", "train_eval"}:
        steps.append(CommandStep(
            "gcp_stage1_train",
            cwd=cwd,
            env=common_env,
            argv=ctx.python(env_name, source / "tools" / "train.py", [
                ctx.path(stage1_cfg, cwd),
                "--work-dir", ctx.path(stage1_dir, cwd),
                "--cfg-options", *stage1_opts,
            ], cwd),
        ))
        steps.append(CommandStep(
            "gcp_stage2_train",
            cwd=cwd,
            env=common_env,
            argv=ctx.python(env_name, source / "tools" / "train.py", [
                ctx.path(stage2_cfg, cwd),
                "--work-dir", ctx.path(stage2_dir, cwd),
                "--cfg-options", *stage2_opts,
            ], cwd),
        ))
    return steps
