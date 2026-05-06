from __future__ import annotations

from utilis.common import CommandStep, RunContext, require_paths


DEFAULT_ENV = "polytopobench"


def build_commands(ctx: RunContext) -> list[CommandStep]:
    env_name = ctx.env_name or DEFAULT_ENV
    source = ctx.source_root()
    data_root = ctx.data_dir("acpvnet")
    output_dir = ctx.output_dir()
    config_name = "inria_building_hrnet32v2_512_vh_m_ldm_kl4_b8_custom.yaml" if ctx.is_inria else "deventer_512_hrnet32v2_singleclass_rot0.yaml"
    config = source / "config-files" / config_name
    require_paths([source / "scripts" / "train_iter.py", source / "scripts" / "test.py", config, data_root / "train", data_root / "val"])

    ims_per_batch = ctx.param("IMS_PER_BATCH", 2 if ctx.smoke else 8)
    total_iters = ctx.param("TOTAL_ITERS", 2 if ctx.smoke else 400000)
    checkpoint_period = ctx.param("CHECKPOINT_PERIOD", 1 if ctx.smoke else 5000)
    num_workers = ctx.param("NUM_WORKERS", 8)
    cwd = source
    pythonpath = ":".join([
        ctx.path(source, cwd),
        ctx.path(source / "kernels" / "selective_scan", cwd),
        ctx.param("PYTHONPATH", ""),
    ])
    common_env = ctx.env(
        PYTHONPATH=pythonpath,
        ACPV_DATA_DIR=ctx.path(data_root, cwd),
    )
    steps: list[CommandStep] = []
    if ctx.mode in {"train", "train_eval"}:
        steps.append(CommandStep(
            "acpvnet_train",
            cwd=cwd,
            env=common_env,
            argv=ctx.python(env_name, source / "scripts" / "train_iter.py", [
                "--config-file", ctx.path(config, cwd),
                "--seed", str(ctx.seed),
                "OUTPUT_DIR", ctx.path(output_dir, cwd),
                "SOLVER.IMS_PER_BATCH", ims_per_batch,
                "SOLVER.BASE_LR", ctx.param("BASE_LR", "6e-5"),
                "SOLVER.TOTAL_ITERS", total_iters,
                "SOLVER.CHECKPOINT_PERIOD", checkpoint_period,
                "DATALOADER.NUM_WORKERS", num_workers,
            ], cwd),
        ))
    if ctx.run_val and ctx.mode in {"eval", "train_eval"}:
        steps.append(CommandStep(
            "acpvnet_eval",
            cwd=cwd,
            env=common_env,
            argv=ctx.python(env_name, source / "scripts" / "test.py", [
                "--config-file", ctx.path(config, cwd),
                "--eval-type", ctx.param("EVAL_TYPE", "coco_iou"),
                "--sampler", ctx.param("SAMPLER", "direct"),
                "--ddim-steps", ctx.param("DDIM_STEPS", 200),
                "OUTPUT_DIR", ctx.path(output_dir, cwd),
                "DATALOADER.NUM_WORKERS", num_workers,
            ], cwd),
        ))
    return steps
