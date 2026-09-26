from __future__ import annotations

from pathlib import Path

from utilis.common import CommandStep, RunContext, require_paths


DEFAULT_ENV = "polytopobench"
LATENT_DIR = Path("train") / "heatmap_augmented_latent_kl-4" / "rot0"


def latent_scale_factor(ctx: RunContext, data_root: Path) -> str:
    """Scale factor used when the training latents were encoded (needed to decode predictions)."""
    override = ctx.param("ACPV_SCALE_FACTOR", "")
    if override:
        return override
    path = data_root / LATENT_DIR / "scale_factor.txt"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Build the ACPV-Net mirror with "
            "`prepare_data.py --methods acpvnet --encode-acpv-latents`, or pass --set ACPV_SCALE_FACTOR=<value>."
        )
    return path.read_text(encoding="utf-8").strip()


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
    # Diffusion schedule length must be identical at train and test time; default = config value.
    ddpm_opts = ["DDPM_TIMESTEPS", ctx.params["DDPM_TIMESTEPS"]] if "DDPM_TIMESTEPS" in ctx.params else []
    steps: list[CommandStep] = []
    if ctx.mode in {"train", "train_eval"}:
        require_paths([data_root / LATENT_DIR / "z"])
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
                *ddpm_opts,
            ], cwd),
        ))
    if not (ctx.run_val and ctx.mode in {"infer", "eval", "train_eval"}):
        return steps

    # Validation: test.py (seg masks + latent vertex heatmaps) -> decode latents -> extract vertices
    # -> PSLG polygonization into hisup records -> unified evaluator (pred-type hisup).
    # Smoke runs use the canonical val/annotation-smoke.json subset (same image ids as the full GT).
    gt_json = ctx.hisup_gt_json()
    smoke_gt = gt_json.with_name("annotation-smoke.json")
    if ctx.smoke and ctx.param("SMOKE_EVAL_SUBSET", 1) != "0" and smoke_gt.is_file():
        gt_json = smoke_gt
    ae_config = Path(ctx.param("ACPV_AUTOENCODER_CONFIG", source / "config-files" / "autoencoder_kl_f4.yaml"))
    require_paths([ae_config, gt_json])
    scale_factor = latent_scale_factor(ctx, data_root)
    sampler = ctx.param("SAMPLER", "direct")
    sampler_dir = output_dir / sampler
    heatmap_npy = sampler_dir / "vertex_heatmap_npy"
    vertex_dir = sampler_dir / "vertices"
    pred_json = output_dir / "predictions_hisup.json"
    eval_env = {**common_env, "ACPV_EVAL_ANN_FILE": ctx.path(gt_json, cwd)}

    steps.append(CommandStep(
        "acpvnet_infer",
        cwd=cwd,
        env=eval_env,
        argv=ctx.python(env_name, source / "scripts" / "test.py", [
            "--config-file", ctx.path(config, cwd),
            "--eval-type", ctx.param("EVAL_TYPE", "coco_iou"),
            "--sampler", sampler,
            "--ddim-steps", ctx.param("DDIM_STEPS", 200),
            "OUTPUT_DIR", ctx.path(output_dir, cwd),
            "DATASETS.TEST", "('custom_acpv_val_with_latent_vertex_heatmap',)",
            "DATALOADER.NUM_WORKERS", num_workers,
            *ddpm_opts,
        ], cwd),
    ))
    steps.append(CommandStep(
        "acpvnet_decode_latents",
        cwd=cwd,
        env=common_env,
        argv=ctx.python(env_name, source / "latent_decoder_accelerated.py", [
            "--config", ctx.path(ae_config, cwd),
            "--latent_dir", ctx.path(sampler_dir / "vertex_heatmap_latent_pt", cwd),
            "--output_dir", ctx.path(sampler_dir / "vertex_heatmap_viz", cwd),
            "--npy_dir", ctx.path(heatmap_npy, cwd),
            "--scale_factor", scale_factor,
            "--type", "heatmap",
            "--batch-size", ctx.param("DECODE_BATCH_SIZE", 8),
            "--num-workers", num_workers,
        ], cwd),
    ))
    steps.append(CommandStep(
        "acpvnet_extract_vertices",
        cwd=cwd,
        env=common_env,
        argv=ctx.python(env_name, source / "tools" / "extract_vertices_from_heatmap.py", [
            "--input_dir", ctx.path(heatmap_npy, cwd),
            "--save_dir", ctx.path(vertex_dir, cwd),
            "--threshold", ctx.param("VERTEX_THRESHOLD", 0.1),
            "--topk", ctx.param("VERTEX_TOPK", 1000),
            "--kernel_size", ctx.param("VERTEX_NMS_KERNEL", 3),
        ], cwd),
    ))
    steps.append(CommandStep(
        "acpvnet_polygonize_to_hisup",
        cwd=cwd,
        env=common_env,
        argv=ctx.python(env_name, ctx.model_root() / "convert_acpvnet_to_hisup.py", [
            "--seg-dir", ctx.path(output_dir / "seg_mask_npy", cwd),
            "--vertex-dir", ctx.path(vertex_dir, cwd),
            "--prob-dir", ctx.path(output_dir / "prob_map_npy", cwd),
            "--gt-json", ctx.path(gt_json, cwd),
            "--out-json", ctx.path(pred_json, cwd),
            "--dist-thresh", ctx.param("PSLG_DIST_THRESH", 5.0),
            "--corner-eps", ctx.param("PSLG_CORNER_EPS", 2.0),
            "--num-workers", ctx.param("POLYGONIZE_NUM_WORKERS", 8),
        ], cwd),
    ))
    steps.append(ctx.evaluate_step(
        name="acpvnet_unified_eval",
        pred=pred_json,
        gt=gt_json,
        env_name=env_name,
        pred_type="hisup",
        gt_type="hisup",
    ))
    return steps
