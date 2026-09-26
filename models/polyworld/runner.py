from __future__ import annotations

from pathlib import Path

from utilis.common import CommandStep, RunContext, require_paths


DEFAULT_ENV = "polytopobench"


def build_commands(ctx: RunContext) -> list[CommandStep]:
    env_name = ctx.env_name or DEFAULT_ENV
    source = ctx.source_root()
    data_root = ctx.data_dir("hisup") if ctx.is_inria else ctx.data_dir("polyworld", fallback="hisup")
    output_dir = ctx.output_dir()
    ann_name = "annotation-smoke.json" if ctx.smoke and (data_root / "train" / "annotation-smoke.json").exists() else "annotation.json"
    require_paths([source / "train_inria_multiring.py", source / "prediction.py"])

    cwd = source
    window_size = ctx.param("WINDOW_SIZE", 320)
    num_workers = ctx.param("NUM_WORKERS", 4 if ctx.smoke else 8)
    steps: list[CommandStep] = []
    if ctx.mode in {"train", "train_eval"}:
        require_paths([data_root / "train" / "images", data_root / "train" / ann_name, data_root / "val" / ann_name])
        args = [
            "--train-images-directory", ctx.path(data_root / "train" / "images", cwd),
            "--train-annotations-path", ctx.path(data_root / "train" / ann_name, cwd),
            "--val-images-directory", ctx.path(data_root / "val" / "images", cwd),
            "--val-annotations-path", ctx.path(data_root / "val" / ann_name, cwd),
            "--output-dir", ctx.path(output_dir, cwd),
            "--window-size", window_size,
            "--max-points", ctx.param("MAX_POINTS", 256),
            "--target-rings", ctx.param("TARGET_RINGS", "all"),
            "--batch-size", ctx.param("BATCH_SIZE", 4 if ctx.smoke else 16),
            "--num-workers", num_workers,
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
        steps.append(CommandStep(
            "polyworld_train",
            cwd=cwd,
            env=ctx.env(PYTHONNOUSERSITE="1"),
            argv=ctx.python(env_name, source / "train_inria_multiring.py", args, cwd),
        ))

    if ctx.run_val and ctx.mode in {"infer", "eval", "train_eval"}:
        # Inference always runs on the canonical hisup val split so prediction image ids are the
        # evaluator GT ids. Smoke runs use the same-id 16-image subset (annotation-smoke.json).
        gt_json = ctx.hisup_gt_json()
        canonical_val = gt_json.parent
        infer_ann = canonical_val / "annotation-smoke.json" if ctx.smoke and (canonical_val / "annotation-smoke.json").exists() else gt_json
        weights_name = ctx.param("WEIGHTS", "weights_best")
        weights_dir = Path(weights_name) if Path(weights_name).is_absolute() else output_dir / weights_name
        pred_json = output_dir / f"predictions_val_{weights_dir.name}.json"
        require_paths([canonical_val / "images", infer_ann])
        steps.append(CommandStep(
            "polyworld_predict",
            cwd=cwd,
            env=ctx.env(PYTHONNOUSERSITE="1"),
            argv=ctx.python(env_name, source / "prediction.py", [
                "--images-directory", ctx.path(canonical_val / "images", cwd),
                "--annotations-path", ctx.path(infer_ann, cwd),
                "--output-json", ctx.path(pred_json, cwd),
                "--weights-dir", ctx.path(weights_dir, cwd),
                "--window-size", window_size,
                "--batch-size", ctx.param("PRED_BATCH_SIZE", 6),
                "--num-workers", num_workers,
                "--n-peaks", ctx.param("N_PEAKS", 256),
            ], cwd),
        ))
        # PolyWorld emits one record per closed ring; rebuild exterior/hole hierarchy by containment.
        steps.append(ctx.evaluate_step(name="polyworld_eval", pred=pred_json, gt=gt_json, env_name=env_name, cwd=cwd, pred_type="roipoly"))
    return steps
