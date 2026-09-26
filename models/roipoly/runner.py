from __future__ import annotations

from pathlib import Path

from utilis.common import CommandStep, RunContext, require_paths


DEFAULT_ENV = "polytopobench"
R50_IMAGENET = "detectron2://ImageNetPretrained/torchvision/R-50.pkl"


def build_commands(ctx: RunContext) -> list[CommandStep]:
    env_name = ctx.env_name or DEFAULT_ENV
    source = ctx.source_root()
    sparsercnn = source / "detection_baselines" / "SparseRCNN"
    data_root = ctx.data_dir("roipoly")
    output_dir = ctx.output_dir()
    use_smoke_ann = ctx.smoke and (data_root / "train" / "annotation_roipoly_smoke.json").exists()
    ann_name = "annotation_roipoly_smoke.json" if use_smoke_ann else "annotation_roipoly.json"
    tag = "inria" if ctx.is_inria else "deventer"
    config = source / "configs" / f"roipoly.res50.160pro.64corner.{tag}512.yaml"
    require_paths([source / "scripts" / "train_net.py", source / "scripts" / "evaluate.py", config, data_root / "train" / ann_name])

    cwd = source
    pythonpath = ":".join([
        ctx.path(source, cwd),
        ctx.path(source / "roipoly" / "ops", cwd),
        ctx.param("PYTHONPATH", ""),
    ])
    max_iter = ctx.param("MAX_ITER", 2 if ctx.smoke else 135450)
    steps: list[CommandStep] = []
    if ctx.mode in {"train", "train_eval"}:
        steps.append(CommandStep(
            "roipoly_train",
            cwd=cwd,
            env=ctx.env(PYTHONPATH=pythonpath),
            argv=ctx.python(env_name, source / "scripts" / "train_net.py", [
                "--num-gpus", "1",
                "--config-file", ctx.path(config, cwd),
                "--dataset-name", f"{tag}_train",
                "--train-json", ctx.path(data_root / "train" / ann_name, cwd),
                "--train-path", ctx.path(data_root / "train" / "images", cwd),
                "MODEL.WEIGHTS", R50_IMAGENET,
                "OUTPUT_DIR", ctx.path(output_dir, cwd),
                "SOLVER.IMS_PER_BATCH", ctx.param("IMS_PER_BATCH", 8),
                "SOLVER.BASE_LR", ctx.param("BASE_LR", "0.000025"),
                "SOLVER.MAX_ITER", max_iter,
                "SOLVER.CHECKPOINT_PERIOD", ctx.param("CHECKPOINT_PERIOD", 1 if ctx.smoke else 6020),
                "SOLVER.STEPS", f"({ctx.param('STEP1', 1 if ctx.smoke else 90300)},)",
                "DATALOADER.NUM_WORKERS", ctx.param("NUM_WORKERS", 8),
            ], cwd),
        ))

    if ctx.run_val and ctx.mode in {"infer", "eval", "train_eval"}:
        # RoIPoly decodes one polygon per proposal box; the paper setting feeds it Sparse R-CNN
        # detections. Smoke runs use the 16-image val subset (same image ids as the full split).
        eval_split = ctx.param("EVAL_SPLIT", "val")
        split_root = data_root / eval_split
        split_ann = split_root / ann_name
        require_paths([split_ann, split_root / "images"])
        det_config = sparsercnn / "configs" / f"sparsercnn.res50.160pro.{tag}512.yaml"
        det_name = ctx.param("DETECTOR_RUN_NAME", "res50_160pro_512_smoke" if ctx.smoke else "res50_160pro_512")
        det_dir = ctx.output_root / "sparsercnn" / ctx.dataset / ctx.task / det_name
        det_weights = Path(ctx.param("DETECTOR_WEIGHTS", det_dir / "model_final.pth"))
        det_eval_dir = det_dir.parent / f"{det_name}_eval_{eval_split}"
        det_results = det_eval_dir / "inference" / "coco_instances_results.json"
        proposal_json = det_eval_dir / "inference" / f"annotation_sparsercnn_{eval_split}_for_roipoly{'_smoke' if ctx.smoke else ''}.json"
        eval_dir = output_dir / f"eval_sparsercnn_{eval_split}"
        pred_json = eval_dir / "predictions.json"
        det_env = ctx.env(PYTHONPATH=f"{ctx.path(sparsercnn, sparsercnn)}:{ctx.param('PYTHONPATH', '')}")
        det_data_args = [
            "--num-gpus", "1",
            "--config-file", ctx.path(det_config, sparsercnn),
            "--train-dataset", f"{tag}_train",
            "--train-json", ctx.path(data_root / "train" / ann_name, sparsercnn),
            "--train-path", ctx.path(data_root / "train" / "images", sparsercnn),
            "--val-dataset", f"{tag}_{eval_split}",
            "--val-json", ctx.path(split_ann, sparsercnn),
            "--val-path", ctx.path(split_root / "images", sparsercnn),
        ]
        det_workers = ctx.param("DET_NUM_WORKERS", 0 if ctx.smoke else (4 if ctx.is_inria else 8))

        # RUN_DETECTOR_TRAIN: 1 = always train, 0 = never (weights must exist), auto = train if missing.
        run_det_train = ctx.param("RUN_DETECTOR_TRAIN", "auto")
        if run_det_train == "1" or (run_det_train == "auto" and not det_weights.exists()):
            det_steps = ctx.param("DET_STEPS", "(1,)" if ctx.smoke else "(27090, 33198)")
            steps.append(CommandStep(
                "roipoly_sparsercnn_train",
                cwd=sparsercnn,
                env=det_env,
                argv=ctx.python(env_name, sparsercnn / "train_net.py", [
                    *det_data_args,
                    "MODEL.WEIGHTS", R50_IMAGENET,
                    "OUTPUT_DIR", ctx.path(det_dir, sparsercnn),
                    "SOLVER.IMS_PER_BATCH", ctx.param("DET_IMS_PER_BATCH", 2 if ctx.smoke else 4),
                    "SOLVER.BASE_LR", ctx.param("DET_BASE_LR", "0.000025"),
                    "SOLVER.MAX_ITER", ctx.param("DET_MAX_ITER", 2 if ctx.smoke else 36120),
                    "SOLVER.STEPS", det_steps,
                    "SOLVER.CHECKPOINT_PERIOD", ctx.param("DET_CHECKPOINT_PERIOD", 1 if ctx.smoke else 1505),
                    "TEST.EVAL_PERIOD", ctx.param("DET_EVAL_PERIOD", 0 if ctx.smoke else 1505),
                    "DATALOADER.NUM_WORKERS", det_workers,
                ], sparsercnn),
            ))
            det_weights = det_dir / "model_final.pth"
        elif not det_weights.exists():
            raise FileNotFoundError(f"Sparse R-CNN weights not found: {det_weights} (set RUN_DETECTOR_TRAIN=auto/1 or DETECTOR_WEIGHTS=...)")
        steps.append(CommandStep(
            "roipoly_sparsercnn_infer",
            cwd=sparsercnn,
            env=det_env,
            argv=ctx.python(env_name, sparsercnn / "train_net.py", [
                *det_data_args,
                "--eval-only",
                "MODEL.WEIGHTS", ctx.path(det_weights, sparsercnn),
                "OUTPUT_DIR", ctx.path(det_eval_dir, sparsercnn),
                "DATALOADER.NUM_WORKERS", det_workers,
            ], sparsercnn),
        ))
        steps.append(CommandStep(
            "roipoly_sparsercnn_to_proposals",
            cwd=cwd,
            env=ctx.env(),
            argv=ctx.python(env_name, source / "data_preprocessing" / "predictions_to_coco.py", [
                "--json_path", ctx.path(det_results, cwd),
                "--annotation_path", ctx.path(split_ann, cwd),
                "--save_path", ctx.path(proposal_json, cwd),
                "--type", "all",
                "--score_threshold", ctx.param("PROPOSAL_SCORE_THRESHOLD", 0.0 if ctx.smoke else 0.05),
                "--topk_per_image", ctx.param("PROPOSAL_TOPK", 160),
            ], cwd),
        ))
        roipoly_weights = Path(ctx.param("ROIPOLY_WEIGHTS", output_dir / "model_final.pth"))
        steps.append(CommandStep(
            "roipoly_infer",
            cwd=cwd,
            env=ctx.env(PYTHONPATH=pythonpath),
            argv=ctx.python(env_name, source / "scripts" / "evaluate.py", [
                "--config-file", ctx.path(config, cwd),
                "--dataset-name", f"{tag}_{eval_split}_sparsercnn_bbox",
                "--test-json", ctx.path(proposal_json, cwd),
                "--test-path", ctx.path(split_root / "images", cwd),
                "--output", ctx.path(eval_dir, cwd),
                "--pred-json", ctx.path(pred_json, cwd),
                "--corner-threshold", ctx.param("CORNER_THRESHOLD", 0.4),
                "--batch-size", ctx.param("EVAL_BATCH_SIZE", 2),
                "--opts",
                "MODEL.WEIGHTS", ctx.path(roipoly_weights, cwd),
                "OUTPUT_DIR", ctx.path(eval_dir, cwd),
            ], cwd),
        ))
        # One record per ring; exterior/hole hierarchy is rebuilt by containment. metrics.json lands in
        # eval_dir because detectron2 already writes its own training-log metrics.json into output_dir.
        gt = ctx.hisup_gt_json() if eval_split == "val" else ctx.data_dir("hisup") / eval_split / "annotation.json"
        steps.append(ctx.evaluate_step(name="roipoly_eval", pred=pred_json, gt=gt, env_name=env_name, cwd=cwd, pred_type="roipoly"))
    return steps
