from __future__ import annotations

from utilis.common import CommandStep, RunContext, require_paths


DEFAULT_ENV = "polytopobench"


def build_commands(ctx: RunContext) -> list[CommandStep]:
    env_name = ctx.env_name or DEFAULT_ENV
    source = ctx.source_root()
    data_root = ctx.data_dir("roipoly")
    output_dir = ctx.output_dir()
    ann_name = "annotation_roipoly_smoke.json" if ctx.smoke and (data_root / "train" / "annotation_roipoly_smoke.json").exists() else "annotation_roipoly.json"
    config = source / "configs" / ("roipoly.res50.160pro.64corner.inria512.yaml" if ctx.is_inria else "roipoly.res50.160pro.64corner.deventer512.yaml")
    require_paths([source / "scripts" / "train_net.py", source / "scripts" / "evaluate.py", config, data_root / "train" / ann_name, data_root / "val" / ann_name])

    cwd = source
    pythonpath = ":".join([
        ctx.path(source, cwd),
        ctx.path(source / "roipoly" / "ops", cwd),
        ctx.param("PYTHONPATH", ""),
    ])
    steps: list[CommandStep] = []
    if ctx.mode in {"train", "train_eval"}:
        steps.append(CommandStep(
            "roipoly_train",
            cwd=cwd,
            env=ctx.env(PYTHONPATH=pythonpath),
            argv=ctx.python(env_name, source / "scripts" / "train_net.py", [
                "--num-gpus", "1",
                "--config-file", ctx.path(config, cwd),
                "--dataset-name", "inria_train" if ctx.is_inria else "deventer_train",
                "--train-json", ctx.path(data_root / "train" / ann_name, cwd),
                "--train-path", ctx.path(data_root / "train" / "images", cwd),
                "MODEL.WEIGHTS", "detectron2://ImageNetPretrained/torchvision/R-50.pkl",
                "OUTPUT_DIR", ctx.path(output_dir, cwd),
                "SOLVER.IMS_PER_BATCH", ctx.param("IMS_PER_BATCH", 8),
                "SOLVER.BASE_LR", ctx.param("BASE_LR", "0.000025"),
                "SOLVER.MAX_ITER", ctx.param("MAX_ITER", 2 if ctx.smoke else 135450),
                "SOLVER.CHECKPOINT_PERIOD", ctx.param("CHECKPOINT_PERIOD", 1 if ctx.smoke else 6020),
                "SOLVER.STEPS", f"({ctx.param('STEP1', 1 if ctx.smoke else 90300)},)",
                "DATALOADER.NUM_WORKERS", ctx.param("NUM_WORKERS", 8),
            ], cwd),
        ))

    if ctx.run_val and ctx.mode in {"eval", "train_eval"}:
        eval_split = ctx.param("EVAL_SPLIT", "val")
        eval_output = output_dir / f"eval_sparsercnn_{eval_split}"
        detector_prefix = ctx.output_root / "sparsercnn" / ctx.dataset / ctx.task / ("res50_160pro_512_smoke" if ctx.smoke else "res50_160pro_512")
        detector_eval = ctx.output_root / "sparsercnn" / ctx.dataset / ctx.task / f"{detector_prefix.name}_eval_{eval_split}"
        proposal_json = detector_eval / "inference" / f"annotation_sparsercnn_{eval_split}_for_roipoly{'_smoke' if ctx.smoke else ''}.json"
        train_script = source / "detection_baselines" / "SparseRCNN" / "scripts" / ("train_inria_sparsercnn.sh" if ctx.is_inria else "train_deventer_sparsercnn.sh")
        eval_script = source / "detection_baselines" / "SparseRCNN" / "scripts" / ("eval_inria_sparsercnn.sh" if ctx.is_inria else "eval_deventer_sparsercnn.sh")
        convert_script = source / "detection_baselines" / "SparseRCNN" / "scripts" / ("convert_inria_sparsercnn_predictions_to_roipoly_json.sh" if ctx.is_inria else "convert_deventer_sparsercnn_predictions_to_roipoly_json.sh")
        require_paths([train_script, eval_script, convert_script])
        if ctx.param("RUN_DETECTOR_TRAIN", 0) == "1":
            steps.append(CommandStep(
                "roipoly_sparsercnn_train",
                cwd=cwd,
                env=ctx.env(DATA_ROOT=ctx.path(data_root, cwd), GPU_ID=ctx.gpu, CONDA_ENV=env_name, OUTPUT_DIR=ctx.path(detector_prefix, cwd)),
                argv=["bash", ctx.path(train_script, cwd)],
            ))
        steps.append(CommandStep(
            "roipoly_sparsercnn_eval",
            cwd=cwd,
            env=ctx.env(DATA_ROOT=ctx.path(data_root, cwd), GPU_ID=ctx.gpu, CONDA_ENV=env_name, MODEL_WEIGHTS=ctx.path(detector_prefix / "model_final.pth", cwd), OUTPUT_DIR=ctx.path(detector_eval, cwd)),
            argv=["bash", ctx.path(eval_script, cwd)],
        ))
        steps.append(CommandStep(
            "roipoly_sparsercnn_convert",
            cwd=cwd,
            env=ctx.env(DATA_ROOT=ctx.path(data_root, cwd), CONDA_ENV=env_name, DETECTOR_OUTPUT_DIR=ctx.path(detector_eval, cwd), PREDICTIONS_JSON=ctx.path(detector_eval / "inference" / "coco_instances_results.json", cwd), SAVE_PATH=ctx.path(proposal_json, cwd)),
            argv=["bash", ctx.path(convert_script, cwd)],
        ))
        test_json = proposal_json
        dataset_name = ("inria" if ctx.is_inria else "deventer") + f"_{eval_split}_sparsercnn_bbox"
        steps.append(CommandStep(
            "roipoly_eval",
            cwd=cwd,
            env=ctx.env(PYTHONPATH=pythonpath),
            argv=ctx.python(env_name, source / "scripts" / "evaluate.py", [
                "--config-file", ctx.path(config, cwd),
                "--dataset-name", dataset_name,
                "--test-json", ctx.path(test_json, cwd),
                "--test-path", ctx.path(data_root / eval_split / "images", cwd),
                "--output", ctx.path(eval_output, cwd),
                "--corner-threshold", ctx.param("CORNER_THRESHOLD", 0.4),
                "--batch-size", ctx.param("EVAL_BATCH_SIZE", 2),
                "--opts",
                "MODEL.WEIGHTS", ctx.path(output_dir / "model_final.pth", cwd),
                "OUTPUT_DIR", ctx.path(output_dir, cwd),
            ], cwd),
        ))
    return steps
