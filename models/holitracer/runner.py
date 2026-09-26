from __future__ import annotations

from pathlib import Path

from utilis.common import CommandStep, RunContext, require_paths

from .polytopobench_steps import build_work_dataset


DEFAULT_ENV = "polytopobench"
DATASET_NAME = "polytopobench"
TRAIN_STAGES = ("seg_h5", "seg_train", "seg_infer", "vector_prep", "vector_train")


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _layout(ctx: RunContext) -> dict[str, Path]:
    # HoliTracer engines join os.getcwd() with every configured path, so all paths
    # handed to them are absolute and live under the run's output dir.
    work_root = ctx.output_dir().resolve()
    corner = ctx.param("CORNER_ANGLE_THRESHOLD", 135)
    vector_tag = f"{corner}_{ctx.param('INTERPOLATION_DISTANCE', 25)}_{ctx.param('SAMPLING_SIZE', 32)}"
    return {
        "work_root": work_root,
        "data": work_root / "work_dataset",
        "h5": work_root / "h5",
        "configs": work_root / "configs",
        "seg_run": work_root / "seg_run",
        "seg_pred": work_root / "seg_predictions",
        "vector_run": work_root / "vector_run",
        # SEG_CHECKPOINT / VECTOR_CHECKPOINT let mode=eval score existing weights.
        "seg_best": Path(ctx.param("SEG_CHECKPOINT", "")).expanduser().resolve() if ctx.param("SEG_CHECKPOINT", "")
        else work_root / "seg_run" / DATASET_NAME / "1_3_6" / "swin_l" / "best_model.pth",
        "vector_train_dir": work_root / "vector_run" / DATASET_NAME / "vlras" / "train" / vector_tag,
        "vector_weights": work_root / "val_inference" / "vector_weights.pth",
        "infer_dir": work_root / "val_inference",
    }


def _image_dir(split_root: Path) -> Path:
    return split_root / "img_tif" if (split_root / "img_tif").exists() else split_root / "img"


def prepare(ctx: RunContext) -> None:
    data_root = ctx.data_dir("holitracer")
    lay = _layout(ctx)
    build_work_dataset(data_root, lay["data"], ctx.smoke, "coco_label_with_inter_smoke.json")
    view_size = ctx.param("VIEW_SIZE", 512)
    image_size = ctx.param("IMAGE_SIZE", 512)
    common_vector = f"""down_ratio: 4
dataset: "{DATASET_NAME}"
num_points: {ctx.param('SAMPLING_SIZE', 32)}
corner_threshold: {ctx.param('CORNER_THRESHOLD', 0.1)}
d: {ctx.param('INTERPOLATION_DISTANCE', 25)}
backbone_path: "{lay['seg_best']}"
num_workers: {ctx.param('VECTOR_NUM_WORKERS', 8)}
batch_size: {ctx.param('VECTOR_BATCH_SIZE', 4 if ctx.smoke else 8)}
epochs: {ctx.param('VECTOR_EPOCHS', 1 if ctx.smoke else 10)}
run_dir: "{lay['vector_run']}"
distributed: false
dist_backend: "nccl"
dist_url: "env://"
"""

    _write(lay["configs"] / "seg_train.yaml", f"""device: "cuda"
backbone: "swin_l"
segHead: "upernet"
nclass: 2
isContext: True
pretrain: False
resume: null
seed: 2333
learn_rate: {ctx.param('SEG_LEARN_RATE', '0.00001')}
lr_patience: 3
ignore_index: -1
data_root: "{lay['h5']}"
dataset: "{DATASET_NAME}"
downsample_factors: [1, 3, 6]
num_workers: {ctx.param('SEG_NUM_WORKERS', 8)}
batch_size: {ctx.param('SEG_BATCH_SIZE', 2 if ctx.smoke else 7)}
epochs: {ctx.param('SEG_EPOCHS', 1 if ctx.smoke else 10)}
early_stopping_patience: 5
run_dir: "{lay['seg_run']}"
distributed: false
dist_backend: "nccl"
dist_url: "env://"
""")
    for split in ("train", "val"):
        _write(lay["configs"] / f"seg_infer_{split}.yaml", f"""device: "cuda"
dataset: "{DATASET_NAME}"
backbone: "swin_l"
segHead: "upernet"
ignore_index: -1
isContext: True
crop_size: {image_size}
seed: 2333
run_dir: "{lay['seg_run']}"
batch_size: {ctx.param('SEG_INFER_BATCH_SIZE', 2 if ctx.smoke else 4)}
num_workers: {ctx.param('SEG_INFER_NUM_WORKERS', 8)}
distributed: false
dist_backend: "nccl"
dist_url: "env://"
image_path: "{_image_dir(lay['data'] / split)}"
result_dir: "{lay['seg_pred'] / split / 'seg'}"
view_size: {view_size}
downsample_factors: [1, 3, 6]
nclass: 2
resume: "{lay['seg_best']}"
""")
    _write(lay["configs"] / "vector_train.yaml", f"""device: "cuda"
resume: null
seed: 2333
model: "vlras"
loss_type: "angle"
corner_angle_threshold: {ctx.param('CORNER_ANGLE_THRESHOLD', 135)}
noncorner_angle_threshold: {ctx.param('NONCORNER_ANGLE_THRESHOLD', 135)}
learn_rate: {ctx.param('VECTOR_LEARN_RATE', '0.001')}
lr_patience: 5
weight_regression: 1.0
weight_classification: 1.0
weight_angle: 1.0
train_h5_path: "{lay['h5'] / (DATASET_NAME + '_vector_train.h5')}"
image_path: "{lay['data']}"
eval_whole: {"true" if ctx.is_inria else "false"}
{common_vector}""")
    _write(lay["configs"] / "vector_infer.yaml", f"""device: "cuda"
resume: "{lay['vector_weights']}"
seed: 2333
model: "vlras"
corner_angle_threshold: {ctx.param('CORNER_ANGLE_THRESHOLD', 135)}
noncorner_angle_threshold: {ctx.param('NONCORNER_ANGLE_THRESHOLD', 135)}
learn_rate: {ctx.param('VECTOR_LEARN_RATE', '0.001')}
lr_patience: 5
image_path: "{lay['data'] / 'val' / 'img'}"
result_json: "{lay['infer_dir'] / 'holitracer_val.coco.json'}"
coco_predictions: "{lay['data'] / 'val' / 'predict' / 'holitracer' / 'holitracer.json'}"
coco_labels: "{lay['data'] / 'val' / 'coco_label_with_inter.json'}"
visual: false
run_metric: false
{common_vector}""")


def build_commands(ctx: RunContext) -> list[CommandStep]:
    env_name = ctx.env_name or DEFAULT_ENV
    source = ctx.source_root()
    helper = ctx.model_root() / "polytopobench_steps.py"
    data_root = ctx.data_dir("holitracer")
    lay = _layout(ctx)
    require_paths([source / "tools" / "seg_train.py", source / "tools" / "vector_train.py", data_root / "train", data_root / "val"])
    cwd = source
    py_env = ctx.env(PYTHONPATH=f"{ctx.path(source, cwd)}:{ctx.param('PYTHONPATH', '')}")
    stage = ctx.param("PIPELINE_STAGE", "all")
    train = ctx.mode in {"train", "train_eval"}
    evaluate = ctx.run_val and ctx.mode in {"infer", "eval", "train_eval"}

    def want(name: str) -> bool:
        return stage in {"all", name}

    def cfg(name: str) -> list[str]:
        return ["--config", ctx.path(lay["configs"] / name, cwd)]

    steps: list[CommandStep] = []
    if train and want("seg_h5"):
        for split in ("train", "val"):
            steps.append(CommandStep(f"holitracer_seg_h5_{split}", cwd=cwd, argv=ctx.python(env_name, source / "tools" / "dataset" / "make_seg_h5.py", [
                "--view_size", ctx.param("VIEW_SIZE", 512),
                "--downsample_factors", "1", "3", "6",
                "--image_path", ctx.path(lay["data"] / split / "img", cwd),
                "--label_path", ctx.path(lay["data"] / split / "mask", cwd),
                "--output_hdf5", ctx.path(lay["h5"] / f"{DATASET_NAME}_seg_{split}.h5", cwd),
                "--max_images", ctx.param("MAX_H5_IMAGES", 2 if ctx.smoke else 0),
            ], cwd)))
    if train and want("seg_train"):
        steps.append(CommandStep("holitracer_seg_train", cwd=cwd, env=py_env, argv=ctx.python(env_name, source / "tools" / "seg_train.py", cfg("seg_train.yaml"), cwd)))

    # Seg masks + COCO seeds: train split feeds the vector-stage training data, val split
    # feeds both the vector trainer's model selection (Inria eval_whole) and val inference.
    seg_splits = (["train"] if train and want("seg_infer") else []) + (["val"] if (train and want("seg_infer")) or (evaluate and want("vector_infer")) else [])
    # seg_infer skips images whose mask already exists: drop old masks whenever the
    # seg weights may have changed (retrained here, or SEG_CHECKPOINT given).
    if seg_splits and ((train and want("seg_train")) or ctx.param("SEG_CHECKPOINT", "")):
        steps.append(CommandStep("holitracer_clean_stale_masks", cwd=ctx.release_root, argv=ctx.python(
            env_name, helper, ["clean", *[ctx.path(lay["seg_pred"] / s / "seg", ctx.release_root) for s in seg_splits]], ctx.release_root)))
    for split in seg_splits:
        steps.append(CommandStep(f"holitracer_seg_infer_{split}", cwd=cwd, env=py_env, argv=ctx.python(env_name, source / "tools" / "seg_infer.py", cfg(f"seg_infer_{split}.yaml"), cwd)))
    coco_splits = (["train"] if train and want("vector_prep") else []) + (["val"] if (train and want("vector_prep")) or (evaluate and want("vector_infer")) else [])
    # Smoke only (VECTOR_TRAIN_SEEDS=gt): seed the vector-stage training data with GT
    # polygons, see polytopobench_steps.py gt-seeds. Full runs always use seg predictions.
    gt_train_seeds = ctx.smoke and ctx.param("VECTOR_TRAIN_SEEDS", "gt") == "gt"
    for split in coco_splits:
        if split == "train" and gt_train_seeds:
            steps.append(CommandStep("holitracer_smoke_gt_seeds_train", cwd=ctx.release_root, argv=ctx.python(env_name, helper, [
                "gt-seeds",
                "--labels", ctx.path(lay["data"] / "train" / "coco_label_with_inter.json", ctx.release_root),
                "--out", ctx.path(lay["data"] / "train" / "predict" / "holitracer" / "holitracer.json", ctx.release_root),
            ], ctx.release_root)))
            continue
        steps.append(CommandStep(f"holitracer_mask_to_coco_{split}", cwd=cwd, argv=ctx.python(env_name, source / "tools" / "trans" / "mask_to_coco.py", [
            "--ground_truth_json", ctx.path(lay["data"] / split / "coco_label_with_inter.json", cwd),
            "--masks_directory", ctx.path(lay["seg_pred"] / split / "seg", cwd),
            "--output_json", ctx.path(lay["data"] / split / "predict" / "holitracer" / "holitracer.json", cwd),
            "--dataset", DATASET_NAME,
            "--simplify_value", "0.0",
            "-n", ctx.param("MASK_TO_COCO_NPROC", 8),
        ], cwd)))
    if train and want("vector_prep"):
        # make_vector_h5 dumps debug PNGs to ./visual_match, so run it from the output dir.
        h5_cwd = lay["work_root"]
        steps.append(CommandStep("holitracer_vector_h5", cwd=h5_cwd, env=ctx.env(PYTHONPATH=f"{source.resolve()}:{ctx.param('PYTHONPATH', '')}"), argv=ctx.python(env_name, source / "tools" / "dataset" / "make_vector_h5.py", [
            "--pred_coco_file", ctx.path(lay["data"] / "train" / "predict" / "holitracer" / "holitracer.json", h5_cwd),
            "--gt_coco_file", ctx.path(lay["data"] / "train" / "coco_label_with_inter.json", h5_cwd),
            "--image_dir", ctx.path(lay["data"] / "train" / "img", h5_cwd),
            "--interpolation_distance", ctx.param("INTERPOLATION_DISTANCE", 25),
            "--sampling_size", ctx.param("SAMPLING_SIZE", 32),
            "--sliding_step", ctx.param("SLIDING_STEP", 16),
            "--output_file", ctx.path(lay["h5"] / f"{DATASET_NAME}_vector_train.h5", h5_cwd),
            "--process_num", ctx.param("VECTOR_PROCESS_NUM", 8),
        ], h5_cwd)))
    if train and want("vector_train"):
        steps.append(CommandStep("holitracer_vector_train", cwd=cwd, env=py_env, argv=ctx.python(env_name, source / "tools" / "vector_train.py", cfg("vector_train.yaml"), cwd)))

    if evaluate and want("vector_infer"):
        rel = ctx.release_root
        pred_path = lay["infer_dir"] / "predictions_hisup.json"
        gt_path = ctx.data_dir("hisup") / "val" / ("annotation-smoke.json" if ctx.smoke else "annotation.json")
        require_paths([gt_path])
        select_args = ["select-weights", "--run-dir", ctx.path(lay["vector_train_dir"], rel), "--out", ctx.path(lay["vector_weights"], rel)]
        if ctx.param("VECTOR_CHECKPOINT", ""):
            select_args += ["--weights", str(Path(ctx.param("VECTOR_CHECKPOINT", "")).expanduser().resolve())]
        steps.append(CommandStep("holitracer_select_vector_weights", cwd=rel, argv=ctx.python(env_name, helper, select_args, rel)))
        steps.append(CommandStep("holitracer_vector_infer_val", cwd=cwd, env=py_env, argv=ctx.python(env_name, source / "tools" / "vector_infer.py", cfg("vector_infer.yaml"), cwd)))
        steps.append(CommandStep("holitracer_to_hisup", cwd=rel, argv=ctx.python(env_name, helper, [
            "to-hisup",
            "--input", ctx.path(lay["infer_dir"] / "holitracer_val.coco.json", rel),
            "--labels", ctx.path(lay["data"] / "val" / "coco_label_with_inter.json", rel),
            "--gt", ctx.path(gt_path, rel),
            "--output", ctx.path(pred_path, rel),
        ], rel)))
        # Smoke runs infer on the smoke val subset, scored against the canonical smoke GT
        # (same image ids); full runs use the canonical val GT.
        steps.append(ctx.evaluate_step(name="holitracer_eval", pred=pred_path, gt=gt_path, env_name=env_name, pred_type="hisup"))
    return steps
