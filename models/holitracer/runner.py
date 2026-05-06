from __future__ import annotations

from pathlib import Path

from utilis.common import CommandStep, RunContext, require_paths


DEFAULT_ENV = "polytopobench"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def prepare(ctx: RunContext) -> None:
    data_root = ctx.data_dir("holitracer")
    work_root = ctx.output_dir()
    h5_root = work_root / "h5"
    config_root = work_root / "configs"
    seg_run_dir = work_root / "seg_run"
    seg_pred_root = work_root / "seg_predictions"
    vector_run_dir = work_root / "vector_run"
    dataset_name = "polytopobench"
    view_size = ctx.param("VIEW_SIZE", 512)
    image_size = ctx.param("IMAGE_SIZE", 512)
    seg_resume = seg_run_dir / dataset_name / "1_3_6" / "swin_l" / "best_model.pth"
    train_img_tif = data_root / "train" / "img_tif"
    val_img_tif = data_root / "val" / "img_tif"
    train_img = train_img_tif if train_img_tif.exists() else data_root / "train" / "img"
    val_img = val_img_tif if val_img_tif.exists() else data_root / "val" / "img"

    _write(config_root / "seg_train.yaml", f"""device: "cuda"
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
data_root: "{ctx.path(h5_root, config_root)}"
dataset: "{dataset_name}"
downsample_factors: [1, 3, 6]
num_workers: {ctx.param('SEG_NUM_WORKERS', 8)}
batch_size: {ctx.param('SEG_BATCH_SIZE', 2 if ctx.smoke else 7)}
epochs: {ctx.param('SEG_EPOCHS', 1 if ctx.smoke else 10)}
early_stopping_patience: 5
run_dir: "{ctx.path(seg_run_dir, config_root)}"
distributed: false
dist_backend: "nccl"
dist_url: "env://"
""")
    _write(config_root / "seg_infer_train.yaml", f"""device: "cuda"
dataset: "{dataset_name}"
backbone: "swin_l"
segHead: "upernet"
ignore_index: -1
isContext: True
crop_size: {image_size}
seed: 2333
run_dir: "{ctx.path(seg_run_dir, config_root)}"
batch_size: {ctx.param('SEG_INFER_BATCH_SIZE', 2 if ctx.smoke else 4)}
num_workers: {ctx.param('SEG_INFER_NUM_WORKERS', 8)}
distributed: false
dist_backend: "nccl"
dist_url: "env://"
image_path: "{ctx.path(train_img, config_root)}"
result_dir: "{ctx.path(seg_pred_root / 'train' / 'seg', config_root)}"
view_size: {view_size}
downsample_factors: [1, 3, 6]
nclass: 2
resume: "{ctx.path(seg_resume, config_root)}"
""")
    _write(config_root / "seg_infer_val.yaml", f"""device: "cuda"
dataset: "{dataset_name}"
backbone: "swin_l"
segHead: "upernet"
ignore_index: -1
isContext: True
crop_size: {image_size}
seed: 2333
run_dir: "{ctx.path(seg_run_dir, config_root)}"
batch_size: {ctx.param('SEG_INFER_BATCH_SIZE', 2 if ctx.smoke else 4)}
num_workers: {ctx.param('SEG_INFER_NUM_WORKERS', 8)}
distributed: false
dist_backend: "nccl"
dist_url: "env://"
image_path: "{ctx.path(val_img, config_root)}"
result_dir: "{ctx.path(seg_pred_root / 'val' / 'seg', config_root)}"
view_size: {view_size}
downsample_factors: [1, 3, 6]
nclass: 2
resume: "{ctx.path(seg_resume, config_root)}"
""")
    _write(config_root / "vector_train.yaml", f"""device: "cuda"
resume: null
seed: 2333
model: "vlras"
loss_type: "angle"
corner_angle_threshold: {ctx.param('CORNER_ANGLE_THRESHOLD', 135)}
noncorner_angle_threshold: {ctx.param('NONCORNER_ANGLE_THRESHOLD', 135)}
learn_rate: {ctx.param('VECTOR_LEARN_RATE', '0.001')}
lr_patience: 5
down_ratio: 4
weight_regression: 1.0
weight_classification: 1.0
weight_angle: 1.0
dataset: "{dataset_name}"
train_h5_path: "{ctx.path(h5_root / (dataset_name + '_vector_train.h5'), config_root)}"
image_path: "{ctx.path(data_root, config_root)}"
num_points: {ctx.param('SAMPLING_SIZE', 32)}
corner_threshold: {ctx.param('CORNER_THRESHOLD', 0.1)}
d: {ctx.param('INTERPOLATION_DISTANCE', 25)}
backbone_path: "{ctx.path(seg_resume, config_root)}"
num_workers: {ctx.param('VECTOR_NUM_WORKERS', 8)}
batch_size: {ctx.param('VECTOR_BATCH_SIZE', 4 if ctx.smoke else 8)}
epochs: {ctx.param('VECTOR_EPOCHS', 1 if ctx.smoke else 10)}
eval_whole: {"true" if ctx.is_inria else "false"}
run_dir: "{ctx.path(vector_run_dir, config_root)}"
distributed: false
dist_backend: "nccl"
dist_url: "env://"
""")


def build_commands(ctx: RunContext) -> list[CommandStep]:
    env_name = ctx.env_name or DEFAULT_ENV
    source = ctx.source_root()
    data_root = ctx.data_dir("holitracer")
    work_root = ctx.output_dir()
    h5_root = work_root / "h5"
    config_root = work_root / "configs"
    seg_pred_root = work_root / "seg_predictions"
    dataset_name = "polytopobench"
    require_paths([source / "tools" / "seg_train.py", source / "tools" / "vector_train.py", data_root / "train", data_root / "val"])
    cwd = source
    steps: list[CommandStep] = []
    stage = ctx.param("PIPELINE_STAGE", "all")
    if stage in {"all", "seg_h5"}:
        steps.extend([
            CommandStep("holitracer_seg_h5_train", cwd=cwd, argv=ctx.python(env_name, source / "tools" / "dataset" / "make_seg_h5.py", [
                "--view_size", ctx.param("VIEW_SIZE", 512),
                "--downsample_factors", "1", "3", "6",
                "--image_path", ctx.path(data_root / "train" / "img", cwd),
                "--label_path", ctx.path(data_root / "train" / "mask", cwd),
                "--output_hdf5", ctx.path(h5_root / f"{dataset_name}_seg_train.h5", cwd),
                "--max_images", ctx.param("MAX_H5_IMAGES", 2 if ctx.smoke else 0),
            ], cwd)),
            CommandStep("holitracer_seg_h5_val", cwd=cwd, argv=ctx.python(env_name, source / "tools" / "dataset" / "make_seg_h5.py", [
                "--view_size", ctx.param("VIEW_SIZE", 512),
                "--downsample_factors", "1", "3", "6",
                "--image_path", ctx.path(data_root / "val" / "img", cwd),
                "--label_path", ctx.path(data_root / "val" / "mask", cwd),
                "--output_hdf5", ctx.path(h5_root / f"{dataset_name}_seg_val.h5", cwd),
                "--max_images", ctx.param("MAX_H5_IMAGES", 2 if ctx.smoke else 0),
            ], cwd)),
        ])
    if stage in {"all", "seg_train"}:
        steps.append(CommandStep("holitracer_seg_train", cwd=cwd, env=ctx.env(PYTHONPATH=f"{ctx.path(source, cwd)}:{ctx.param('PYTHONPATH', '')}"), argv=ctx.python(env_name, source / "tools" / "seg_train.py", ["--config", ctx.path(config_root / "seg_train.yaml", cwd)], cwd)))
    if stage in {"all", "seg_infer"}:
        steps.append(CommandStep("holitracer_seg_infer_train", cwd=cwd, env=ctx.env(PYTHONPATH=f"{ctx.path(source, cwd)}:{ctx.param('PYTHONPATH', '')}"), argv=ctx.python(env_name, source / "tools" / "seg_infer.py", ["--config", ctx.path(config_root / "seg_infer_train.yaml", cwd)], cwd)))
        steps.append(CommandStep("holitracer_seg_infer_val", cwd=cwd, env=ctx.env(PYTHONPATH=f"{ctx.path(source, cwd)}:{ctx.param('PYTHONPATH', '')}"), argv=ctx.python(env_name, source / "tools" / "seg_infer.py", ["--config", ctx.path(config_root / "seg_infer_val.yaml", cwd)], cwd)))
    if stage in {"all", "vector_prep"}:
        steps.append(CommandStep("holitracer_mask_to_coco", cwd=cwd, argv=ctx.python(env_name, source / "tools" / "trans" / "mask_to_coco.py", [
            "--ground_truth_json", ctx.path(data_root / "train" / "coco_label_with_inter.json", cwd),
            "--masks_directory", ctx.path(seg_pred_root / "train" / "seg", cwd),
            "--output_json", ctx.path(seg_pred_root / "train" / "holitracer.json", cwd),
            "--dataset", dataset_name,
            "--simplify_value", "0.0",
            "-n", ctx.param("MASK_TO_COCO_NPROC", 8),
        ], cwd)))
        steps.append(CommandStep("holitracer_vector_h5", cwd=cwd, argv=ctx.python(env_name, source / "tools" / "dataset" / "make_vector_h5.py", [
            "--pred_coco_file", ctx.path(seg_pred_root / "train" / "holitracer.json", cwd),
            "--gt_coco_file", ctx.path(data_root / "train" / "coco_label_with_inter.json", cwd),
            "--image_dir", ctx.path(data_root / "train" / "img", cwd),
            "--interpolation_distance", ctx.param("INTERPOLATION_DISTANCE", 25),
            "--sampling_size", ctx.param("SAMPLING_SIZE", 32),
            "--sliding_step", ctx.param("SLIDING_STEP", 16),
            "--output_file", ctx.path(h5_root / f"{dataset_name}_vector_train.h5", cwd),
            "--process_num", ctx.param("VECTOR_PROCESS_NUM", 8),
        ], cwd)))
    if stage in {"all", "vector_train"}:
        steps.append(CommandStep("holitracer_vector_train", cwd=cwd, env=ctx.env(PYTHONPATH=f"{ctx.path(source, cwd)}:{ctx.param('PYTHONPATH', '')}"), argv=ctx.python(env_name, source / "tools" / "vector_train.py", ["--config", ctx.path(config_root / "vector_train.yaml", cwd)], cwd)))
    return steps
