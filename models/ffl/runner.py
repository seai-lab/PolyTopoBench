from __future__ import annotations

import csv
import json
from pathlib import Path

from utilis.common import CommandStep, RunContext, require_paths


DEFAULT_ENV = "polytopobench"


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_smoke_split(source_csv: Path, dest_csv: Path, train_count: int, val_count: int) -> Path:
    selected: list[dict[str, str]] = []
    counts = {"train": 0, "val": 0}
    with source_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or ["image_name", "split"]
        for row in reader:
            split = row.get("split", "").strip()
            if split == "train" and counts["train"] < train_count:
                selected.append(row)
                counts["train"] += 1
            elif split == "val" and counts["val"] < val_count:
                selected.append(row)
                counts["val"] += 1
            if counts["train"] >= train_count and counts["val"] >= val_count:
                break
    if not selected:
        raise ValueError(f"No smoke rows found in split CSV: {source_csv}")

    dest_csv.parent.mkdir(parents=True, exist_ok=True)
    with dest_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)
    return dest_csv


def prepare(ctx: RunContext) -> None:
    source = ctx.source_root()
    data_root = ctx.data_dir("ffl")
    tmp_dir = ctx.output_dir() / "tmp_configs"
    if ctx.is_inria:
        defaults = _load(source / "configs" / "config.defaults.inria_building_polygonized.json")
        dataset_params = _load(source / "configs" / "dataset_params.inria_building_polygonized.json")
        train_cfg = _load(source / "configs" / "config.inria_building_polygonized.unet_resnet101_csv_train.json")
        eval_cfg = _load(source / "configs" / "config.inria_building_polygonized.unet_resnet101_csv_eval_acm_fast_viz.json")
        root_dirname = "inria_building"
        split_csv = data_root / "inria_building" / "train" / ("image_split_smoke_train_val.csv" if ctx.smoke else "image_split_official_train_val.csv")
        data_base_root = data_root
        tag = "inria_building"
    else:
        defaults = _load(source / "configs" / "config.defaults.deventer_512_ffl_building.json")
        dataset_params = _load(source / "configs" / "dataset_params.deventer_512_ffl_building.json")
        train_cfg = _load(source / "configs" / "config.deventer_512_ffl_building.unet_resnet101_train.json")
        eval_cfg = _load(source / "configs" / "config.deventer_512_ffl_building.unet_resnet101_eval_acm_fast_viz.json")
        root_dirname = f"ffl/{ctx.task}"
        split_csv = data_root / ("train/image_split_smoke_train_val.csv" if ctx.smoke else "train/image_split_official_train_val.csv")
        data_base_root = ctx.dataset_root
        tag = f"deventer_512_ffl_{ctx.task}"

    if ctx.smoke:
        split_csv = _write_smoke_split(
            split_csv,
            tmp_dir / "image_split_smoke_runtime.csv",
            int(ctx.param("SMOKE_TRAIN_TILES", 1)),
            int(ctx.param("SMOKE_VAL_TILES", 1)),
        )
        dataset_params["process_only_filtered_tiles"] = True

    dataset_root_for_split = data_base_root / root_dirname
    dataset_params["root_dirname"] = root_dirname
    if ctx.smoke:
        dataset_params["processed_dirname"] = ctx.path(
            ctx.output_dir() / "tmp_ffl_processed" / "processed",
            dataset_root_for_split,
        )
    dataset_params["split_csv"] = ctx.path(split_csv, dataset_root_for_split)
    dataset_params["small"] = False
    dataset_params_path = tmp_dir / f"dataset_params.{tag}.json"
    defaults_path = tmp_dir / f"config.defaults.{tag}.json"
    train_path = tmp_dir / f"config.{tag}.train.json"
    eval_path = tmp_dir / f"config.{tag}.eval.json"

    num_workers = max(1, int(ctx.param("NUM_WORKERS", 8)))
    defaults["num_workers"] = num_workers
    defaults["data_dir_candidates"] = [ctx.path(data_base_root, source)]
    defaults["dataset_params"]["defaults_filepath"] = ctx.path(dataset_params_path, source)

    train_cfg["defaults_filepath"] = ctx.path(defaults_path, source)
    train_cfg["run_name"] = ctx.run_name
    train_cfg["num_workers"] = num_workers
    train_cfg.setdefault("optim_params", {})
    train_cfg["optim_params"]["checkpoint_epoch"] = int(ctx.param("CHECKPOINT_EPOCH", 1 if ctx.smoke else 5))

    eval_cfg["defaults_filepath"] = ctx.path(defaults_path, source)
    eval_cfg["run_name"] = ctx.run_name
    eval_cfg["num_workers"] = num_workers
    eval_cfg.setdefault("optim_params", {})
    eval_cfg["optim_params"]["batch_size"] = int(ctx.param("EVAL_BATCH_SIZE", 2))
    eval_cfg.setdefault("eval_params", {})
    eval_cfg["eval_params"]["results_dirname"] = ctx.path(ctx.output_dir() / "results", source)
    eval_cfg.setdefault("polygonize_params", {}).setdefault("acm_method", {})
    eval_cfg["polygonize_params"]["acm_method"]["steps"] = int(ctx.param("POLY_STEPS", 20))
    eval_cfg["polygonize_params"]["acm_method"]["poly_lr"] = float(ctx.param("POLY_LR", 0.01))

    _dump(dataset_params_path, dataset_params)
    _dump(defaults_path, defaults)
    _dump(train_path, train_cfg)
    _dump(eval_path, eval_cfg)


def build_commands(ctx: RunContext) -> list[CommandStep]:
    env_name = ctx.env_name or DEFAULT_ENV
    source = ctx.source_root()
    data_root = ctx.data_dir("ffl")
    tmp_dir = ctx.output_dir() / "tmp_configs"
    tag = "inria_building" if ctx.is_inria else f"deventer_512_ffl_{ctx.task}"
    train_config = tmp_dir / f"config.{tag}.train.json"
    eval_config = tmp_dir / f"config.{tag}.eval.json"
    require_paths([source / "main.py", data_root])

    cwd = source
    ffl_runtime_cwd = source / "frame_field_learning"
    runs_dir = ctx.output_root / "ffl_runs"
    steps: list[CommandStep] = []
    if ctx.mode in {"train", "train_eval"}:
        env = ctx.env()
        if ctx.smoke:
            env["FFL_SKIP_LOSS_NORM_INIT"] = "1"
        steps.append(CommandStep(
            "ffl_train",
            cwd=cwd,
            env=env,
            argv=ctx.python(env_name, source / "main.py", [
                "--config", ctx.path(train_config, cwd),
                "--new_run",
                "--fold", "train",
                "--gpus", "1",
                "--runs_dirpath", ctx.path(runs_dir, ffl_runtime_cwd),
                "--batch_size", ctx.param("TRAIN_BATCH_SIZE", 1 if ctx.smoke else 13),
                "--max_epoch", ctx.param("MAX_EPOCH", 1 if ctx.smoke else 5),
            ], cwd),
        ))
    if ctx.run_val and ctx.mode in {"eval", "train_eval"}:
        steps.append(CommandStep(
            "ffl_eval",
            cwd=cwd,
            env=ctx.env(),
            argv=ctx.python(env_name, source / "main.py", [
                "--config", ctx.path(eval_config, cwd),
                "--mode", "eval",
                "--fold", "val",
                "--gpus", "1",
                "--runs_dirpath", ctx.path(runs_dir, ffl_runtime_cwd),
                "--run_name", ctx.run_name,
                "--eval_batch_size", ctx.param("EVAL_BATCH_SIZE", 2),
            ], cwd),
        ))
    return steps
