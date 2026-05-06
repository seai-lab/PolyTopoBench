import csv
import functools
import os
import re

import torch
import torch.utils.data

from frame_field_learning import data_transforms
from lydorn_utils import print_utils


def inria_aerial_train_tile_filter(tile, train_val_split_point):
    return tile["number"] <= train_val_split_point


def inria_aerial_val_tile_filter(tile, train_val_split_point):
    return train_val_split_point < tile["number"]


def inria_aerial_number_tile_filter(tile, numbers):
    return tile["number"] in numbers


def inria_aerial_name_tile_filter(tile, tile_names):
    return f"{tile['city']}{tile['number']}" in tile_names


def deventer_aerial_name_tile_filter(tile, tile_names):
    return tile["name_stem"] in tile_names


def load_inria_aerial_split_csv(root_dir, split_csv):
    split_csv_path = split_csv if os.path.isabs(split_csv) else os.path.join(root_dir, split_csv)
    if not os.path.exists(split_csv_path):
        raise FileNotFoundError(f"split_csv not found: {split_csv_path}")

    split_to_tile_names = {}
    with open(split_csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_fields = {"image_name", "split"}
        missing_fields = required_fields - set(reader.fieldnames or [])
        if missing_fields:
            raise ValueError(
                f"split_csv is missing required columns {sorted(missing_fields)}: {split_csv_path}"
            )
        for row in reader:
            image_name = row["image_name"].strip()
            split_name = row["split"].strip()
            tile_name, _ = os.path.splitext(image_name)
            split_to_tile_names.setdefault(split_name, set()).add(tile_name)
    return split_to_tile_names, split_csv_path


def sanitize_processed_tag(text):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text)


def get_inria_aerial_folds(config, root_dir, folds):
    from torch_lydorn.torchvision.datasets import InriaAerial

    # --- Online transform done on the host (CPU):
    online_cpu_transform = data_transforms.get_online_cpu_transform(config,
                                                                    augmentations=config["data_aug_params"]["enable"])
    mask_only = config["dataset_params"]["mask_only"]
    kwargs = {
        "pre_process": config["dataset_params"]["pre_process"],
        "transform": online_cpu_transform,
        "patch_size": config["dataset_params"]["data_patch_size"],
        "patch_stride": config["dataset_params"]["input_patch_size"],
        "pre_transform": data_transforms.get_offline_transform_patch(distances=not mask_only, sizes=not mask_only),
        "small": config["dataset_params"]["small"],
        "pool_size": config["num_workers"],
        "gt_source": config["dataset_params"]["gt_source"],
        "gt_type": config["dataset_params"]["gt_type"],
        "gt_dirname": config["dataset_params"]["gt_dirname"],
        "mask_only": mask_only,
        "process_only_filtered_tiles": config["dataset_params"].get("process_only_filtered_tiles", False),
        "preprocess_tile_filter": None,
        "processed_tag": None,
    }
    if "processed_dirname" in config["dataset_params"]:
        kwargs["processed_dirname"] = config["dataset_params"]["processed_dirname"]
    if "split_csv" in config["dataset_params"]:
        split_to_tile_names, split_csv_path = load_inria_aerial_split_csv(
            root_dir, config["dataset_params"]["split_csv"]
        )
        print_utils.print_info(f'Using split_csv from {split_csv_path}')
        kwargs["processed_tag"] = sanitize_processed_tag(
            f"split_{os.path.splitext(os.path.basename(split_csv_path))[0]}"
        )
        train_tile_names = split_to_tile_names.get("train", set())
        val_tile_names = split_to_tile_names.get("val", set())
        partial_train_tile_filter = functools.partial(inria_aerial_name_tile_filter, tile_names=train_tile_names)
        partial_val_tile_filter = functools.partial(inria_aerial_name_tile_filter, tile_names=val_tile_names)

        if kwargs["process_only_filtered_tiles"]:
            preprocess_tile_names = train_tile_names | val_tile_names
            kwargs["preprocess_tile_filter"] = functools.partial(
                inria_aerial_name_tile_filter, tile_names=preprocess_tile_names
            )
    elif "train_numbers" in config["dataset_params"] or "val_numbers" in config["dataset_params"]:
        train_numbers = set(config["dataset_params"].get("train_numbers", []))
        val_numbers = set(config["dataset_params"].get("val_numbers", []))
        partial_train_tile_filter = functools.partial(inria_aerial_number_tile_filter, numbers=train_numbers)
        partial_val_tile_filter = functools.partial(inria_aerial_number_tile_filter, numbers=val_numbers)

        if kwargs["process_only_filtered_tiles"]:
            preprocess_numbers = train_numbers | val_numbers
            kwargs["preprocess_tile_filter"] = functools.partial(inria_aerial_number_tile_filter, numbers=preprocess_numbers)
    else:
        train_val_split_point = config["dataset_params"]["train_fraction"] * 36
        partial_train_tile_filter = functools.partial(inria_aerial_train_tile_filter, train_val_split_point=train_val_split_point)
        partial_val_tile_filter = functools.partial(inria_aerial_val_tile_filter, train_val_split_point=train_val_split_point)

    ds_list = []
    for fold in folds:
        if fold == "train":
            ds = InriaAerial(root_dir, fold="train", tile_filter=partial_train_tile_filter, **kwargs)
            ds_list.append(ds)
        elif fold == "val":
            ds = InriaAerial(root_dir, fold="train", tile_filter=partial_val_tile_filter, **kwargs)
            ds_list.append(ds)
        elif fold == "train_val":
            ds = InriaAerial(root_dir, fold="train", **kwargs)
            ds_list.append(ds)
        elif fold == "test":
            ds = InriaAerial(root_dir, fold="test", **kwargs)
            ds_list.append(ds)
        else:
            print_utils.print_error("ERROR: fold \"{}\" not recognized, implement it in dataset_folds.py.".format(fold))

    return ds_list


def get_deventer_aerial_folds(config, root_dir, folds):
    from torch_lydorn.torchvision.datasets import DeventerAerial

    online_cpu_transform = data_transforms.get_online_cpu_transform(
        config,
        augmentations=config["data_aug_params"]["enable"],
    )
    mask_only = config["dataset_params"]["mask_only"]
    kwargs = {
        "pre_process": config["dataset_params"]["pre_process"],
        "transform": online_cpu_transform,
        "patch_size": config["dataset_params"]["data_patch_size"],
        "patch_stride": config["dataset_params"]["input_patch_size"],
        "pre_transform": data_transforms.get_offline_transform_patch(distances=not mask_only, sizes=not mask_only),
        "small": config["dataset_params"]["small"],
        "pool_size": config["num_workers"],
        "gt_source": config["dataset_params"]["gt_source"],
        "gt_type": config["dataset_params"]["gt_type"],
        "gt_dirname": config["dataset_params"]["gt_dirname"],
        "mask_only": mask_only,
        "process_only_filtered_tiles": config["dataset_params"].get("process_only_filtered_tiles", False),
        "preprocess_tile_filter": None,
        "processed_tag": None,
        "split_csv_path": None,
    }
    if "processed_dirname" in config["dataset_params"]:
        kwargs["processed_dirname"] = config["dataset_params"]["processed_dirname"]
    if "split_csv" in config["dataset_params"]:
        split_to_tile_names, split_csv_path = load_inria_aerial_split_csv(
            root_dir, config["dataset_params"]["split_csv"]
        )
        print_utils.print_info(f'Using split_csv from {split_csv_path}')
        kwargs["split_csv_path"] = os.path.abspath(split_csv_path)
        kwargs["processed_tag"] = sanitize_processed_tag(
            f"split_{os.path.splitext(os.path.basename(split_csv_path))[0]}"
        )
        train_tile_names = split_to_tile_names.get("train", set())
        val_tile_names = split_to_tile_names.get("val", set())
        partial_train_tile_filter = functools.partial(deventer_aerial_name_tile_filter, tile_names=train_tile_names)
        partial_val_tile_filter = functools.partial(deventer_aerial_name_tile_filter, tile_names=val_tile_names)

        if kwargs["process_only_filtered_tiles"]:
            preprocess_tile_names = train_tile_names | val_tile_names
            kwargs["preprocess_tile_filter"] = functools.partial(
                deventer_aerial_name_tile_filter, tile_names=preprocess_tile_names
            )
    else:
        raise ValueError("DeventerAerial requires dataset_params.split_csv to define train/val folds.")

    ds_list = []
    for fold in folds:
        if fold == "train":
            ds = DeventerAerial(root_dir, fold="train", tile_filter=partial_train_tile_filter, **kwargs)
            ds_list.append(ds)
        elif fold == "val":
            ds = DeventerAerial(root_dir, fold="train", tile_filter=partial_val_tile_filter, **kwargs)
            ds_list.append(ds)
        elif fold == "train_val":
            ds = DeventerAerial(root_dir, fold="train", **kwargs)
            ds_list.append(ds)
        elif fold == "test":
            # The train/val split CSV should not constrain preprocessing for the
            # held-out test fold, otherwise no test tiles get materialized.
            test_kwargs = dict(kwargs)
            test_kwargs["preprocess_tile_filter"] = None
            ds = DeventerAerial(root_dir, fold="test", **test_kwargs)
            ds_list.append(ds)
        else:
            print_utils.print_error(f'ERROR: fold "{fold}" not recognized, implement it in dataset_folds.py.')
    return ds_list


def get_folds(config, root_dir, folds):
    assert set(folds).issubset({"train", "val", "train_val", "test"}), \
        'fold in folds should be in ["train", "val", "train_val", "test"]'

    if config["dataset_params"].get("dataset_adapter") == "deventer_ffl":
        return get_deventer_aerial_folds(config, root_dir, folds)

    elif config["dataset_params"]["root_dirname"] in {"AerialImageDataset", "inria_building"}:
        return get_inria_aerial_folds(config, root_dir, folds)

    else:
        print_utils.print_error("ERROR: config[\"data_root_partial_dirpath\"] = \"{}\" is an unknown dataset! "
                                "If it is a new dataset, add it in dataset_folds.py's get_folds() function.".format(
            config["dataset_params"]["root_dirname"]))
        exit()
