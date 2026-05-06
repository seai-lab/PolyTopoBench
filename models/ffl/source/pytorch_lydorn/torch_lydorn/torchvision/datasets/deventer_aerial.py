import fnmatch
import csv
import os.path
import pathlib

import multiprocess
import numpy as np
import shapely.geometry
import skimage.io
import torch
import torch.utils.data

from tqdm import tqdm

from lydorn_utils import image_utils, polygon_utils, print_utils, python_utils


IMAGE_DIRNAME = "images"
DEFAULT_IMAGE_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
DEFAULT_IMAGE_STD = np.array([0.25, 0.25, 0.25], dtype=np.float32)
_NUMPY_COMPAT_ALIASES = {
    "bool": np.bool_,
    "int": int,
    "float": float,
    "complex": complex,
    "object": object,
}
for _alias_name, _alias_value in _NUMPY_COMPAT_ALIASES.items():
    if _alias_name not in np.__dict__:
        setattr(np, _alias_name, _alias_value)


class DeventerAerial(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        fold: str = "train",
        pre_process: bool = True,
        tile_filter=None,
        patch_size: int = None,
        patch_stride: int = None,
        pre_transform=None,
        transform=None,
        small: bool = False,
        pool_size: int = 1,
        raw_dirname: str = "raw",
        processed_dirname: str = "processed",
        gt_source: str = "disk",
        gt_type: str = "geojson",
        gt_dirname: str = "gt_polygonized",
        mask_only: bool = False,
        process_only_filtered_tiles: bool = False,
        preprocess_tile_filter=None,
        processed_tag: str = None,
        split_csv_path: str = None,
    ):
        assert gt_source in {"disk", "osm"}, "gt_source should be disk or osm"
        assert gt_type in {"npy", "geojson", "tif"}, f"gt_type should be npy, geojson or tif, not {gt_type}"
        self.root = root
        self.fold = fold
        self.pre_process = pre_process
        self.tile_filter = tile_filter
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.pre_transform = pre_transform
        self.transform = transform
        self.small = small
        self.pool_size = pool_size
        self.raw_dirname = raw_dirname
        self.gt_source = gt_source
        self.gt_type = gt_type
        self.gt_dirname = gt_dirname
        self.mask_only = mask_only
        self.process_only_filtered_tiles = process_only_filtered_tiles
        self.preprocess_tile_filter = preprocess_tile_filter
        self.processed_tag = processed_tag
        self.split_csv_path = split_csv_path
        self.tile_metadata = self.load_tile_metadata()

        if self.pre_process:
            processed_dirname_extension = f"{processed_dirname}.source_{self.gt_source}.type_{self.gt_type}"
            if self.gt_dirname is not None:
                processed_dirname_extension += f".dirname_{self.gt_dirname}"
            if self.mask_only:
                processed_dirname_extension += f".mask_only_{int(self.mask_only)}"
            processed_dirname_extension += f".patch_size_{int(self.patch_size)}"
            if self.processed_tag:
                processed_dirname_extension += f".tag_{self.processed_tag}"
            self.processed_dirpath = os.path.join(self.root, processed_dirname_extension, self.fold)
            self.stats_filepath = os.path.join(self.processed_dirpath, "stats-small.pt" if self.small else "stats.pt")
            self.processed_flag_filepath = os.path.join(
                self.processed_dirpath,
                "processed_flag-small" if self.small else "processed_flag",
            )

            if os.path.exists(self.processed_flag_filepath):
                self.stats = torch.load(self.stats_filepath)
            else:
                preprocess_tile_filter = None
                if self.process_only_filtered_tiles:
                    preprocess_tile_filter = self.preprocess_tile_filter if self.preprocess_tile_filter is not None else self.tile_filter
                tile_info_list = self.get_tile_info_list(tile_filter=preprocess_tile_filter)
                self.stats = self.process(tile_info_list)
                torch.save(self.stats, self.stats_filepath)
                pathlib.Path(self.processed_flag_filepath).touch()

            tile_info_list = self.get_tile_info_list(tile_filter=self.tile_filter)
            self.processed_relative_paths = self.get_processed_relative_paths(tile_info_list)
        else:
            self.tile_info_list = self.get_tile_info_list(tile_filter=self.tile_filter)

    def load_tile_metadata(self):
        if self.split_csv_path is None:
            raise ValueError("split_csv_path is required for DeventerAerial.")

        split_csv_path = self.split_csv_path if os.path.isabs(self.split_csv_path) else os.path.join(self.root, self.split_csv_path)
        if not os.path.exists(split_csv_path):
            raise FileNotFoundError(f"split_csv not found: {split_csv_path}")

        tiles = []
        with open(split_csv_path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required_fields = {"image_name", "split"}
            missing_fields = required_fields - set(reader.fieldnames or [])
            if missing_fields:
                raise ValueError(f"split_csv is missing required columns {sorted(missing_fields)}: {split_csv_path}")
            for fallback_id, row in enumerate(reader):
                file_name = row["image_name"].strip()
                name_stem = os.path.splitext(file_name)[0]
                image_id_value = row.get("image_id", "").strip()
                image_id = int(image_id_value) if image_id_value else fallback_id
                image_path = os.path.join(self.root, self.raw_dirname, self.fold, IMAGE_DIRNAME, file_name)
                image = skimage.io.imread(image_path)
                height, width = image.shape[:2]
                tiles.append(
                    {
                        "fold": self.fold,
                        "source_split": row.get("source_split", row["split"]).strip(),
                        "image_id": image_id,
                        "file_name": file_name,
                        "name_stem": name_stem,
                        "width": int(width),
                        "height": int(height),
                    }
                )
        if not tiles:
            raise ValueError(f"No tiles found in {split_csv_path}")
        return tiles

    def get_tile_info_list(self, tile_filter=None):
        tile_info_list = []
        for tile in self.tile_metadata:
            if tile["fold"] != self.fold:
                continue
            tile_info = {
                "file_name": str(tile["file_name"]),
                "name_stem": str(tile["name_stem"]),
                "image_id": int(tile["image_id"]),
                "width": int(tile["width"]),
                "height": int(tile["height"]),
                "source_split": str(tile.get("source_split", self.fold)),
            }
            tile_info_list.append(tile_info)
        if self.small:
            tile_info_list = tile_info_list[: min(len(tile_info_list), 8)]
        if tile_filter is not None:
            tile_info_list = list(filter(tile_filter, tile_info_list))
        return sorted(tile_info_list, key=lambda item: item["file_name"])

    def get_processed_relative_paths(self, tile_info_list):
        processed_relative_paths = []
        for tile_info in tile_info_list:
            processed_tile_relative_dirpath = tile_info["name_stem"]
            processed_tile_dirpath = os.path.join(self.processed_dirpath, processed_tile_relative_dirpath)
            sample_filenames = fnmatch.filter(os.listdir(processed_tile_dirpath), "data.*.pt")
            processed_tile_relative_paths = [
                os.path.join(processed_tile_relative_dirpath, sample_filename) for sample_filename in sample_filenames
            ]
            processed_relative_paths.extend(processed_tile_relative_paths)
        return sorted(processed_relative_paths)

    def process(self, tile_info_list):
        with multiprocess.Pool(self.pool_size) as pool:
            stats_all = list(tqdm(pool.imap(self._process_one, tile_info_list), total=len(tile_info_list), desc="Process"))

        stats = {}
        if not self.mask_only:
            stats_all = list(filter(None.__ne__, stats_all))
            stat_lists = {}
            for stats_one in stats_all:
                for key, stat in stats_one.items():
                    stat_lists.setdefault(key, []).append(stat)

            if "class_freq" in stat_lists and "num" in stat_lists:
                class_freq_array = np.stack(stat_lists["class_freq"], axis=0)
                num_array = np.stack(stat_lists["num"], axis=0)
                if num_array.min() == 0:
                    raise ZeroDivisionError("num_array has some zeros values, cannot divide!")
                stats["class_freq"] = np.sum(class_freq_array * num_array[:, None], axis=0) / np.sum(num_array)

        return stats

    def load_raw_data(self, tile_info):
        raw_data = {}
        image_filepath = os.path.join(self.root, self.raw_dirname, self.fold, IMAGE_DIRNAME, tile_info["file_name"])
        raw_data["image_filepath"] = image_filepath
        raw_data["image"] = skimage.io.imread(image_filepath)
        assert len(raw_data["image"].shape) == 3 and raw_data["image"].shape[2] == 3, \
            f"image should have shape (H, W, 3), not {raw_data['image'].shape}"
        image_float = raw_data["image"].astype(np.float32) / 255.0
        raw_data["image_mean"] = np.mean(image_float, axis=(0, 1)).astype(np.float32)
        raw_data["image_std"] = np.std(image_float, axis=(0, 1)).astype(np.float32)
        raw_data["image_std"] = np.maximum(raw_data["image_std"], 1e-6)
        raw_data["image_id"] = int(tile_info["image_id"])

        if self.gt_source == "disk":
            gt_base_filepath = os.path.join(self.root, self.raw_dirname, self.fold, self.gt_dirname, tile_info["name_stem"])
            gt_filepath = gt_base_filepath + "." + self.gt_type
            if not os.path.exists(gt_filepath):
                raw_data["gt_polygons"] = []
                return raw_data
            if self.gt_type == "npy":
                np_gt_polygons = np.load(gt_filepath, allow_pickle=True)
                gt_polygons = []
                for np_gt_polygon in np_gt_polygons:
                    try:
                        gt_polygons.append(shapely.geometry.Polygon(np_gt_polygon[:, ::-1]))
                    except ValueError:
                        continue
                raw_data["gt_polygons"] = gt_polygons
            elif self.gt_type == "geojson":
                geojson = python_utils.load_json(gt_filepath)
                gt_geometry = shapely.geometry.shape(geojson)
                if isinstance(gt_geometry, shapely.geometry.Polygon):
                    raw_data["gt_polygons"] = [gt_geometry]
                elif hasattr(gt_geometry, "geoms"):
                    raw_data["gt_polygons"] = [
                        geom for geom in gt_geometry.geoms if isinstance(geom, shapely.geometry.Polygon)
                    ]
                else:
                    raw_data["gt_polygons"] = []
            elif self.gt_type == "tif":
                raw_data["gt_polygons_image"] = skimage.io.imread(gt_filepath)[:, :, None]
                assert len(raw_data["gt_polygons_image"].shape) == 3 and raw_data["gt_polygons_image"].shape[2] == 1, \
                    f"Mask should have shape (H, W, 1), not {raw_data['gt_polygons_image'].shape}"
        elif self.gt_source == "osm":
            raise NotImplementedError("Downloading from OSM is not implemented.")

        return raw_data

    def _process_one(self, tile_info):
        process_id = int(multiprocess.current_process().name[-1])
        tile_name = tile_info["name_stem"]
        processed_tile_relative_dirpath = tile_name
        processed_tile_dirpath = os.path.join(self.processed_dirpath, processed_tile_relative_dirpath)
        processed_flag_filepath = os.path.join(processed_tile_dirpath, "processed_flag")
        stats_filepath = os.path.join(processed_tile_dirpath, "stats.pt")
        os.makedirs(processed_tile_dirpath, exist_ok=True)
        stats = {}

        if os.path.exists(processed_flag_filepath):
            if not self.mask_only:
                stats = torch.load(stats_filepath)
            return stats

        raw_data = self.load_raw_data(tile_info)

        if self.patch_size is None:
            raise NotImplementedError("patch_size=None is not implemented")

        patch_stride = self.patch_stride if self.patch_stride is not None else self.patch_size
        patch_boundingboxes = image_utils.compute_patch_boundingboxes(
            raw_data["image"].shape[0:2],
            stride=patch_stride,
            patch_res=self.patch_size,
        )
        class_freq_list = []
        for index, bbox in enumerate(
            tqdm(patch_boundingboxes, desc=f"Patching {tile_name}", leave=False, position=process_id)
        ):
            sample = {
                "image_filepath": raw_data["image_filepath"],
                "name": f"{tile_name}.rowmin_{bbox[0]}_colmin_{bbox[1]}_rowmax_{bbox[2]}_colmax_{bbox[3]}",
                "bbox": bbox,
                "image_id": raw_data["image_id"],
                "source_split": tile_info.get("source_split", self.fold),
                "image_mean": raw_data["image_mean"],
                "image_std": raw_data["image_std"],
            }

            if self.gt_type in {"npy", "geojson"}:
                patch_gt_polygons = polygon_utils.patch_polygons(
                    raw_data["gt_polygons"],
                    minx=bbox[1],
                    miny=bbox[0],
                    maxx=bbox[3],
                    maxy=bbox[2],
                )
                sample["gt_polygons"] = patch_gt_polygons
            elif self.gt_type == "tif":
                patch_gt_mask = raw_data["gt_polygons_image"][bbox[0]:bbox[2], bbox[1]:bbox[3], :]
                sample["gt_polygons_image"] = patch_gt_mask

            sample["image"] = raw_data["image"][bbox[0]:bbox[2], bbox[1]:bbox[3], :]

            sample = self.pre_transform(sample)
            if self.mask_only:
                del sample["image"]

            relative_filepath = os.path.join(processed_tile_relative_dirpath, f"data.{index:06d}.pt")
            filepath = os.path.join(self.processed_dirpath, relative_filepath)
            torch.save(sample, filepath)

            if not self.mask_only:
                if self.gt_type in {"npy", "geojson"}:
                    class_freq_list.append(np.mean(sample["gt_polygons_image"], axis=(0, 1)) / 255)
                else:
                    raise NotImplementedError(f"gt_type={self.gt_type} not implemented for computing stats")

        if not self.mask_only:
            if class_freq_list:
                class_freq_array = np.stack(class_freq_list, axis=0)
                stats["class_freq"] = np.mean(class_freq_array, axis=0)
                stats["num"] = len(class_freq_list)
            else:
                print_utils.print_warning(f"Empty tile: {tile_name}")
            torch.save(stats, stats_filepath)

        pathlib.Path(processed_flag_filepath).touch()
        return stats

    def __len__(self):
        if self.pre_process:
            return len(self.processed_relative_paths)
        return len(self.tile_info_list)

    def __getitem__(self, idx):
        if self.pre_process:
            filepath = os.path.join(self.processed_dirpath, self.processed_relative_paths[idx])
            data = torch.load(filepath)
            if self.mask_only:
                data["image"] = np.repeat(data["gt_polygons_image"][:, :, 0:1], 3, axis=-1)
                data["image_mean"] = DEFAULT_IMAGE_MEAN
                data["image_std"] = np.array([1.0, 1.0, 1.0], dtype=np.float32)
            else:
                data["image_mean"] = np.asarray(data.get("image_mean", DEFAULT_IMAGE_MEAN), dtype=np.float32)
                data["image_std"] = np.asarray(data.get("image_std", DEFAULT_IMAGE_STD), dtype=np.float32)
                data["class_freq"] = self.stats["class_freq"]
        else:
            tile_info = self.tile_info_list[idx]
            data = self.load_raw_data(tile_info)
            data["name"] = tile_info["name_stem"]
            data["image_mean"] = np.asarray(data.get("image_mean", DEFAULT_IMAGE_MEAN), dtype=np.float32)
            data["image_std"] = np.asarray(data.get("image_std", DEFAULT_IMAGE_STD), dtype=np.float32)
            data["image_id"] = int(tile_info["image_id"])
        data = self.transform(data)
        return data
