#!/usr/bin/env python3
"""Build the public PolyTopoBench v1.0 data release from the canonical hisup mirrors.

Output layout (images are hardlinked, never copied):

    <out>/
    ├── tasks.json
    ├── inria/
    │   ├── images/{train,val}/<city>/*.tif
    │   ├── splits.csv
    │   └── building/{train,val}.{json,parquet}
    ├── deventer/
    │   ├── images/{train,val}/*.png
    │   ├── splits.csv
    │   └── {road,vegetation,unvegetated}/{train,val}.{json,parquet}
    └── raw/{inria,deventer}/...

Annotation JSON files keep the canonical content (ids, categories, extra fields) and only
change Inria ``file_name`` to ``<city>/<patch>.tif`` so it resolves against
``images/<split>/``. Each ``segmentation`` is a list of rings: ring 0 is the exterior and
every further ring is a hole of that exterior (not a COCO multi-part union).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from shapely.geometry import Polygon

VERSION = "1.0"
SPLITS = ("train", "val")
DEVENTER_CATEGORIES = ("road", "vegetation", "unvegetated")
SEGMENTATION_FORMAT = (
    "segmentation is a list of flat rings [x1, y1, x2, y2, ...] in 512x512 pixel "
    "coordinates (origin top-left, y down); ring 0 is the exterior and every further "
    "ring is a hole of that exterior. This differs from COCO multi-part polygons, "
    "which pycocotools would union."
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.link(src, dst)


def link_tree(src: Path, dst: Path) -> int:
    n = 0
    for p in sorted(src.rglob("*")):
        if p.is_file():
            link(p, dst / p.relative_to(src))
            n += 1
    return n


def inria_city(source_image_name: str) -> str:
    return re.sub(r"\d+$", "", Path(source_image_name).stem)


def ring_pairs(flat: list[float]) -> list[tuple[float, float]]:
    return list(zip(flat[0::2], flat[1::2]))


def write_geoparquet(coco: dict, category: str, out: Path) -> dict:
    file_names = {img["id"]: img["file_name"] for img in coco["images"]}
    cols = {k: [] for k in ("annotation_id", "image_id", "file_name", "category", "area", "hole_count", "geometry")}
    invalid = 0
    for ann in coco["annotations"]:
        rings = [ring_pairs(r) for r in ann["segmentation"]]
        poly = Polygon(rings[0], rings[1:])
        invalid += not poly.is_valid
        cols["annotation_id"].append(ann["id"])
        cols["image_id"].append(ann["image_id"])
        cols["file_name"].append(file_names[ann["image_id"]])
        cols["category"].append(category)
        cols["area"].append(float(ann["area"]))
        cols["hole_count"].append(len(rings) - 1)
        cols["geometry"].append(poly.wkb)
    table = pa.table(
        {
            "annotation_id": pa.array(cols["annotation_id"], pa.int64()),
            "image_id": pa.array(cols["image_id"], pa.int64()),
            "file_name": pa.array(cols["file_name"], pa.string()),
            "category": pa.array(cols["category"], pa.string()),
            "area": pa.array(cols["area"], pa.float64()),
            "hole_count": pa.array(cols["hole_count"], pa.int32()),
            "geometry": pa.array(cols["geometry"], pa.binary()),
        }
    )
    geo = {
        "version": "1.1.0",
        "primary_column": "geometry",
        "columns": {"geometry": {"encoding": "WKB", "geometry_types": ["Polygon"], "crs": None}},
    }
    table = table.replace_schema_metadata({b"geo": json.dumps(geo).encode()})
    pq.write_table(table, out, compression="zstd")
    return {"invalid_geometries": invalid}


def stats(coco: dict) -> dict:
    holes = [len(a["segmentation"]) - 1 for a in coco["annotations"]]
    return {
        "images": len(coco["images"]),
        "instances": len(coco["annotations"]),
        "instances_with_holes": sum(h > 0 for h in holes),
        "holes": sum(holes),
    }


def write_task_split(coco: dict, category: str, task_dir: Path, split: str, root: Path) -> dict:
    coco["info"] = {**coco.get("info", {}), "polytopobench_version": VERSION, "segmentation_format": SEGMENTATION_FORMAT}
    json_path = task_dir / f"{split}.json"
    parquet_path = task_dir / f"{split}.parquet"
    task_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(coco))
    extra = write_geoparquet(coco, category, parquet_path)
    return {
        "annotations": str(json_path.relative_to(root)),
        "annotations_sha256": sha256(json_path),
        "geoparquet": str(parquet_path.relative_to(root)),
        "geoparquet_sha256": sha256(parquet_path),
        **stats(coco),
        **extra,
    }


def build_inria(src_root: Path, raw_root: Path, out: Path) -> dict:
    ds = out / "inria"
    task = {"dataset": "inria", "category": "building", "image_format": "tif", "splits": {}}
    rows = []
    for split in SPLITS:
        src = src_root / "inria_building" / "hisup" / split
        coco = json.loads((src / "annotation.json").read_text())
        for img in coco["images"]:
            city = inria_city(img["source_image_name"])
            rel = f"{city}/{img['file_name']}"
            link(src / "images" / img["file_name"], ds / "images" / split / rel)
            x0, y0 = img["patch_origin"]
            rows.append([rel, img["id"], split, city, img["source_image_name"], x0, y0])
            img["file_name"] = rel
        info = write_task_split(coco, "building", ds / "building", split, out)
        task["splits"][split] = {"image_dir": f"inria/images/{split}", **info}
    with (ds / "splits.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file_name", "image_id", "split", "city", "source_tile", "x0", "y0"])
        w.writerows(rows)
    task["raw_files_linked"] = link_tree(raw_root / "inria_dataset_aligned", out / "raw" / "inria")
    return {"inria/building": task}


def build_deventer(src_root: Path, raw_root: Path, out: Path) -> dict:
    ds = out / "deventer"
    tasks = {}
    rows = []
    for split in SPLITS:
        raw_split = raw_root / "deventer_512_valtest_as_val" / split
        raw_ids = {img["file_name"]: img["id"] for img in json.loads((raw_split / "annotations" / "road.json").read_text())["images"]}
        images = None
        for category in DEVENTER_CATEGORIES:
            src = src_root / "deventer_512_valtest_as_val" / "hisup" / category / split
            coco = json.loads((src / "annotation.json").read_text())
            cat_images = {img["id"]: img["file_name"] for img in coco["images"]}
            if images is None:
                images = cat_images
                for image_id, name in sorted(images.items()):
                    link(src / "images" / name, ds / "images" / split / name)
                    rows.append([name, image_id, split, raw_ids[name]])
            elif cat_images != images:
                raise SystemExit(f"deventer {split}/{category}: image ids differ from {DEVENTER_CATEGORIES[0]}")
            else:
                for name in images.values():
                    if (src / "images" / name).stat().st_ino != (ds / "images" / split / name).stat().st_ino:
                        raise SystemExit(f"deventer {split}/{category}/{name}: image differs across categories")
            info = write_task_split(coco, category, ds / category, split, out)
            task = tasks.setdefault(f"deventer/{category}", {"dataset": "deventer", "category": category, "image_format": "png", "splits": {}})
            task["splits"][split] = {"image_dir": f"deventer/images/{split}", **info}
    with (ds / "splits.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file_name", "image_id", "split", "raw_image_id"])
        w.writerows(rows)
    n_raw = link_tree(raw_root / "deventer_512_valtest_as_val", out / "raw" / "deventer")
    for task in tasks.values():
        task["raw_files_linked"] = n_raw
    return tasks


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--processed-root", type=Path, default=repo / "dataset" / "data_processed")
    parser.add_argument("--raw-root", type=Path, default=repo / "dataset")
    parser.add_argument("--out", type=Path, default=repo / "dataset" / "hf_release_v1")
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} already exists; remove it first")

    tasks = {}
    tasks.update(build_inria(args.processed_root, args.raw_root, args.out))
    tasks.update(build_deventer(args.processed_root, args.raw_root, args.out))
    manifest = {"polytopobench_version": VERSION, "segmentation_format": SEGMENTATION_FORMAT, "tasks": tasks}
    (args.out / "tasks.json").write_text(json.dumps(manifest, indent=2) + "\n")

    per_dir = Counter(str(p.parent.relative_to(args.out)) for p in args.out.rglob("*") if p.is_file())
    print(json.dumps({k: {s: {x: v[x] for x in ("images", "instances", "holes", "invalid_geometries")} for s, v in t["splits"].items()} for k, t in tasks.items()}, indent=1))
    print("max files in one folder:", max(per_dir.values()), max(per_dir, key=per_dir.get))


if __name__ == "__main__":
    main()
