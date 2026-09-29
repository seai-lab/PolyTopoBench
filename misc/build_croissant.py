#!/usr/bin/env python3
"""Write croissant.json for the PolyTopoBench HF release built by build_hf_release.py.

Responsible-AI fields are carried over from the per-dataset Croissant files in metadata/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_hf_release import sha256

REPO_ID = "PingL/PolyTopoBench"
HF_URL = f"https://huggingface.co/datasets/{REPO_ID}"
RESOLVE = f"{HF_URL}/resolve/main"
AUTHORS = [
    "Zeping Liu", "Ni Lao", "Weiwei Sun", "Gil Wolff",
    "Yiqun Xie", "Liang Zhao", "Junfeng Jiao", "Gengchen Mai",
]
CITATION = (
    "@misc{liu2026polytopobenchbenchmarkcomplexvector, title={PolyTopoBench: A Benchmark for Complex Vector "
    "Polygon Generation from Remote Sensing Imagery}, author={Zeping Liu and Ni Lao and Weiwei Sun and Gil Wolff "
    "and Yiqun Xie and Liang Zhao and Junfeng Jiao and Gengchen Mai}, year={2026}, eprint={2609.32856}, "
    "archivePrefix={arXiv}, primaryClass={cs.CV}, url={https://arxiv.org/abs/2609.32856}}"
)
RAI_KEYS = (
    "rai:dataLimitations", "rai:dataBiases", "rai:personalSensitiveInformation",
    "rai:dataUseCases", "rai:dataSocialImpact",
)
ANNOTATION_COLUMNS = [
    ("annotation_id", "sc:Integer", "Annotation id, unique within the task split; matches the COCO JSON."),
    ("image_id", "sc:Integer", "Image id, contiguous from 1 within the split; matches the COCO JSON and splits.csv."),
    ("file_name", "sc:Text", "Image path relative to <dataset>/images/<split>/."),
    ("category", "sc:Text", "Task category name."),
    ("area", "sc:Float", "Polygon area in square pixels (exterior minus holes)."),
    ("hole_count", "sc:Integer", "Number of interior rings."),
    ("geometry", "sc:Text", "WKB-encoded Polygon in 512x512 pixel coordinates (origin top-left, y down, no CRS); "
                            "ring 0 is the exterior and further rings are holes."),
]
SPLIT_COLUMNS = {
    "inria": [
        ("file_name", "sc:Text", "Patch path relative to inria/images/<split>/."),
        ("image_id", "sc:Integer", "Image id within the split."),
        ("split", "sc:Text", "train or val."),
        ("city", "sc:Text", "Inria city of the source tile."),
        ("source_tile", "sc:Text", "Source 5000x5000 Inria tile."),
        ("x0", "sc:Integer", "Patch x offset in the source tile, in pixels."),
        ("y0", "sc:Integer", "Patch y offset in the source tile, in pixels."),
    ],
    "deventer": [
        ("file_name", "sc:Text", "Patch file name in deventer/images/<split>/."),
        ("image_id", "sc:Integer", "Image id within the split, shared by the three Deventer tasks."),
        ("split", "sc:Text", "train or val (val merges the upstream val and test splits)."),
        ("raw_image_id", "sc:Integer", "Image id in the upstream Deventer-512 annotations."),
    ],
}


def file_object(oid: str, path: str, fmt: str, sha: str, description: str) -> dict:
    return {
        "@type": "cr:FileObject", "@id": oid, "name": oid, "description": description,
        "contentUrl": f"{RESOLVE}/{path}", "encodingFormat": fmt, "sha256": sha,
    }


def file_set(oid: str, includes: str, fmt: str, description: str) -> dict:
    return {
        "@type": "cr:FileSet", "@id": oid, "name": oid, "description": description,
        "containedIn": {"@id": "hf-repository"}, "includes": includes, "encodingFormat": fmt,
    }


def column_fields(rs: str, source_id: str, columns: list, source_key: str = "fileObject") -> list:
    return [
        {
            "@type": "cr:Field", "@id": f"{rs}/{name}", "name": name, "description": desc, "dataType": dtype,
            "source": {source_key: {"@id": source_id}, "extract": {"column": name}},
        }
        for name, dtype, desc in columns
    ]


def build(release: Path, metadata_dir: Path, date: str) -> dict:
    tasks = json.loads((release / "tasks.json").read_text())["tasks"]
    inria_old = json.loads((metadata_dir / "croissant_rai_PolyTopoBench_Inria_Aligned_Building_Polygons.json").read_text())
    deventer_old = json.loads((metadata_dir / "croissant_rai_PolyTopoBench_Deventer-512_valtest-as-val.json").read_text())

    distribution = [{
        "@type": "cr:FileObject", "@id": "hf-repository", "name": "hf-repository",
        "description": "The PolyTopoBench Hugging Face dataset repository.",
        "contentUrl": f"{HF_URL}/tree/main", "encodingFormat": "git+https", "sha256": "main",
    }]
    record_sets = []

    distribution.append(file_object("tasks-json", "tasks.json", "application/json", sha256(release / "tasks.json"),
                                    "Task registry with file paths, checksums and statistics."))
    for ds in ("inria", "deventer"):
        oid = f"{ds}-splits-csv"
        distribution.append(file_object(oid, f"{ds}/splits.csv", "text/csv", sha256(release / ds / "splits.csv"),
                                        f"One row per {ds} patch with its split and provenance."))
        record_sets.append({
            "@type": "cr:RecordSet", "@id": f"{ds}-patches", "name": f"{ds}-patches",
            "description": f"One record per {ds} image patch.", "key": {"@id": f"{ds}-patches/file_name"},
            "field": column_fields(f"{ds}-patches", oid, SPLIT_COLUMNS[ds]),
        })

    distribution += [
        file_set("inria-images", "inria/images/*/*/*.tif", "image/tiff", "Inria 512x512 RGB patches, grouped by split and city."),
        file_set("deventer-images", "deventer/images/*/*.png", "image/png", "Deventer 512x512 RGB patches, shared by all Deventer tasks."),
        file_set("raw-inria-tiles", "raw/inria/train/images/*.tif", "image/tiff", "180 source Inria tiles, 5000x5000."),
        file_set("raw-inria-masks", "raw/inria/train/gt/*.tif", "image/tiff", "Official Inria binary building masks (not the PolyTopoBench ground truth)."),
        file_set("raw-inria-geojson", "raw/inria/raw/train/gt_polygonized/*.geojson", "application/geo+json",
                 "Full-tile vector ground truth: OSM-aligned, manually corrected building polygons with interior rings."),
        file_set("raw-deventer-masks", "raw/deventer/*/masks/*.png", "image/png", "Upstream Deventer-512 multi-class masks (values 0-4)."),
        file_set("raw-deventer-annotations", "raw/deventer/*/annotations/*.json", "application/json",
                 "Upstream Deventer-512 per-class COCO annotations (building, road, vegetation, unvegetated, water)."),
    ]
    for ds, glob in (("inria", "inria/images/*/*/*.tif"), ("deventer", "deventer/images/*/*.png")):
        rs = f"{ds}-image-files"
        record_sets.append({
            "@type": "cr:RecordSet", "@id": rs, "name": rs, "description": f"One record per {ds} image patch file.",
            "field": [
                {"@type": "cr:Field", "@id": f"{rs}/path", "name": "path", "description": "Path of the image in the repository.",
                 "dataType": "sc:Text", "source": {"fileSet": {"@id": f"{ds}-images"}, "extract": {"fileProperty": "fullpath"}}},
                {"@type": "cr:Field", "@id": f"{rs}/image", "name": "image", "description": "RGB image patch.",
                 "dataType": "sc:ImageObject", "source": {"fileSet": {"@id": f"{ds}-images"}, "extract": {"fileProperty": "content"}}},
            ],
        })

    for task_name, task in tasks.items():
        slug = task_name.replace("/", "-")
        for split, info in task["splits"].items():
            distribution.append(file_object(f"{slug}-{split}-json", info["annotations"], "application/json", info["annotations_sha256"],
                                            f"{task_name} {split} annotations, COCO layout; segmentation is [exterior, hole_1, ...]."))
            oid = f"{slug}-{split}-parquet"
            distribution.append(file_object(oid, info["geoparquet"], "application/x-parquet", info["geoparquet_sha256"],
                                            f"{task_name} {split} annotations as GeoParquet, one row per polygon."))
            rs = f"{slug}-{split}"
            record_sets.append({
                "@type": "cr:RecordSet", "@id": rs, "name": rs,
                "description": f"{task_name} {split}: one record per polygon ({info['instances']} polygons, "
                               f"{info['holes']} interior rings, {info['images']} images).",
                "key": {"@id": f"{rs}/annotation_id"},
                "field": column_fields(rs, oid, ANNOTATION_COLUMNS),
            })

    rai = {k: f"Inria building: {inria_old[k]} Deventer land cover: {deventer_old[k]}" for k in RAI_KEYS}
    return {
        "@context": inria_old["@context"],
        "@type": "sc:Dataset",
        "name": "PolyTopoBench",
        "description": (
            "PolyTopoBench is a benchmark for complex vector polygon generation from remote-sensing imagery, "
            "evaluating whether models recover complete polygon topology including interior rings. It has four "
            "single-class tasks on 512x512 aerial image patches: Inria building (OpenStreetMap footprints aligned "
            "to the Inria Aerial Image Labeling tiles and manually corrected) and Deventer-512 road, vegetation and "
            "unvegetated land cover. Annotations are provided as COCO-layout JSON, whose segmentation lists the "
            "exterior ring followed by hole rings, and as GeoParquet."
        ),
        "conformsTo": "http://mlcommons.org/croissant/1.1",
        "url": HF_URL,
        "sameAs": ["https://github.com/seai-lab/PolyTopoBench", "https://arxiv.org/abs/2609.32856"],
        "license": [
            "https://spdx.org/licenses/ODbL-1.0.html",
            "https://spdx.org/licenses/CC-BY-4.0.html",
        ],
        "creator": [{"@type": "sc:Person", "name": name} for name in AUTHORS],
        "citeAs": CITATION,
        "version": "1.0.0",
        "datePublished": date,
        "keywords": sorted(set(inria_old["keywords"]) | set(deventer_old["keywords"]) | {"interior rings", "polygon topology", "GeoParquet"}),
        "distribution": distribution,
        "recordSet": record_sets,
        **rai,
        "rai:hasSyntheticData": False,
        "prov:wasDerivedFrom": [
            {"@id": "https://project.inria.fr/aerialimagelabeling/", "prov:label": "Inria Aerial Image Labeling Dataset",
             "sc:license": "Public domain imagery and building footprints (per the dataset page)"},
            {"@id": "https://www.openstreetmap.org", "prov:label": "OpenStreetMap", "sc:license": "ODbL-1.0"},
            {"@id": "https://huggingface.co/datasets/HeinzJiao/Deventer-512", "prov:label": "Deventer-512", "sc:license": "CC-BY-4.0"},
        ],
        "prov:wasGeneratedBy": inria_old["prov:wasGeneratedBy"] + deventer_old["prov:wasGeneratedBy"] + [{
            "@type": "prov:Activity",
            "prov:label": "Patch extraction and packaging",
            "sc:description": (
                "Inria tiles were split by city (29 train / 7 val tiles per city) and cut into a 10x10 grid of "
                "512x512 patches; polygons were clipped to each patch and clipped parts with a bounding-box side of "
                "5 px or less were removed; training patches without buildings were dropped. Deventer-512 val and "
                "test splits were merged into val and image ids were renumbered. Each task split was exported as "
                "COCO-layout JSON and GeoParquet."
            ),
        }],
    }


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, default=repo / "dataset" / "hf_release_v1")
    parser.add_argument("--metadata-dir", type=Path, default=repo / "metadata")
    parser.add_argument("--date", default="2026-09-26")
    args = parser.parse_args()
    out = args.release / "croissant.json"
    out.write_text(json.dumps(build(args.release, args.metadata_dir, args.date), indent=2, ensure_ascii=False) + "\n")
    print(out)


if __name__ == "__main__":
    main()
