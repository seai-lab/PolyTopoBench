#!/usr/bin/env python3
"""Small PolyTopoBench glue steps around the unmodified HoliTracer tools.

Subcommands
-----------
work-dataset   Build <out>/{train,val} views of the HoliTracer mirror (symlinks, or a
               smoke subset listed in coco_label_with_inter_smoke.json) plus
               <out>/test -> val, which HoliTracer's vector trainer expects next to val.
               Keeps every generated file (predict/..., test/) out of data_processed.
clean          Remove stale seg masks (HoliTracer's seg_infer silently skips images whose
               mask already exists, so a retrained model would reuse old predictions).
select-weights Pick the vector-stage weights for inference: best_model.pth, otherwise the
               model_state_dict of last_model.pth (best is only written after
               eval_start_epoch, i.e. never in 1-epoch smoke runs).
gt-seeds       Smoke only: use the GT polygons as the vector-stage *training* seeds. A
               1-epoch seg model predicts near-full-image blobs, which match no GT
               building, so make_vector_h5 would write 0 samples and vector_train fails.
to-hisup       Convert HoliTracer's refined COCO json (segmentation = [exterior, hole1,
               ...], patch coords) into hisup-format predictions and check that every
               image id maps to the same patch file in the canonical GT.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def _link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.symlink_to(src.resolve())


def build_work_dataset(data_root: Path, out: Path, smoke: bool, smoke_label: str) -> None:
    for split in ("train", "val"):
        src = data_root / split
        dst = out / split
        dst.mkdir(parents=True, exist_ok=True)
        if not smoke:
            for name in ("img", "img_tif", "mask"):
                if (src / name).is_dir():
                    _link(src / name, dst / name)
            _link(src / "coco_label_with_inter.json", dst / "coco_label_with_inter.json")
            continue
        label = json.loads((src / smoke_label).read_text(encoding="utf-8"))
        for name in ("img", "img_tif", "mask"):
            if (dst / name).is_symlink():
                (dst / name).unlink()
            elif (dst / name).is_dir():
                shutil.rmtree(dst / name)
        for image in label["images"]:
            stem = Path(image["file_name"]).stem
            for name, ext in (("img", Path(image["file_name"]).suffix), ("img_tif", ".tif"), ("mask", ".png")):
                path = src / name / f"{stem}{ext}"
                if path.is_file():
                    _link(path, dst / name / path.name)
        (dst / "coco_label_with_inter.json").unlink(missing_ok=True)
        (dst / "coco_label_with_inter.json").write_text(json.dumps(label), encoding="utf-8")
    test = out / "test"
    if not test.is_symlink():
        if test.exists():
            shutil.rmtree(test)
        test.symlink_to("val")


def clean(paths: list[Path]) -> None:
    for path in paths:
        if path.is_dir():
            shutil.rmtree(path)
            print(f"[holitracer] removed stale {path}")


def select_weights(run_dir: Path, out: Path, weights: Path | None = None) -> None:
    import torch

    best = run_dir / "best_model.pth"
    last = run_dir / "last_model.pth"
    if weights is not None:
        ckpt = torch.load(weights, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        source = weights
    elif best.is_file():
        state, source = torch.load(best, map_location="cpu"), best
    elif last.is_file():
        ckpt = torch.load(last, map_location="cpu")
        state, source = ckpt.get("model_state_dict", ckpt), last
    else:
        raise FileNotFoundError(f"No best_model.pth or last_model.pth under {run_dir}")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, out)
    print(json.dumps({"vector_weights_source": str(source), "written": str(out)}))


def gt_seeds(labels: Path, out: Path) -> None:
    payload = json.loads(labels.read_text(encoding="utf-8"))
    for ann in payload["annotations"]:
        ann.setdefault("score", 1.0)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload), encoding="utf-8")
    print(f"[holitracer] smoke: wrote {len(payload['annotations'])} GT seeds to {out}")


def to_hisup(input_path: Path, labels_path: Path, gt_path: Path, output: Path) -> None:
    result = json.loads(input_path.read_text(encoding="utf-8"))
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    gt_stem = {int(img["id"]): Path(img["file_name"]).stem for img in gt["images"]}
    label_stem = {int(img["id"]): Path(img["file_name"]).stem for img in labels["images"]}
    mismatched = [i for i, stem in label_stem.items() if gt_stem.get(i) != stem]
    if mismatched:
        raise RuntimeError(f"{len(mismatched)} HoliTracer image ids do not match the canonical GT file names (e.g. {mismatched[:5]})")

    records = []
    for index, ann in enumerate(result.get("annotations", [])):
        seg = [
            [float(v) for v in ring]
            for ring in ann.get("segmentation", [])
            if isinstance(ring, list) and len(ring) >= 6 and len(ring) % 2 == 0
        ]
        if not seg:
            continue
        xs = [v for ring in seg for v in ring[0::2]]
        ys = [v for ring in seg for v in ring[1::2]]
        records.append({
            "image_id": int(ann["image_id"]),
            "category_id": int(ann.get("category_id", 1)),
            "score": float(ann.get("score", 0.0)),
            "segmentation": seg,
            "bbox": [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)],
            "id": int(ann.get("id", index + 1)),
        })
    outside = {r["image_id"] for r in records} - set(gt_stem)
    if outside:
        raise RuntimeError(f"{len(outside)} predicted image ids are not in {gt_path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records), encoding="utf-8")
    print(json.dumps({
        "input": str(input_path),
        "output": str(output),
        "num_predictions": len(records),
        "images_with_predictions": len({r["image_id"] for r in records}),
        "gt_images": len(gt_stem),
        "holes": sum(len(r["segmentation"]) - 1 for r in records),
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("work-dataset")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--smoke", type=int, default=0)
    p.add_argument("--smoke-label", default="coco_label_with_inter_smoke.json")
    p = sub.add_parser("clean")
    p.add_argument("paths", type=Path, nargs="+")
    p = sub.add_parser("select-weights")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--weights", type=Path, default=None, help="Explicit vector checkpoint (VECTOR_CHECKPOINT).")
    p = sub.add_parser("gt-seeds")
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("to-hisup")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True, help="coco_label_with_inter.json used for inference (file names per id).")
    p.add_argument("--gt", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.cmd == "work-dataset":
        build_work_dataset(args.data_root, args.out, bool(args.smoke), args.smoke_label)
    elif args.cmd == "clean":
        clean(args.paths)
    elif args.cmd == "gt-seeds":
        gt_seeds(args.labels, args.out)
    elif args.cmd == "select-weights":
        select_weights(args.run_dir, args.out, args.weights)
    else:
        to_hisup(args.input, args.labels, args.gt, args.output)


if __name__ == "__main__":
    main()
