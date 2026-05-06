#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
from pathlib import Path

from utilis.common import (
    DEVENTER_DATASET,
    INRIA_DATASET,
    RunContext,
    default_run_name,
    default_task,
    find_data_processed_root,
    parse_key_values,
    require_paths,
    run_steps,
)


METHODS = [
    "unet_poly",
    "maskrcnn_poly",
    "sam2_poly",
    "hisup",
    "acpvnet",
    "ffl",
    "gcp",
    "holitracer",
    "pix2poly",
    "polyworld",
    "roipoly",
]

ALIASES = {
    "unet": "unet_poly",
    "unet_seg": "unet_poly",
    "maskrcnn": "maskrcnn_poly",
    "maskrcnn_seg": "maskrcnn_poly",
    "mask_r_cnn": "maskrcnn_poly",
    "sam2": "sam2_poly",
    "sam2_seg": "sam2_poly",
    "acpv": "acpvnet",
}


def parse_methods(value: str) -> list[str]:
    raw = [part.strip() for part in value.split(",") if part.strip()]
    if raw == ["all"]:
        return METHODS.copy()
    methods = []
    for item in raw:
        method = ALIASES.get(item, item)
        if method not in METHODS:
            raise SystemExit(f"Unknown method '{item}'. Use --list to see valid names.")
        methods.append(method)
    return methods


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run PolyTopoBench baselines.")
    parser.add_argument("--list", action="store_true", help="List available baseline names and exit.")
    parser.add_argument("--method", default="unet_poly", help="Baseline name, comma list, or 'all'.")
    parser.add_argument("--dataset", choices=[INRIA_DATASET, DEVENTER_DATASET], default=INRIA_DATASET)
    parser.add_argument("--task", default=None, help="building for Inria; road/vegetation/unvegetated for Deventer.")
    parser.add_argument("--mode", choices=["train", "infer", "eval", "train_eval"], default="train_eval")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-val", action="store_true", help="Skip validation/inference step when a runner supports it.")
    parser.add_argument("--no-conda", action="store_true", help="Run commands without 'conda run -n ...'.")
    parser.add_argument("--env", default=None, help="Override conda environment for the selected method(s).")
    parser.add_argument("--bbox-json", type=Path, default=None, help="SAM2 bbox prior JSON from Mask R-CNN.")
    parser.add_argument("--data-processed-root", default=None, help="Path containing inria_building/ and deventer_512_valtest_as_val/.")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="Override runner hyperparameters.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.list:
        print("\n".join(METHODS))
        return

    release_root = Path(__file__).resolve().parent
    data_processed_root = find_data_processed_root(release_root, args.data_processed_root)
    output_root = args.output_root or (release_root / "output")
    if not output_root.is_absolute():
        output_root = release_root / output_root
    task = default_task(args.dataset, args.task)
    params = parse_key_values(args.set)

    if args.dataset == INRIA_DATASET and task != "building":
        raise SystemExit("Inria release data supports task=building.")

    require_paths([data_processed_root / args.dataset])

    for method in parse_methods(args.method):
        run_name = args.run_name or default_run_name(method, task, args.mode, args.smoke)
        ctx = RunContext(
            release_root=release_root,
            data_processed_root=data_processed_root,
            output_root=output_root,
            method=method,
            dataset=args.dataset,
            task=task,
            mode=args.mode,
            run_name=run_name,
            gpu=args.gpu,
            seed=args.seed,
            smoke=args.smoke,
            run_val=not args.no_val,
            no_conda=args.no_conda,
            env_name=args.env,
            bbox_json=args.bbox_json,
            params=params,
        )
        module = importlib.import_module(f"models.{method}.runner")
        if not args.dry_run and hasattr(module, "prepare"):
            module.prepare(ctx)
        steps = module.build_commands(ctx)
        run_steps(steps, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
