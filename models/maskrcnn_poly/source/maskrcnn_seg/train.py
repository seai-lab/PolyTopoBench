#!/usr/bin/env python3
"""Train torchvision Mask R-CNN on the maskrcnn_seg mirror.

Architecture: `maskrcnn_resnet50_fpn_v2(weights='DEFAULT')` with the
`box_predictor` / `mask_predictor` heads re-initialized for num_classes=2
(background + foreground).

Optimizer: SGD(lr=0.005, momentum=0.9, weight_decay=1e-4) — torchvision ref.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from maskrcnn_seg.dataset import MaskRcnnSegDataset, collate_train


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(num_classes: int = 2):
    from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    model = maskrcnn_resnet50_fpn_v2(weights="DEFAULT")
    in_feats = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feats, num_classes)
    in_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_mask, 256, num_classes)
    return model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mirror-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--warmup-iters", type=int, default=500)
    p.add_argument("--max-train-steps", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ckpt_dir = args.output_dir / "checkpoints"
    logs_dir = args.output_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-4)

    train_ds = MaskRcnnSegDataset(args.mirror_root, split="train", augment=True)
    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_train,
        persistent_workers=args.num_workers > 0,
    )

    iters_per_epoch = len(loader)
    warmup_iters = min(args.warmup_iters, iters_per_epoch - 1) if iters_per_epoch > 1 else 0
    warmup_factor = 1.0 / 1000
    def warmup_lr(step: int) -> float:
        if step >= warmup_iters:
            return 1.0
        alpha = float(step) / float(max(warmup_iters, 1))
        return warmup_factor * (1 - alpha) + alpha
    milestones = {int(args.max_epochs * 0.6), int(args.max_epochs * 0.85)}

    log_file = logs_dir / "train.log"
    t0 = time.time()
    global_step = 0

    for epoch in range(1, args.max_epochs + 1):
        if epoch in milestones:
            for pg in optimizer.param_groups:
                pg["lr"] *= 0.1
        model.train()
        epoch_loss = 0.0
        n_steps = 0
        ep_start = time.time()

        for step, (images, targets) in enumerate(loader):
            if args.max_train_steps and step >= args.max_train_steps:
                break
            images = [img.to(device, non_blocking=True) for img in images]
            targets = [
                {k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets
            ]
            # drop images with no targets for torchvision
            keep_indices = [i for i, t in enumerate(targets) if t["boxes"].shape[0] > 0]
            if not keep_indices:
                continue
            images = [images[i] for i in keep_indices]
            targets = [targets[i] for i in keep_indices]

            if epoch == 1 and warmup_iters > 0:
                factor = warmup_lr(global_step)
                for pg in optimizer.param_groups:
                    pg["lr"] = args.lr * factor

            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.detach().cpu())
            n_steps += 1
            global_step += 1
            if step % args.log_every == 0:
                msg = (
                    f"[epoch {epoch}/{args.max_epochs} step {step}/{iters_per_epoch}] "
                    f"loss={float(loss.detach().cpu()):.4f}"
                )
                print(msg, flush=True)
                with log_file.open("a") as f:
                    f.write(msg + "\n")

        avg_loss = epoch_loss / max(n_steps, 1)
        ep_time = time.time() - ep_start
        summary = {
            "epoch": epoch,
            "avg_train_loss": avg_loss,
            "epoch_time_sec": ep_time,
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed_sec": time.time() - t0,
        }
        with log_file.open("a") as f:
            f.write(json.dumps(summary) + "\n")
        print(json.dumps(summary), flush=True)

        last_path = ckpt_dir / "last.pt"
        torch.save({"model": model.state_dict(), "epoch": epoch}, last_path)

    # copy last -> best (no canonical val signal during training; we rely on the last epoch)
    import shutil
    shutil.copy2(ckpt_dir / "last.pt", ckpt_dir / "best.pt")

    print(f"[maskrcnn_seg train done] total_time={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
