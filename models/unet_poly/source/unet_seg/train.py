#!/usr/bin/env python3
"""Train a U-Net semantic segmentation model for the unet_seg baseline.

Architecture: `smp.Unet(efficientnet-b3, imagenet, in_channels=3, classes=2)`
Loss: DiceLoss(multiclass) + CrossEntropyLoss
Optimizer: AdamW lr=1e-4

Outputs:
    <output-dir>/checkpoints/best.pt         -- best val-mIoU weights
    <output-dir>/checkpoints/last.pt         -- final weights
    <output-dir>/logs/train.log              -- per-epoch loss/metric log
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
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from unet_seg.dataset import UnetSegImageMaskDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model():
    import segmentation_models_pytorch as smp
    return smp.Unet(
        encoder_name="efficientnet-b3",
        encoder_weights="imagenet",
        in_channels=3,
        classes=2,
    )


def compute_miou(model: nn.Module, loader: DataLoader, device: torch.device, max_batches: int = 0) -> dict:
    model.eval()
    tp_fg = 0
    fp_fg = 0
    fn_fg = 0
    tp_bg = 0
    fp_bg = 0
    fn_bg = 0
    with torch.no_grad():
        for step, batch in enumerate(loader):
            if max_batches and step >= max_batches:
                break
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            logits = model(images)
            pred = logits.argmax(dim=1)
            fg_p = (pred == 1)
            fg_t = (masks == 1)
            tp_fg += (fg_p & fg_t).sum().item()
            fp_fg += (fg_p & ~fg_t).sum().item()
            fn_fg += (~fg_p & fg_t).sum().item()
            bg_p = (pred == 0)
            bg_t = (masks == 0)
            tp_bg += (bg_p & bg_t).sum().item()
            fp_bg += (bg_p & ~bg_t).sum().item()
            fn_bg += (~bg_p & bg_t).sum().item()
    iou_fg = tp_fg / max(tp_fg + fp_fg + fn_fg, 1)
    iou_bg = tp_bg / max(tp_bg + fp_bg + fn_bg, 1)
    return {
        "miou": float((iou_fg + iou_bg) / 2),
        "iou_fg": float(iou_fg),
        "iou_bg": float(iou_bg),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mirror-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--val-every", type=int, default=1)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--max-train-steps", type=int, default=0)
    p.add_argument("--max-val-batches", type=int, default=0)
    return p.parse_args()


def main() -> None:
    import segmentation_models_pytorch as smp

    args = parse_args()
    set_seed(args.seed)
    ckpt_dir = args.output_dir / "checkpoints"
    logs_dir = args.output_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    dice_loss = smp.losses.DiceLoss(mode="multiclass")
    ce_loss = nn.CrossEntropyLoss()

    train_ds = UnetSegImageMaskDataset(args.mirror_root, split="train", augment=True)
    val_ds = UnetSegImageMaskDataset(args.mirror_root, split="val", augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, args.batch_size // 2),
        shuffle=False,
        num_workers=max(2, args.num_workers // 2),
        pin_memory=True,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    log_file = logs_dir / "train.log"
    best_miou = -1.0
    t0 = time.time()

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_steps = 0
        ep_start = time.time()

        for step, batch in enumerate(train_loader):
            if args.max_train_steps and step >= args.max_train_steps:
                break
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                logits = model(images)
                loss = 0.5 * dice_loss(logits, masks) + 0.5 * ce_loss(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += float(loss.detach().cpu())
            n_steps += 1
            if step % args.log_every == 0:
                msg = (
                    f"[epoch {epoch}/{args.max_epochs} step {step}/{len(train_loader)}] "
                    f"loss={float(loss.detach().cpu()):.4f}"
                )
                print(msg, flush=True)
                with log_file.open("a") as f:
                    f.write(msg + "\n")

        avg_loss = epoch_loss / max(n_steps, 1)
        ep_time = time.time() - ep_start

        val_stats = {}
        if epoch % args.val_every == 0:
            val_stats = compute_miou(model, val_loader, device, args.max_val_batches)

        summary = {
            "epoch": epoch,
            "avg_train_loss": avg_loss,
            "epoch_time_sec": ep_time,
            "lr": args.lr,
            "elapsed_sec": time.time() - t0,
            **{f"val_{k}": v for k, v in val_stats.items()},
        }
        with log_file.open("a") as f:
            f.write(json.dumps(summary) + "\n")
        print(json.dumps(summary), flush=True)

        last_path = ckpt_dir / "last.pt"
        torch.save({"model": model.state_dict(), "epoch": epoch}, last_path)
        miou_now = val_stats.get("miou", -1.0)
        if miou_now > best_miou:
            best_miou = miou_now
            best_path = ckpt_dir / "best.pt"
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_miou": miou_now}, best_path)

    print(f"[unet_seg train done] best_miou={best_miou:.4f}, total_time={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
