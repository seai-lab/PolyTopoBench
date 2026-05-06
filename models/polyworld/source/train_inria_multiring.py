#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader_hisup_multiring import HiSupMultiringTrainDataset
from models.backbone import DetectionBranch, NonMaxSuppression, R2U_Net
from models.matching import OptimalMatching


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_TRAIN_IMAGES = REPO_ROOT / "data_processed" / "inria_building" / "for_hisup" / "train" / "images"
DEFAULT_TRAIN_ANN = REPO_ROOT / "data_processed" / "inria_building" / "for_hisup" / "train" / "annotation.json"
DEFAULT_VAL_IMAGES = REPO_ROOT / "data_processed" / "inria_building" / "for_hisup" / "val" / "images"
DEFAULT_VAL_ANN = REPO_ROOT / "data_processed" / "inria_building" / "for_hisup" / "val" / "annotation.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "polyworld" / "inria" / "multiring_train"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train PolyWorld on HiSup-style multi-ring annotations by supervising "
            "vertex heatmaps and multi-cycle permutation targets."
        )
    )
    parser.add_argument("--train-images-directory", type=Path, default=DEFAULT_TRAIN_IMAGES)
    parser.add_argument("--train-annotations-path", type=Path, default=DEFAULT_TRAIN_ANN)
    parser.add_argument("--val-images-directory", type=Path, default=DEFAULT_VAL_IMAGES)
    parser.add_argument("--val-annotations-path", type=Path, default=DEFAULT_VAL_ANN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--window-size", type=int, default=320)
    parser.add_argument("--max-points", type=int, default=256)
    parser.add_argument(
        "--target-rings",
        type=str,
        choices=("all", "exterior"),
        default="all",
        help="Which rings from each HiSup building annotation to supervise.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--heatmap-loss-weight", type=float, default=1.0)
    parser.add_argument("--perm-loss-weight", type=float, default=1.0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-image-limit", type=int, default=None)
    parser.add_argument("--val-image-limit", type=int, default=None)
    parser.add_argument("--max-train-steps-per-epoch", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=20)
    # v4 additions
    parser.add_argument("--heatmap-radius", type=int, default=0,
                        help="Radius (pixels) of Gaussian/disk heatmap target around each GT vertex. 0 = single-pixel.")
    parser.add_argument("--heatmap-sigma", type=float, default=0.0,
                        help="Gaussian sigma (pixels) for heatmap target; ignored when heatmap-radius=0.")
    parser.add_argument("--pos-weight-cap", type=float, default=100.0,
                        help="Upper clamp of the pos_weight term used in balanced BCE.")
    parser.add_argument("--use-d4-aug", action="store_true",
                        help="Enable random D4 (rot0/90/180/270 + flips) synchronous augmentation on train.")
    parser.add_argument("--nms-border-margin", type=int, default=0,
                        help="Border margin in pixels where NMS peaks are suppressed (diagnostics + val).")
    parser.add_argument("--nms-score-threshold", type=float, default=0.0,
                        help="Sigmoid score threshold below which NMS peaks are suppressed.")
    parser.add_argument("--best-selection", type=str, default="val_loss",
                        choices=("val_loss", "recall5"),
                        help="Primary criterion for selecting weights_best; both ckpts are always saved.")
    parser.add_argument("--diag-match-radius", type=int, default=5,
                        help="Pixel radius used when computing vertex recall@K / precision@K diagnostics.")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def balanced_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None = None,
    pos_weight_cap: float = 100.0,
) -> torch.Tensor:
    if mask is not None:
        logits = logits[mask]
        targets = targets[mask]
    if logits.numel() == 0:
        return logits.sum() * 0.0
    positives = targets.sum()
    negatives = targets.numel() - positives
    if positives.item() > 0 and negatives.item() > 0:
        pos_weight = torch.clamp(negatives / positives, min=1.0, max=float(pos_weight_cap))
        pos_weight = pos_weight.to(device=logits.device, dtype=logits.dtype)
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
    return F.binary_cross_entropy_with_logits(logits, targets)


def build_dataloaders(args: argparse.Namespace, device: torch.device):
    train_dataset = HiSupMultiringTrainDataset(
        images_directory=args.train_images_directory,
        annotations_path=args.train_annotations_path,
        window_size=args.window_size,
        max_points=args.max_points,
        image_limit=args.train_image_limit,
        target_rings=args.target_rings,
        heatmap_radius=args.heatmap_radius,
        heatmap_sigma=args.heatmap_sigma,
        use_d4_aug=args.use_d4_aug,
    )
    val_dataset = HiSupMultiringTrainDataset(
        images_directory=args.val_images_directory,
        annotations_path=args.val_annotations_path,
        window_size=args.window_size,
        max_points=args.max_points,
        image_limit=args.val_image_limit,
        target_rings=args.target_rings,
        heatmap_radius=args.heatmap_radius,
        heatmap_sigma=args.heatmap_sigma,
        use_d4_aug=False,  # never augment val
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    return train_loader, val_loader


def init_modules(args: argparse.Namespace, device: torch.device):
    backbone = R2U_Net().to(device)
    seg_head = DetectionBranch().to(device)
    matching = OptimalMatching().to(device)
    return backbone, seg_head, matching


def compute_batch_losses(
    batch,
    backbone: R2U_Net,
    seg_head: DetectionBranch,
    matching: OptimalMatching,
    device: torch.device,
    heatmap_loss_weight: float,
    perm_loss_weight: float,
    pos_weight_cap: float = 100.0,
    return_logits: bool = False,
):
    images = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
    heatmap_targets = batch["heatmap"].to(device=device, dtype=torch.float32, non_blocking=True)
    graph = batch["graph"].to(device=device, dtype=torch.long, non_blocking=True)
    perm_targets = batch["perm_target"].to(device=device, dtype=torch.float32, non_blocking=True)
    perm_mask = batch["perm_mask"].to(device=device, dtype=torch.bool, non_blocking=True)

    features = backbone(images)
    heatmap_logits = seg_head(features)
    perm_logits, _graph_corrected = matching.compute_scores(features, graph)

    heatmap_loss = balanced_bce_with_logits(heatmap_logits, heatmap_targets, pos_weight_cap=pos_weight_cap)
    perm_loss = balanced_bce_with_logits(perm_logits, perm_targets, mask=perm_mask, pos_weight_cap=pos_weight_cap)
    total_loss = heatmap_loss_weight * heatmap_loss + perm_loss_weight * perm_loss

    stats = {
        "loss": float(total_loss.detach().item()),
        "heatmap_loss": float(heatmap_loss.detach().item()),
        "perm_loss": float(perm_loss.detach().item()),
        "kept_vertices_mean": float(batch["stats_kept_vertex_count"].float().mean().item()),
        "kept_rings_mean": float(batch["stats_kept_ring_count"].float().mean().item()),
        "dropped_rings_mean": float(batch["stats_dropped_ring_count"].float().mean().item()),
    }
    extras = None
    if return_logits:
        extras = {
            "heatmap_logits": heatmap_logits.detach(),
            "heatmap_targets": heatmap_targets.detach(),
            "graph_gt": graph.detach(),
            "valid_vertex_mask": batch["valid_vertex_mask"].to(device=device, dtype=torch.bool, non_blocking=True),
        }
    return total_loss, stats, extras


def _pairwise_match_count(pred_xy: np.ndarray, gt_xy: np.ndarray, thr: float) -> tuple[int, int, int]:
    """Greedy nearest-first matching. Returns (tp, pred_matched, gt_matched)."""
    if len(pred_xy) == 0 or len(gt_xy) == 0:
        return 0, 0, 0
    diff = pred_xy[:, None, :] - gt_xy[None, :, :]
    dist2 = (diff[..., 0] ** 2 + diff[..., 1] ** 2)
    # For each pred, pick closest GT
    best_gt = dist2.argmin(axis=1)
    best_d2 = dist2[np.arange(len(pred_xy)), best_gt]
    # Enforce unique GT assignment greedily by sorted distance
    order = np.argsort(best_d2)
    used_gt = set()
    tp = 0
    for p_idx in order:
        if best_d2[p_idx] <= thr * thr:
            g = int(best_gt[p_idx])
            if g not in used_gt:
                used_gt.add(g)
                tp += 1
    return tp, len(pred_xy), len(gt_xy)


def compute_heatmap_diagnostics(
    heatmap_logits: torch.Tensor,
    graph_gt: torch.Tensor,
    valid_mask: torch.Tensor,
    nms_module: NonMaxSuppression,
    heatmap_targets: torch.Tensor,
    thr2: float,
    thr5: float,
    border_margin: int,
) -> dict:
    """Per-batch vertex-level diagnostics in window-pixel space."""
    _, peaks = nms_module(heatmap_logits)   # peaks: (B, n_peaks, 2), (row, col)
    probs = torch.sigmoid(heatmap_logits)
    B, _, H, W = heatmap_logits.shape

    # Convert to numpy for matching
    peaks_np = peaks.detach().cpu().numpy()
    graph_np = graph_gt.detach().cpu().numpy()
    valid_np = valid_mask.detach().cpu().numpy()
    probs_np = probs.detach().cpu().numpy()[:, 0]
    targets_np = heatmap_targets.detach().cpu().numpy()[:, 0]

    tp2 = pr2 = gt2 = 0
    tp5 = pr5 = gt5 = 0
    border_count = 0
    pos_score_sum = 0.0
    pos_count = 0
    neg_score_sum = 0.0
    neg_count = 0
    for b in range(B):
        pb = peaks_np[b]   # (N, 2) row, col
        gb = graph_np[b]   # (max_points, 2)
        vb = valid_np[b]   # (max_points,)
        gt_pts = gb[vb]
        if border_margin > 0:
            m = border_margin
            if len(pb) > 0:
                keep = (pb[:, 0] >= m) & (pb[:, 0] < H - m) & (pb[:, 1] >= m) & (pb[:, 1] < W - m)
                border_count += int((~keep).sum())
                pb_filt = pb[keep]
            else:
                pb_filt = pb
        else:
            pb_filt = pb
        # recall/precision@2 and @5
        tp_b, pr_b, gt_b = _pairwise_match_count(pb_filt.astype(np.float32), gt_pts.astype(np.float32), thr2)
        tp2 += tp_b; pr2 += pr_b; gt2 += gt_b
        tp_b, pr_b, gt_b = _pairwise_match_count(pb_filt.astype(np.float32), gt_pts.astype(np.float32), thr5)
        tp5 += tp_b; pr5 += pr_b; gt5 += gt_b
        # pos/neg score averages using the batch's own Gaussian targets (mask = targets > 0.5 for positives)
        pos_pixel_mask = targets_np[b] > 0.5
        neg_pixel_mask = targets_np[b] < 0.05
        if pos_pixel_mask.any():
            pos_score_sum += float(probs_np[b][pos_pixel_mask].mean())
            pos_count += 1
        if neg_pixel_mask.any():
            neg_score_sum += float(probs_np[b][neg_pixel_mask].mean())
            neg_count += 1

    return {
        "recall_at_2": (tp2 / gt2) if gt2 else 0.0,
        "precision_at_2": (tp2 / pr2) if pr2 else 0.0,
        "recall_at_5": (tp5 / gt5) if gt5 else 0.0,
        "precision_at_5": (tp5 / pr5) if pr5 else 0.0,
        "border_peak_ratio": (border_count / (B * peaks_np.shape[1])) if peaks_np.size else 0.0,
        "mean_pos_score": (pos_score_sum / pos_count) if pos_count else 0.0,
        "mean_neg_score": (neg_score_sum / neg_count) if neg_count else 0.0,
    }


def run_epoch(
    loader,
    backbone: R2U_Net,
    seg_head: DetectionBranch,
    matching: OptimalMatching,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    args: argparse.Namespace,
    desc: str,
    max_batches: int | None = None,
    diag_nms: NonMaxSuppression | None = None,
):
    train_mode = optimizer is not None
    if train_mode:
        backbone.train()
        seg_head.train()
        matching.train()
    else:
        backbone.eval()
        seg_head.eval()
        matching.eval()

    aggregates = {
        "loss": 0.0,
        "heatmap_loss": 0.0,
        "perm_loss": 0.0,
        "kept_vertices_mean": 0.0,
        "kept_rings_mean": 0.0,
        "dropped_rings_mean": 0.0,
    }
    diag_aggregates = {
        "recall_at_2": 0.0,
        "precision_at_2": 0.0,
        "recall_at_5": 0.0,
        "precision_at_5": 0.0,
        "border_peak_ratio": 0.0,
        "mean_pos_score": 0.0,
        "mean_neg_score": 0.0,
    }
    count = 0
    diag_count = 0

    iterator = tqdm(loader, desc=desc, leave=False)
    for batch_index, batch in enumerate(iterator, start=1):
        if max_batches is not None and batch_index > max_batches:
            break

        context = torch.enable_grad() if train_mode else torch.no_grad()
        want_diag = (diag_nms is not None) and (not train_mode)
        with context:
            loss, stats, extras = compute_batch_losses(
                batch=batch,
                backbone=backbone,
                seg_head=seg_head,
                matching=matching,
                device=device,
                heatmap_loss_weight=args.heatmap_loss_weight,
                perm_loss_weight=args.perm_loss_weight,
                pos_weight_cap=args.pos_weight_cap,
                return_logits=want_diag,
            )
            if want_diag and extras is not None:
                diag = compute_heatmap_diagnostics(
                    heatmap_logits=extras["heatmap_logits"],
                    graph_gt=extras["graph_gt"],
                    valid_mask=extras["valid_vertex_mask"],
                    nms_module=diag_nms,
                    heatmap_targets=extras["heatmap_targets"],
                    thr2=2.0,
                    thr5=float(args.diag_match_radius),
                    border_margin=args.nms_border_margin,
                )
                for k, v in diag.items():
                    diag_aggregates[k] += v
                diag_count += 1

        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        count += 1
        for key, value in stats.items():
            aggregates[key] += value
        iterator.set_postfix(
            loss=f"{aggregates['loss'] / count:.4f}",
            perm=f"{aggregates['perm_loss'] / count:.4f}",
            heatmap=f"{aggregates['heatmap_loss'] / count:.4f}",
        )

        if train_mode and args.log_interval > 0 and batch_index % args.log_interval == 0:
            print(
                f"[train] step={batch_index} loss={aggregates['loss'] / count:.4f} "
                f"heatmap={aggregates['heatmap_loss'] / count:.4f} perm={aggregates['perm_loss'] / count:.4f}"
            )

    if count == 0:
        out = {key: None for key in aggregates}
        out.update({key: None for key in diag_aggregates})
        return out
    result = {key: value / count for key, value in aggregates.items()}
    if diag_count > 0:
        for k, v in diag_aggregates.items():
            result[k] = v / diag_count
    else:
        for k in diag_aggregates:
            result[k] = None
    return result


def export_weight_triplet(weights_dir: Path, backbone, seg_head, matching) -> None:
    weights_dir.mkdir(parents=True, exist_ok=True)
    torch.save(backbone.state_dict(), weights_dir / "polyworld_backbone")
    torch.save(seg_head.state_dict(), weights_dir / "polyworld_seg_head")
    torch.save(matching.state_dict(), weights_dir / "polyworld_matching")


def save_epoch_artifacts(
    output_dir: Path,
    epoch: int,
    backbone,
    seg_head,
    matching,
    optimizer,
    history: list[dict],
    is_best_valloss: bool,
    is_best_recall5: bool,
    primary_is_recall5: bool,
) -> None:
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    training_state = {
        "epoch": epoch,
        "backbone": backbone.state_dict(),
        "seg_head": seg_head.state_dict(),
        "matching": matching.state_dict(),
        "optimizer": optimizer.state_dict(),
        "history": history,
    }
    torch.save(training_state, checkpoints_dir / "training_state_latest.pt")
    export_weight_triplet(output_dir / "weights_last", backbone, seg_head, matching)

    if is_best_valloss:
        torch.save(training_state, checkpoints_dir / "training_state_best_valloss.pt")
        export_weight_triplet(output_dir / "weights_best_valloss", backbone, seg_head, matching)

    if is_best_recall5:
        torch.save(training_state, checkpoints_dir / "training_state_best_recall5.pt")
        export_weight_triplet(output_dir / "weights_best_recall5", backbone, seg_head, matching)

    # Maintain legacy "weights_best" symlink pointing at the primary best for compatibility
    primary_src = output_dir / ("weights_best_recall5" if primary_is_recall5 else "weights_best_valloss")
    primary_dst = output_dir / "weights_best"
    if primary_src.exists():
        if primary_dst.is_symlink() or primary_dst.exists():
            try:
                if primary_dst.is_symlink() or primary_dst.is_file():
                    primary_dst.unlink()
                else:
                    import shutil
                    shutil.rmtree(primary_dst)
            except Exception:
                pass
        try:
            primary_dst.symlink_to(primary_src.name, target_is_directory=True)
        except Exception:
            # Filesystem without symlink support: copy instead
            import shutil
            shutil.copytree(primary_src, primary_dst, dirs_exist_ok=True)

    history_path = output_dir / "history.json"
    history_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "config.json"
    config_path.write_text(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=2) + "\n", encoding="utf-8")

    train_loader, val_loader = build_dataloaders(args, device)
    backbone, seg_head, matching = init_modules(args, device)
    optimizer = torch.optim.AdamW(
        list(backbone.parameters()) + list(seg_head.parameters()) + list(matching.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    history: list[dict] = []
    best_val_loss = float("inf")
    best_recall5 = -1.0

    diag_nms = NonMaxSuppression(
        n_peaks=args.max_points,
        border_margin=args.nms_border_margin,
        score_threshold=args.nms_score_threshold,
    ).to(device)

    print(
        f"Training PolyWorld multiring on {device} | "
        f"train_images={len(train_loader.dataset)} val_images={len(val_loader.dataset)} "
        f"target_rings={args.target_rings} | "
        f"heatmap_radius={args.heatmap_radius} sigma={args.heatmap_sigma} | "
        f"d4_aug={args.use_d4_aug} pos_weight_cap={args.pos_weight_cap} | "
        f"hm_w={args.heatmap_loss_weight} perm_w={args.perm_loss_weight} | "
        f"nms_border={args.nms_border_margin} nms_score={args.nms_score_threshold} | "
        f"best={args.best_selection}"
    )

    primary_is_recall5 = (args.best_selection == "recall5")
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(
            loader=train_loader,
            backbone=backbone,
            seg_head=seg_head,
            matching=matching,
            optimizer=optimizer,
            device=device,
            args=args,
            desc=f"Train {epoch}/{args.epochs}",
            max_batches=args.max_train_steps_per_epoch,
            diag_nms=None,
        )
        val_stats = run_epoch(
            loader=val_loader,
            backbone=backbone,
            seg_head=seg_head,
            matching=matching,
            optimizer=None,
            device=device,
            args=args,
            desc=f"Val {epoch}/{args.epochs}",
            max_batches=args.max_val_batches,
            diag_nms=diag_nms,
        )

        epoch_record = {
            "epoch": epoch,
            "train": train_stats,
            "val": val_stats,
            "elapsed_sec": time.time() - start_time,
        }
        history.append(epoch_record)
        val_loss = val_stats["loss"] if val_stats["loss"] is not None else float("inf")
        recall5 = val_stats.get("recall_at_5")
        recall5 = -1.0 if recall5 is None else float(recall5)

        is_best_valloss = val_loss < best_val_loss
        if is_best_valloss:
            best_val_loss = val_loss
        is_best_recall5 = recall5 > best_recall5
        if is_best_recall5:
            best_recall5 = recall5

        save_epoch_artifacts(
            output_dir=args.output_dir,
            epoch=epoch,
            backbone=backbone,
            seg_head=seg_head,
            matching=matching,
            optimizer=optimizer,
            history=history,
            is_best_valloss=is_best_valloss,
            is_best_recall5=is_best_recall5,
            primary_is_recall5=primary_is_recall5,
        )
        r5_str = "N/A" if val_stats.get("recall_at_5") is None else f"{val_stats['recall_at_5']:.3f}"
        r2_str = "N/A" if val_stats.get("recall_at_2") is None else f"{val_stats['recall_at_2']:.3f}"
        bord_str = "N/A" if val_stats.get("border_peak_ratio") is None else f"{val_stats['border_peak_ratio']:.3f}"
        pos_str = "N/A" if val_stats.get("mean_pos_score") is None else f"{val_stats['mean_pos_score']:.3f}"
        neg_str = "N/A" if val_stats.get("mean_neg_score") is None else f"{val_stats['mean_neg_score']:.3f}"
        print(
            f"[epoch {epoch}] train={train_stats['loss']:.4f} val={val_stats['loss']:.4f} "
            f"best_val={best_val_loss:.4f} best_r5={best_recall5:.3f} | "
            f"R@2={r2_str} R@5={r5_str} border%={bord_str} posS={pos_str} negS={neg_str}"
        )

    summary = {
        "best_val_loss": best_val_loss,
        "best_recall5": best_recall5,
        "primary_selection": args.best_selection,
        "epochs": args.epochs,
        "device": str(device),
        "target_rings": args.target_rings,
        "train_images": len(train_loader.dataset),
        "val_images": len(val_loader.dataset),
        "output_dir": str(args.output_dir),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
