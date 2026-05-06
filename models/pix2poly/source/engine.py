import gc
import os
from contextlib import nullcontext

from tqdm import tqdm
import torch

from utils import (
    AverageMeter,
    get_lr,
    save_checkpoint,
    save_single_predictions_as_images
)
from config import CFG

from ddp_utils import is_main_process


def _get_autocast_context():
    use_amp = CFG.USE_AMP and CFG.DEVICE.type == "cuda"
    if use_amp:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _get_loss_weights(epoch):
    if epoch < CFG.MILESTONE:
        return CFG.vertex_loss_weight, 0.0
    return CFG.vertex_loss_weight, CFG.perm_loss_weight


def _compute_perm_loss(perm_loss_fn, perm_mat, y_perm, perm_loss_weight):
    # BCELoss on probability tensors is not autocast-safe in PyTorch.
    # Keep the permutation branch in float32 while the rest of the forward
    # pass can still benefit from AMP.
    return perm_loss_weight * perm_loss_fn(perm_mat.float(), y_perm.float())


def _tensor_stats(name, tensor):
    safe_tensor = torch.nan_to_num(tensor.detach(), nan=0.0, posinf=0.0, neginf=0.0)
    return (
        f"{name}: min={safe_tensor.min().item():.6f}, "
        f"max={safe_tensor.max().item():.6f}, "
        f"nan={int(torch.isnan(tensor).sum().item())}, "
        f"inf={int(torch.isinf(tensor).sum().item())}"
    )


def _has_invalid_perm_inputs(preds, perm_mat, y_perm):
    if not torch.isfinite(preds).all():
        return True
    if not torch.isfinite(perm_mat).all():
        return True
    if not torch.isfinite(y_perm).all():
        return True
    if bool((perm_mat < 0).any()) or bool((perm_mat > 1).any()):
        return True
    return False


def _log_invalid_perm_inputs(epoch, step_idx, preds, perm_mat, y_perm):
    print(
        f"Skipping invalid permutation batch at epoch={epoch + 1}, step={step_idx}. "
        f"{_tensor_stats('preds', preds)}; "
        f"{_tensor_stats('perm_mat', perm_mat)}; "
        f"{_tensor_stats('y_perm', y_perm)}"
    )


def _cleanup_after_invalid_batch():
    gc.collect()
    if CFG.DEVICE.type == "cuda":
        torch.cuda.empty_cache()


def _unpack_supervised_batch(batch):
    if len(batch) == 6:
        x, y_mask, y_corner_mask, y, y_perm, sample_names = batch
        return x, y_mask, y_corner_mask, y, y_perm, sample_names
    x, y_mask, y_corner_mask, y, y_perm = batch
    return x, y_mask, y_corner_mask, y, y_perm, None


def _format_sample_names(sample_names):
    if not sample_names:
        return "samples=<unavailable>"
    return "samples=" + ", ".join(sample_names)


def _log_invalid_forward(epoch, step_idx, preds, perm_mat, y_perm, sample_names=None):
    print(
        f"Skipping invalid forward batch at epoch={epoch + 1}, step={step_idx}. "
        f"{_format_sample_names(sample_names)}; "
        f"{_tensor_stats('preds', preds)}; "
        f"{_tensor_stats('perm_mat', perm_mat)}; "
        f"{_tensor_stats('y_perm', y_perm)}"
    )


def _log_invalid_vertex_loss(epoch, step_idx, vertex_loss, preds, perm_mat, y_perm, sample_names=None):
    print(
        f"Skipping invalid vertex loss batch at epoch={epoch + 1}, step={step_idx}. "
        f"{_format_sample_names(sample_names)}; "
        f"vertex_loss={float(vertex_loss.detach().float().item()) if torch.isfinite(vertex_loss).all() else 'nan_or_inf'}; "
        f"{_tensor_stats('preds', preds)}; "
        f"{_tensor_stats('perm_mat', perm_mat)}; "
        f"{_tensor_stats('y_perm', y_perm)}"
    )


def train_one_epoch(epoch, iter_idx, model, train_loader, optimizer, lr_scheduler, vertex_loss_fn, perm_loss_fn, writer, scaler):
    model.train()
    vertex_loss_fn.train()
    perm_loss_fn.train()

    loss_meter = AverageMeter()
    vertex_loss_meter = AverageMeter()
    perm_loss_meter = AverageMeter()

    loader = train_loader
    if is_main_process():
        loader = tqdm(train_loader, total=len(train_loader))

    # prof = torch.profiler.profile(
    #     schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler(f"runs/{CFG.EXPERIMENT_NAME}/logs/profiler"),
    #     record_shapes=True,
    #     with_stack=True
    # )
    # prof.start()
    for step_idx, batch in enumerate(loader):
        optimizer.zero_grad(set_to_none=True)

        x, y_mask, y_corner_mask, y, y_perm, sample_names = _unpack_supervised_batch(batch)

        x = x.to(CFG.DEVICE, non_blocking=True)
        y = y.to(CFG.DEVICE, non_blocking=True)
        y_perm = y_perm.to(CFG.DEVICE, non_blocking=True)

        y_input = y[:, :-1]
        y_expected = y[:, 1:]
        vertex_loss_weight, perm_loss_weight = _get_loss_weights(epoch)

        with _get_autocast_context():
            preds, perm_mat = model(x, y_input)

        if _has_invalid_perm_inputs(preds, perm_mat, y_perm):
            if is_main_process():
                _log_invalid_forward(epoch, step_idx, preds, perm_mat, y_perm, sample_names)
            del preds, perm_mat, x, y, y_perm, y_input, y_expected
            _cleanup_after_invalid_batch()
            continue

        with _get_autocast_context():
            vertex_loss = vertex_loss_weight*vertex_loss_fn(preds.reshape(-1, preds.shape[-1]), y_expected.reshape(-1))

        if not torch.isfinite(vertex_loss).all():
            if is_main_process():
                _log_invalid_vertex_loss(epoch, step_idx, vertex_loss, preds, perm_mat, y_perm, sample_names)
            del vertex_loss, preds, perm_mat, x, y, y_perm, y_input, y_expected
            _cleanup_after_invalid_batch()
            continue

        perm_loss = _compute_perm_loss(perm_loss_fn, perm_mat, y_perm, perm_loss_weight)
        loss = vertex_loss + perm_loss

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            if CFG.MAX_GRAD_NORM > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CFG.MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if CFG.MAX_GRAD_NORM > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CFG.MAX_GRAD_NORM)
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        loss_meter.update(loss.item(), x.size(0))
        vertex_loss_meter.update(vertex_loss.item(), x.size(0))
        perm_loss_meter.update(perm_loss.item(), x.size(0))

        lr = get_lr(optimizer)
        if is_main_process():
            loader.set_postfix(train_loss=loss_meter.avg, lr=f"{lr:.5f}")
            writer.add_scalar('Running_logs/Train_Loss', loss_meter.avg, iter_idx)
            writer.add_scalar('Running_logs/LR', lr, iter_idx)
            # writer.add_image(f"Running_logs/input_images", torchvision.utils.make_grid(x), iter_idx)
            # writer.add_graph(model, input_to_model=(x, y_input))

        iter_idx += 1
        if CFG.DEBUG_MAX_TRAIN_STEPS and step_idx + 1 >= CFG.DEBUG_MAX_TRAIN_STEPS:
            if is_main_process():
                print(f"Stopping train epoch early at debug step limit {CFG.DEBUG_MAX_TRAIN_STEPS}.")
            break
        # prof.step()
    # prof.stop()
    print(f"Total train loss: {loss_meter.avg}\n\n")
    loss_dict = {
        'total_loss': loss_meter.avg,
        'vertex_loss': vertex_loss_meter.avg,
        'perm_loss': perm_loss_meter.avg,
    }

    return loss_dict, iter_idx


def valid_one_epoch(epoch, model, valid_loader, vertex_loss_fn, perm_loss_fn):
    print(f"\nValidating...")
    model.eval()
    vertex_loss_fn.eval()
    perm_loss_fn.eval()

    loss_meter = AverageMeter()
    vertex_loss_meter = AverageMeter()
    perm_loss_meter = AverageMeter()

    loader = valid_loader
    if is_main_process():
        loader = tqdm(valid_loader, total=len(valid_loader))

    with torch.no_grad():
        for step_idx, batch in enumerate(loader):
            x, y_mask, y_corner_mask, y, y_perm, sample_names = _unpack_supervised_batch(batch)
            x = x.to(CFG.DEVICE, non_blocking=True)
            y = y.to(CFG.DEVICE, non_blocking=True)
            y_perm = y_perm.to(CFG.DEVICE, non_blocking=True)

            y_input = y[:, :-1]
            y_expected = y[:, 1:]
            vertex_loss_weight, perm_loss_weight = _get_loss_weights(epoch)

            with _get_autocast_context():
                preds, perm_mat = model(x, y_input)

            if _has_invalid_perm_inputs(preds, perm_mat, y_perm):
                if is_main_process():
                    _log_invalid_forward(epoch, step_idx, preds, perm_mat, y_perm, sample_names)
                del preds, perm_mat, x, y, y_perm, y_input, y_expected
                _cleanup_after_invalid_batch()
                continue

            with _get_autocast_context():
                vertex_loss = vertex_loss_weight*vertex_loss_fn(preds.reshape(-1, preds.shape[-1]), y_expected.reshape(-1))

            if not torch.isfinite(vertex_loss).all():
                if is_main_process():
                    _log_invalid_vertex_loss(epoch, step_idx, vertex_loss, preds, perm_mat, y_perm, sample_names)
                del vertex_loss, preds, perm_mat, x, y, y_perm, y_input, y_expected
                _cleanup_after_invalid_batch()
                continue

            perm_loss = _compute_perm_loss(perm_loss_fn, perm_mat, y_perm, perm_loss_weight)
            loss = vertex_loss + perm_loss

            loss_meter.update(loss.item(), x.size(0))
            vertex_loss_meter.update(vertex_loss.item(), x.size(0))
            perm_loss_meter.update(perm_loss.item(), x.size(0))

            if CFG.DEBUG_MAX_VAL_STEPS and step_idx + 1 >= CFG.DEBUG_MAX_VAL_STEPS:
                if is_main_process():
                    print(f"Stopping validation early at debug step limit {CFG.DEBUG_MAX_VAL_STEPS}.")
                break

        loss_dict = {
        'total_loss': loss_meter.avg,
        'vertex_loss': vertex_loss_meter.avg,
        'perm_loss': perm_loss_meter.avg,
    }

    return loss_dict


def train_eval(
    model,
    train_loader,
    valid_loader,
    test_loader,
    tokenizer,
    vertex_loss_fn,
    perm_loss_fn,
    optimizer,
    lr_scheduler,
    step,
    writer,
    scaler
):
    best_loss = float('inf')
    best_metric = float('-inf')

    iter_idx=CFG.START_EPOCH * len(train_loader)
    epoch_iterator = range(CFG.START_EPOCH, CFG.NUM_EPOCHS)
    if is_main_process():
        epoch_iterator = tqdm(epoch_iterator)
    for epoch in epoch_iterator:
        if is_main_process():
            print(f"\n\nEPOCH: {epoch + 1}\n\n")

        if CFG.TRAIN_DDP:
            train_loader.sampler.set_epoch(epoch)
            valid_loader.sampler.set_epoch(epoch)
            test_loader.sampler.set_epoch(epoch)

        train_loss_dict, iter_idx = train_one_epoch(
            epoch,
            iter_idx,
            model,
            train_loader,
            optimizer,
            lr_scheduler if step=='batch' else None,
            vertex_loss_fn,
            perm_loss_fn,
            writer,
            scaler
        )
        if is_main_process():
            writer.add_scalar('Train_Losses/Total_Loss', train_loss_dict['total_loss'], epoch)
            writer.add_scalar('Train_Losses/Vertex_Loss', train_loss_dict['vertex_loss'], epoch)
            writer.add_scalar('Train_Losses/Perm_Loss', train_loss_dict['perm_loss'], epoch)

        valid_loss_dict = valid_one_epoch(
            epoch,
            model,
            valid_loader,
            vertex_loss_fn,
            perm_loss_fn,
        )  # TODO: add eval metrics to validation function?
        if is_main_process():
            print(f"Valid loss: {valid_loss_dict['total_loss']:.3f}\n\n")

        if step=='epoch':
            pass

        # Save best validation loss epoch.
        if valid_loss_dict['total_loss'] < best_loss and CFG.SAVE_BEST and is_main_process():
            best_loss = valid_loss_dict['total_loss']
            checkpoint = {
                "state_dict": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": lr_scheduler.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None and scaler.is_enabled() else None,
                "epochs_run": epoch,
                "loss": train_loss_dict["total_loss"]
            }
            save_checkpoint(
                checkpoint,
                folder=os.path.join(CFG.RUN_DIR, "logs", "checkpoints"),
                filename="best_valid_loss.pth"
            )
            # torch.save(model.state_dict(), 'best_valid_loss.pth')
            print(f"Saved best val loss model.")

        # Save latest checkpoint every epoch.
        if CFG.SAVE_LATEST and is_main_process():
            checkpoint = {
                    "state_dict": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": lr_scheduler.state_dict(),
                    "scaler": scaler.state_dict() if scaler is not None and scaler.is_enabled() else None,
                    "epochs_run": epoch,
                    "loss": train_loss_dict["total_loss"]
                }
            save_checkpoint(
                checkpoint,
                folder=os.path.join(CFG.RUN_DIR, "logs", "checkpoints"),
                filename="latest.pth"
            )

        if (epoch + 1) % CFG.SAVE_EVERY == 0 and is_main_process():
            checkpoint = {
                "state_dict": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": lr_scheduler.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None and scaler.is_enabled() else None,
                "epochs_run": epoch,
                "loss": train_loss_dict["total_loss"]
            }
            save_checkpoint(
                checkpoint,
                folder=os.path.join(CFG.RUN_DIR, "logs", "checkpoints"),
                filename=f"epoch_{epoch}.pth"
            )

        if is_main_process():
            writer.add_scalar('Val_Losses/Total_Loss', valid_loss_dict['total_loss'], epoch)
            writer.add_scalar('Val_Losses/Vertex_Loss', valid_loss_dict['vertex_loss'], epoch)
            writer.add_scalar('Val_Losses/Perm_Loss', valid_loss_dict['perm_loss'], epoch)

        # output examples to a folder
        if (epoch + 1) % CFG.VAL_EVERY == 0 and is_main_process():
            val_metrics_dict = save_single_predictions_as_images(
                test_loader,
                model,
                tokenizer,
                epoch,
                writer,
                folder=os.path.join(CFG.RUN_DIR, "runtime_outputs"),
                device=CFG.DEVICE
            )
            for metric, value in zip(val_metrics_dict.keys(), val_metrics_dict.values()):
                print(f"{metric}: {value}")

            # Save best single batch validation metric epoch.
            if val_metrics_dict["miou"] > best_metric and CFG.SAVE_BEST and is_main_process():
                best_metric = val_metrics_dict["miou"]
                checkpoint = {
                    "state_dict": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": lr_scheduler.state_dict(),
                    "scaler": scaler.state_dict() if scaler is not None and scaler.is_enabled() else None,
                    "epochs_run": epoch,
                    "loss": train_loss_dict["total_loss"]
                }
                save_checkpoint(
                    checkpoint,
                    folder=os.path.join(CFG.RUN_DIR, "logs", "checkpoints"),
                    filename="best_valid_metric.pth"
                )
                print(f"Saved best val metric model.")
