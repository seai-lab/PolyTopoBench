import os
import cv2
import random
import numpy as np
from scipy.optimize import linear_sum_assignment

import torch
import torchvision
from torchmetrics.functional.classification import binary_jaccard_index, binary_accuracy
from config import CFG

try:
    from transformers import top_k_top_p_filtering
except ImportError:
    def top_k_top_p_filtering(
        logits,
        top_k=0,
        top_p=1.0,
        filter_value=-float("inf"),
        min_tokens_to_keep=1,
    ):
        if top_k > 0:
            top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))
            threshold = torch.topk(logits, top_k)[0][..., -1, None]
            indices_to_remove = logits < threshold
            logits = logits.masked_fill(indices_to_remove, filter_value)

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            if min_tokens_to_keep > 1:
                sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
            indices_to_remove.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
            logits = logits.masked_fill(indices_to_remove, filter_value)

        return logits


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def save_checkpoint(state, folder="logs/checkpoint/run1", filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    if not os.path.exists(folder):
        os.makedirs(folder)
    torch.save(state, os.path.join(folder, filename))


def load_checkpoint(checkpoint, model, optimizer, scheduler, scaler=None):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and "scaler" in checkpoint and checkpoint["scaler"] is not None:
        scaler.load_state_dict(checkpoint["scaler"])

    return checkpoint["epochs_run"]


def generate_square_subsequent_mask(sz):
    mask = (
        torch.triu(torch.ones((sz, sz), device=CFG.DEVICE)) == 1
    ).transpose(0, 1)

    mask = mask.float().masked_fill(mask==0, float('-inf')).masked_fill(mask==1, float(0.0))

    return mask


def create_mask(tgt, pad_idx):
    """
    tgt shape: (N, L)
    """

    tgt_seq_len = tgt.size(1)
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    tgt_padding_mask = (tgt == pad_idx)

    return tgt_mask, tgt_padding_mask


class AverageMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0]*3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self) -> str:
        text = f"{self.name}: {self.avg:.4f}"
        return text


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def scores_to_permutations(scores):
    """
    Input a batched array of scores and returns the hungarian optimized 
    permutation matrices.
    """
    B, N, N = scores.shape

    scores = scores.detach().cpu().numpy()
    perm = np.zeros_like(scores)
    for b in range(B):
        r, c = linear_sum_assignment(-scores[b])
        perm[b,r,c] = 1
    return torch.tensor(perm)


# TODO: add permalink to polyworld repo
def permutations_to_polygons(perm, graph, out='torch'):
    B, N, N = perm.shape
    device = perm.device

    def bubble_merge(poly):
        s = 0
        P = len(poly)
        while s < P:
            head = poly[s][-1]

            t = s+1
            while t < P:
                tail = poly[t][0]
                if head == tail:
                    poly[s] = poly[s] + poly[t][1:]
                    del poly[t]
                    poly = bubble_merge(poly)
                    P = len(poly)
                t += 1
            s += 1
        return poly

    diag = torch.logical_not(perm[:,range(N),range(N)])
    batch = []
    for b in range(B):
        b_perm = perm[b]
        b_graph = graph[b]
        b_diag = diag[b]

        idx = torch.arange(N, device=perm.device)[b_diag]

        if idx.shape[0] > 0:
            # If there are vertices in the batch

            b_perm = b_perm[idx,:]
            b_graph = b_graph[idx,:]
            b_perm = b_perm[:,idx]

            first = torch.arange(idx.shape[0]).unsqueeze(1).to(device=device)
            second = torch.argmax(b_perm, dim=1).unsqueeze(1)

            polygons_idx = torch.cat((first, second), dim=1).tolist()
            polygons_idx = bubble_merge(polygons_idx)

            batch_poly = []
            for p_idx in polygons_idx:
                if out == 'torch':
                    batch_poly.append(b_graph[p_idx,:])
                elif out == 'numpy':
                    batch_poly.append(b_graph[p_idx,:].cpu().numpy())
                elif out == 'list':
                    g = b_graph[p_idx,:] * 300 / 320
                    g[:,0] = -g[:,0]
                    g = torch.fliplr(g)
                    batch_poly.append(g.tolist())
                elif out == 'coco':
                    g = b_graph[p_idx,:]# * CFG.IMG_SIZE / CFG.INPUT_WIDTH
                    g = torch.fliplr(g)
                    batch_poly.append(g.view(-1).tolist())
                elif out == 'inria-torch':
                    batch_poly.append(b_graph[p_idx,:])
                else:
                    print("Indicate a valid output polygon format")
                    exit()

            batch.append(batch_poly)

        else:
            # If the batch has no vertices
            batch.append([])

    return batch


def test_generate_fast(
    model,
    x,
    tokenizer,
    max_len=50,
    top_k=0,
    top_p=1,
    amp_dtype=None,
    early_stop=True,
):
    """Fast variant of `test_generate`.

    Optimisations vs. the reference `test_generate`:
      * Encoder is run once; the encoder memory (with encoder pos embeds
        already applied) is cached and reused for every decoding step.
      * At each decoding step the decoder sees only the prefix that has
        actually been generated (length grows from 1 up to max_len). The
        original path always padded the target to MAX_LEN-1, making every
        step cost O(MAX_LEN). We now pay O(step) instead, which is ~>=2x
        speedup on average over the sequence.
      * Early stop when every item in the batch has emitted EOS. For Inria
        tiles the real sequence length is well below max_len, so this is
        typically where most of the win comes from.
      * Optional fp16 autocast for the decoder loop.

    Outputs match `test_generate` in shape/semantics:
      batch_preds  -> (N, L+1) int64 tensor on CPU, with BOS prepended; any
                      remaining positions beyond the early-stop cut-off are
                      filled with PAD_code so postprocess()'s EOS-argmax trick
                      still works.
      confs        -> list of per-"even-step" confidence tensors on CPU.
      perm_preds   -> hungarian-rounded permutation matrix, on CPU.
    """
    from models.model import EncoderDecoder  # local import to avoid cycles

    device = CFG.DEVICE
    x = x.to(device, non_blocking=True)
    N = x.size(0)

    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        core = model.module
    else:
        core = model

    if top_k != 0 or top_p != 1:
        def sample(preds):
            return torch.softmax(preds, dim=-1).multinomial(num_samples=1).view(-1, 1)
    else:
        def sample(preds):
            return torch.argmax(preds, dim=-1, keepdim=True)

    # Autocast context: fp16 on cuda gives a large speedup for this model
    # with negligible accuracy impact for discrete token argmax decoding.
    use_amp = amp_dtype is not None and device.type == "cuda"
    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype)
        if use_amp
        else torch.cuda.amp.autocast(enabled=False)
    )

    with torch.inference_mode():
        with amp_ctx:
            memory = core.encode(x)  # (S, N, D) with encoder pos embed applied

        # Seed with BOS.
        tokens = torch.full(
            (N, 1), tokenizer.BOS_code, dtype=torch.long, device=device
        )
        finished = torch.zeros(N, dtype=torch.bool, device=device)
        confs = []
        for i in range(max_len):
            with amp_ctx:
                logits = core.decode_step(memory, tokens)  # (N, V) fp16 maybe
            logits = logits.float()
            logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
            # Confidence is recorded on even-index steps (matches reference impl).
            if i % 2 == 0:
                confs_ = torch.softmax(logits, dim=-1).sort(axis=-1, descending=True)[0][:, 0].detach().cpu()
                confs.append(confs_)
            nxt = sample(logits)  # (N, 1)
            # Once an item has finished, keep emitting PAD to avoid re-triggering EOS.
            if finished.any():
                nxt = torch.where(
                    finished.unsqueeze(1),
                    torch.full_like(nxt, tokenizer.PAD_code),
                    nxt,
                )
            tokens = torch.cat([tokens, nxt], dim=1)
            just_eos = (nxt.squeeze(1) == tokenizer.EOS_code) & (~finished)
            finished = finished | just_eos
            if early_stop and bool(finished.all().item()):
                break

        # If we early-stopped, right-pad batch_preds to (max_len+1) so that
        # downstream postprocess still finds EOS via argmax on each row.
        total_len = max_len + 1  # matches the reference which returns (N, max_len+1)
        cur_len = tokens.size(1)
        if cur_len < total_len:
            pad = torch.full(
                (N, total_len - cur_len), tokenizer.PAD_code,
                dtype=tokens.dtype, device=tokens.device,
            )
            tokens_full = torch.cat([tokens, pad], dim=1)
        else:
            tokens_full = tokens[:, :total_len]

        # Also make the confs list length match the even-step count of the
        # reference impl. The reference records ceil(max_len/2) entries.
        expected_confs = (max_len + 1) // 2
        while len(confs) < expected_confs:
            confs.append(torch.zeros(N))

        # Final feats/perm computation. Use the fully-padded token layout so
        # the decoder sees the same receptive field as in training (this is
        # what ScoreNet was trained on).
        with amp_ctx:
            feats, perm_probs = core.compute_feats_and_perm(memory, tokens_full)
        perm_preds = scores_to_permutations(perm_probs)

    return tokens_full.cpu(), confs, perm_preds


def test_generate(model, x, tokenizer, max_len=50, top_k=0, top_p=1):
    x = x.to(CFG.DEVICE)
    batch_preds = torch.ones((x.size(0), 1), device=CFG.DEVICE).fill_(tokenizer.BOS_code).long()
    confs = []

    if top_k != 0 or top_p != 1:
        sample = lambda preds: torch.softmax(preds, dim=-1).multinomial(num_samples=1).view(-1, 1)
    else:
        sample = lambda preds: torch.softmax(preds, dim=-1).argmax(dim=-1).view(-1, 1)

    with torch.no_grad():
        for i in range(max_len):
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                preds, feats = model.module.predict(x, batch_preds)
            else:
                preds, feats = model.predict(x, batch_preds)
            preds = top_k_top_p_filtering(preds, top_k=top_k, top_p=top_p)  # if top_k and top_p are set to default, this line does nothing.
            if i % 2 == 0:
                confs_ = torch.softmax(preds, dim=-1).sort(axis=-1, descending=True)[0][:, 0].cpu()
                confs.append(confs_)
            preds = sample(preds)
            batch_preds = torch.cat([batch_preds, preds], dim=1)

        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            # Use the same Sinkhorn+softmax permutation pipeline as training.
            perm_probs = model.module._compute_perm_matrix(feats)
        else:
            perm_probs = model._compute_perm_matrix(feats)

        perm_preds = scores_to_permutations(perm_probs)

    return batch_preds.cpu(), confs, perm_preds


def postprocess(batch_preds, batch_confs, tokenizer):
    EOS_idxs = (batch_preds == tokenizer.EOS_code).float().argmax(dim=-1)
    ## sanity check
    invalid_idxs = ((EOS_idxs - 1) % 2 != 0).nonzero().view(-1)
    EOS_idxs[invalid_idxs] = 0

    all_coords = []
    all_confs = []
    for i, EOS_idx in enumerate(EOS_idxs.tolist()):
        if EOS_idx == 0:
            all_coords.append(None)
            all_confs.append(None)
            continue
        coords = tokenizer.decode(batch_preds[i, :EOS_idx+1])
        confs = [round(batch_confs[j][i].item(), 3) for j in range(len(coords))]

        all_coords.append(coords)
        all_confs.append(confs)

    return all_coords, all_confs


def save_single_predictions_as_images(
    loader, model, tokenizer, epoch, writer, folder="saved_outputs/", device="cuda"
):
    print(f"=> Saving val predictions...")
    if not os.path.exists(folder):
        print(f"==> Creating output subdirectory...")
        os.makedirs(folder)

    model.eval()

    all_coords = []
    all_confs = []

    with torch.no_grad():
        loader_iterator = iter(loader)
        idx = 0
        batch = next(loader_iterator)
        # Training/validation collate may append sample names for debugging.
        # The visualization path only needs the tensors.
        if len(batch) == 6:
            x, y_mask, y_corner_mask, y, y_perm, _sample_names = batch
        else:
            x, y_mask, y_corner_mask, y, y_perm = batch
        batch_preds, batch_confs, perm_preds = test_generate(model, x, tokenizer, max_len=CFG.generation_steps, top_k=0, top_p=1)
        vertex_coords, confs = postprocess(batch_preds, batch_confs, tokenizer)

        all_coords.extend(vertex_coords)
        all_confs.extend(confs)

        coords = []
        for i in range(len(all_coords)):
            if all_coords[i] is not None:
                coord = torch.from_numpy(all_coords[i])
            else:
                coord = torch.tensor([])
            padd = torch.ones((CFG.N_VERTICES - len(coord), 2)).fill_(tokenizer.PAD_code)
            coord = torch.cat((coord, padd), dim=0)
            coords.append(coord)
        batch_polygons = permutations_to_polygons(perm_preds, coords, out='torch')  # list of polygon coordinate tensors

    B, C, H, W = x.shape
    # Write predicted vertices as mask to disk.
    vertex_mask = np.zeros((B, 1, H, W))
    for b in range(len(all_coords)):
        if all_coords[b] is not None:
            print(f"Vertices found!")
            for i in range(len(all_coords[b])):
                coord = all_coords[b][i]
                cx, cy = coord
                cv2.circle(vertex_mask[b, 0], (int(cy), int(cx)), 0, 255, -1)
    vertex_mask = torch.from_numpy(vertex_mask)
    if not os.path.exists(os.path.join(folder, 'corners_mask')):
        os.makedirs(os.path.join(folder, 'corners_mask'))
    vertex_pred_vis = torch.zeros_like(x)
    for b in range(B):
        vertex_pred_vis[b] = torchvision.utils.draw_segmentation_masks(
            (x[b]*255).to(dtype=torch.uint8),
            torch.zeros_like(x[b, 0]).bool()
        )
    vertex_pred_vis = vertex_pred_vis.cpu().numpy().astype(np.uint8)
    for b in range(len(all_coords)):
        if all_coords[b] is not None:
            for i in range(len(all_coords[b])):
                coord = all_coords[b][i]
                cx, cy = coord
                cv2.circle(vertex_pred_vis[b, 0], (int(cy), int(cx)), 3, 255, -1)
    vertex_pred_vis = torch.from_numpy(vertex_pred_vis)
    torchvision.utils.save_image(
        vertex_pred_vis.float()/255, f"{folder}/corners_mask/corners_mask_{b}_{epoch}.png"
    )

    # Write predicted polygons as mask to disk.
    polygons = np.zeros((B, 1, H, W))
    for b in range(B):
        for c in range(len(batch_polygons[b])):
            poly = batch_polygons[b][c]
            poly = poly[poly[:, 0] != tokenizer.PAD_code]
            cnt = np.flip(np.int32(poly.cpu()), 1)
            if len(cnt) > 0:
                cv2.fillPoly(polygons[b, 0], pts=[cnt], color=1.)
    polygons = torch.from_numpy(polygons)
    if not os.path.exists(os.path.join(folder, 'pred_polygons')):
        os.makedirs(os.path.join(folder, 'pred_polygons'))
    poly_out = torch.zeros_like(x)
    for b in range(B):
        poly_out[b] = torchvision.utils.draw_segmentation_masks(
            (x[b]*255).to(dtype=torch.uint8),
            polygons[b, 0].bool()
        )
    poly_out = poly_out.cpu().numpy().astype(np.uint8)
    for b in range(len(all_coords)):
        if all_coords[b] is not None:
            for i in range(len(all_coords[b])):
                coord = all_coords[b][i]
                cx, cy = coord
                cv2.circle(poly_out[b, 0], (int(cy), int(cx)), 2, 255, -1)
    poly_out = torch.from_numpy(poly_out)
    torchvision.utils.save_image(
        poly_out.float()/255, f"{folder}/pred_polygons/polygons_{idx}_{epoch}.png"
    )

    batch_miou = binary_jaccard_index(polygons, y_mask)
    batch_biou = binary_jaccard_index(polygons, y_mask, ignore_index=0)
    batch_macc = binary_accuracy(polygons, y_mask)
    batch_bacc = binary_accuracy(polygons, y_mask, ignore_index=0)

    writer.add_scalar('Val_Metrics/Mean_IoU', batch_miou, epoch)
    writer.add_scalar('Val_Metrics/Building_IoU', batch_biou, epoch)
    writer.add_scalar('Val_Metrics/Mean_Accuracy', batch_macc, epoch)
    writer.add_scalar('Val_Metrics/Building_Accuracy', batch_bacc, epoch)

    metrics_dict = {
        "miou": batch_miou,
        "biou": batch_biou,
        "macc": batch_macc,
        "bacc": batch_bacc
    }

    torchvision.utils.save_image(x, f"{folder}/image_{idx}.png")
    ymask_out = torch.zeros_like(x)
    for b in range(B):
        ymask_out[b] = torchvision.utils.draw_segmentation_masks(
            (x[b]*255).to(dtype=torch.uint8),
            y_mask[b, 0].bool()
        )
    ymask_out = ymask_out.cpu().numpy().astype(np.uint8)
    gt_corner_coords, _ = postprocess(y, batch_confs, tokenizer)
    for b in range(B):
        for corner in gt_corner_coords[b]:
            cx, cy = corner
            cv2.circle(ymask_out[b, 0], (int(cy), int(cx)), 3, 255, -1)
    ymask_out = torch.from_numpy(ymask_out)
    torchvision.utils.save_image(ymask_out/255., f"{folder}/gt_mask_{idx}.png")
    torchvision.utils.save_image(y_corner_mask*255, f"{folder}/gt_corners_{idx}.png")
    torchvision.utils.save_image(y_perm[:, None, :, :]*255, f"{folder}/gt_perm_matrix_{idx}.png")

    return metrics_dict
