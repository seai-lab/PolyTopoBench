import timm
import torch
from torch import nn
from torch.nn import functional as F
from timm.models.layers import trunc_normal_

import os
import sys
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config import CFG
from utils import (
    create_mask,
)


# Borrowed from https://github.com/magicleap/SuperGluePretrainedNetwork/blob/ddcf11f42e7e0732a0c4607648f9448ea8d73590/models/superglue.py#L143
def log_sinkhorn_iterations(Z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)

# Borrowed from https://github.com/magicleap/SuperGluePretrainedNetwork/blob/ddcf11f42e7e0732a0c4607648f9448ea8d73590/models/superglue.py#L152
def log_optimal_transport(scores: torch.Tensor, alpha: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z


class ScoreNet(nn.Module):
    def __init__(self, n_vertices, in_channels=512):
        super().__init__()
        self.n_vertices = n_vertices
        self.in_channels = in_channels
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=1, stride=1, padding=0, bias=True)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=True)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0, bias=True)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, feats):
        # `feats` comes from the decoder and has shape [B, T, D], where:
        # - T = MAX_LEN - 1 because the decoder consumes `y_input`
        # - D = decoder hidden size
        #
        # Token layout is:
        #   BOS, y1, x1, y2, x2, ..., EOS/PAD
        # so after dropping BOS, coordinate tokens appear in y/x pairs.
        feats = feats[:, 1:]
        feats = feats.unsqueeze(2)
        # Group every two token embeddings into one vertex embedding.
        # After the reshape, axis 2 has size 2 and corresponds to (y_token, x_token).
        feats = feats.view(feats.size(0), feats.size(1)//2, 2, feats.size(3))
        # Collapse the y/x token pair into one vertex feature.
        feats = torch.mean(feats, dim=2)

        # Build pairwise features for all vertex pairs (i, j).
        # Resulting tensor is shaped so that a 1x1 conv stack can score whether
        # vertex i should connect to vertex j in the permutation matrix.
        x = torch.transpose(feats, 1, 2)
        x = x.unsqueeze(-1)
        x = x.repeat(1, 1, 1, self.n_vertices)
        t = torch.transpose(x, 2, 3)
        x = torch.cat((x, t), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)

        return x[:, 0]


class Encoder(nn.Module):
    def __init__(self, model_name='deit3_small_patch16_384_in21ft1k', pretrained=False, out_dim=256) -> None:
        super().__init__()
        self.model = timm.create_model(
            model_name=model_name,
            img_size=CFG.INPUT_SIZE,
            num_classes=0,
            global_pool='',
            pretrained=pretrained
        )
        self.bottleneck = nn.AdaptiveAvgPool1d(out_dim)

    def forward(self, x):
        # timm ViT returns [B, 1 + num_patches, C] with a leading CLS token.
        features = self.model(x)
        # Pix2Poly discards CLS and keeps only patch tokens as encoder memory.
        # The adaptive pool reduces the channel dimension to the decoder width.
        return self.bottleneck(features[:, 1:])


class Decoder(nn.Module):
    def __init__(self, cfg, vocab_size, encoder_len, dim, num_heads, num_layers):
        super().__init__()
        self.cfg = cfg
        self.dim = dim

        self.embedding = nn.Embedding(vocab_size, dim)
        self.decoder_pos_embed = nn.Parameter(torch.randn(1, self.cfg.MAX_LEN-1, dim) * .02)
        self.decoder_pos_drop = nn.Dropout(p=0.05)

        decoder_layer = nn.TransformerDecoderLayer(d_model=dim, nhead=num_heads)
        self.decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=num_layers)
        self.output = nn.Linear(dim, vocab_size)

        self.encoder_pos_embed = nn.Parameter(torch.randn(1, encoder_len, dim) * .02)
        self.encoder_pos_drop = nn.Dropout(p=0.05)

        self.init_weights()

    def init_weights(self):
        for name, p in self.named_parameters():
            if 'encoder_pos_embed' in name or 'decoder_pos_embed' in name:
                print(f"Skipping initialization of pos embed layers...")
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        trunc_normal_(self.encoder_pos_embed, std=.02)
        trunc_normal_(self.decoder_pos_embed, std=.02)

    def forward(self, encoder_out, tgt):
        """
        encoder_out shape: (N, L, D)
        tgt shape: (N, L)

        Decoder.forward(...) 是训练态的 teacher-forced 路径。
        这里的 tgt 来自 GT 右移，不是模型自己生成的 token。

        真正推理时走的是 model.py 里的 Decoder.predict(...)：

        一开始只给 BOS
        然后一步一步把自己预测出来的 token 再喂回去

        """
        # `tgt` is teacher-forced input tokens:
        #   [BOS, y1, x1, y2, x2, ...]
        # shifted right from the GT sequence during training.
        tgt_mask, tgt_padding_mask = create_mask(tgt, self.cfg.PAD_IDX) 
        tgt_embedding = self.embedding(tgt)
        tgt_embedding = self.decoder_pos_drop(
            tgt_embedding + self.decoder_pos_embed
        )

        encoder_out = self.encoder_pos_drop(
            encoder_out + self.encoder_pos_embed
        )

        encoder_out = encoder_out.transpose(0, 1)
        tgt_embedding = tgt_embedding.transpose(0, 1)

        preds = self.decoder(
            memory=encoder_out,
            tgt=tgt_embedding,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )

        preds = preds.transpose(0, 1)
        # Returns:
        # - token logits for next-token prediction
        # - decoder hidden states, which are reused by the permutation branch
        return self.output(preds), preds  

    def predict(self, encoder_out, tgt):
        # Autoregressive inference path:
        # `tgt` contains only tokens generated so far. The remainder is padded so
        # the decoder sees the same sequence length as in training.
        length = tgt.size(1)
        padding = torch.ones((tgt.size(0), self.cfg.MAX_LEN-length-1), device=tgt.device).fill_(self.cfg.PAD_IDX).long()
        tgt = torch.cat([tgt, padding], dim=1)
        tgt_mask, tgt_padding_mask = create_mask(tgt, self.cfg.PAD_IDX)
        tgt_embedding = self.embedding(tgt)
        tgt_embedding = self.decoder_pos_drop(
            tgt_embedding + self.decoder_pos_embed
        )

        encoder_out = self.encoder_pos_drop(
            encoder_out + self.encoder_pos_embed
        )

        encoder_out = encoder_out.transpose(0, 1)
        tgt_embedding = tgt_embedding.transpose(0, 1)

        preds = self.decoder(
            memory=encoder_out,
            tgt=tgt_embedding,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask = tgt_padding_mask
        )

        preds = preds.transpose(0, 1)
        # Only the next-token logits are needed for decoding, but the full hidden
        # sequence is returned because the final hidden states are later fed into
        # ScoreNet to predict vertex connectivity.
        return self.output(preds)[:, length-1, :], preds

    # ---------------------------------------------------------------------
    # Fast inference helpers (added for A10G throughput optimisation).
    #
    # The original `predict` path pads `tgt` to MAX_LEN-1 every step and
    # re-adds encoder pos embeddings every step. For MAX_LEN-1 == 385 this
    # makes each step ~O(385) in decoder self-attention even though only
    # the last row of the output is actually consumed. The helpers below:
    #   1. Apply the encoder pos-embed once (outside the decoding loop).
    #   2. At each step, feed the decoder only the prefix of length `L`
    #      that has actually been generated (so attention cost is O(L),
    #      not O(MAX_LEN)). The causal mask is a plain triangular mask
    #      of size (L, L); no padding mask is required because every
    #      token in `tgt` has been emitted.
    # The semantics are equivalent to `predict` for the returned next-token
    # logits, because the original path's future positions were fully
    # masked by both the causal mask and the PAD-based padding mask.
    # ---------------------------------------------------------------------
    def encode_memory(self, encoder_out):
        """Apply encoder pos-embed/dropout once and return [S, N, D] memory."""
        encoder_out = self.encoder_pos_drop(encoder_out + self.encoder_pos_embed)
        return encoder_out.transpose(0, 1).contiguous()

    def step(self, memory, tgt):
        """One autoregressive step. `memory` is precomputed via encode_memory.

        tgt: (N, L) LongTensor of tokens generated so far (no padding).
        Returns next-token logits of shape (N, V).
        """
        L = tgt.size(1)
        # Embed + slice the corresponding pos embeddings (pos 0..L-1).
        tgt_emb = self.embedding(tgt) + self.decoder_pos_embed[:, :L]
        # decoder_pos_drop is disabled at eval time (dropout is a no-op), but
        # we keep it for consistency with training/eval parity.
        tgt_emb = self.decoder_pos_drop(tgt_emb).transpose(0, 1).contiguous()
        # Causal mask of size (L, L); no padding mask since all tokens are real.
        tgt_mask = torch.triu(
            torch.full((L, L), float("-inf"), device=tgt.device, dtype=tgt_emb.dtype),
            diagonal=1,
        )
        out = self.decoder(
            memory=memory,
            tgt=tgt_emb,
            tgt_mask=tgt_mask,
        )
        # Only need the logit at the last time-step.
        last = out[-1]  # (N, D)
        return self.output(last)

    def forward_feats_padded(self, memory, tokens):
        """Run the decoder on a right-padded sequence of shape MAX_LEN-1 and
        return the feature tensor used by ScoreNet. Behaves exactly like the
        original `Decoder.predict` path for the feature output.
        """
        target_len = self.cfg.MAX_LEN - 1
        N, L = tokens.shape
        if L < target_len:
            pad = torch.full(
                (N, target_len - L), self.cfg.PAD_IDX,
                device=tokens.device, dtype=tokens.dtype,
            )
            tokens = torch.cat([tokens, pad], dim=1)
        elif L > target_len:
            tokens = tokens[:, :target_len]
        tgt_mask, tgt_padding_mask = create_mask(tokens, self.cfg.PAD_IDX)
        tgt_emb = self.embedding(tokens) + self.decoder_pos_embed
        tgt_emb = self.decoder_pos_drop(tgt_emb).transpose(0, 1).contiguous()
        preds = self.decoder(
            memory=memory,
            tgt=tgt_emb,
            tgt_mask=tgt_mask.to(tgt_emb.dtype),
            tgt_key_padding_mask=tgt_padding_mask,
        )
        return preds.transpose(0, 1)


class EncoderDecoder(nn.Module):
    def __init__(self, cfg, encoder, decoder):
        super().__init__()
        self.cfg = cfg
        self.encoder = encoder
        self.decoder = decoder
        self.scorenet1 = ScoreNet(self.cfg.N_VERTICES)
        self.scorenet2 = ScoreNet(self.cfg.N_VERTICES)
        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)

    def _compute_perm_matrix(self, feats):
        # The permutation branch is more numerically fragile than the token
        # branch because it stacks pairwise ScoreNet convs, Sinkhorn iterations,
        # and a softmax over dense vertex-to-vertex scores. Keep it in float32
        # even when the outer training step uses AMP.
        autocast_kwargs = {"device_type": feats.device.type, "enabled": False}
        with torch.autocast(**autocast_kwargs):
            feats = feats.float()
            bin_score = self.bin_score.float()

            perm_mat1 = self.scorenet1(feats)
            perm_mat2 = self.scorenet2(feats)
            perm_mat = perm_mat1 + torch.transpose(perm_mat2, 1, 2)
            perm_mat = log_optimal_transport(
                perm_mat,
                bin_score,
                self.cfg.SINKHORN_ITERATIONS
            )[:, :perm_mat.shape[1], :perm_mat.shape[2]]
            return F.softmax(perm_mat, dim=-1)

    def forward(self, image, tgt):
        # Full training / teacher-forced path.
        encoder_out = self.encoder(image)
        preds, feats = self.decoder(encoder_out, tgt)

        # Keep the permutation branch in float32 for stability; AMP still
        # accelerates the encoder/decoder path above.
        perm_mat = self._compute_perm_matrix(feats)

        return preds, perm_mat

    def predict(self, image, tgt):
        # Pure inference path used during autoregressive decoding.
        # This does not consume GT tokens; it only consumes the already-generated prefix.
        encoder_out = self.encoder(image)
        preds, feats = self.decoder.predict(encoder_out, tgt)
        return preds, feats

    # --- Fast inference helpers (encoder-once, short-tgt decode). ---
    def encode(self, image):
        encoder_out = self.encoder(image)
        memory = self.decoder.encode_memory(encoder_out)
        return memory

    def decode_step(self, memory, tgt):
        return self.decoder.step(memory, tgt)

    def compute_feats_and_perm(self, memory, tokens):
        """Given the cached memory and the full generated token sequence,
        recompute decoder features exactly as the original path would, then
        run the permutation branch. Returns (feats, perm_probs)."""
        feats = self.decoder.forward_feats_padded(memory, tokens)
        perm_probs = self._compute_perm_matrix(feats)
        return feats, perm_probs


if __name__ == "__main__":
    import argparse
    import random

    import numpy as np
    import torch

    from tokenizer import Tokenizer
    from utils import permutations_to_polygons, scores_to_permutations

    def parse_args():
        parser = argparse.ArgumentParser(
            description="Run a no-training forward smoke test for Pix2Poly."
        )
        parser.add_argument(
            "--dataset-dir",
            default="",
            help=(
                "Optional Pix2Poly-format dataset split directory containing images/ "
                "and annotation.json. If omitted, a synthetic sample is used."
            ),
        )
        parser.add_argument(
            "--sample-index",
            type=int,
            default=0,
            help="Dataset sample index to load when --dataset-dir is provided.",
        )
        parser.add_argument(
            "--device",
            choices=["auto", "cpu", "cuda"],
            default="auto",
            help="Device to use for the smoke test.",
        )
        parser.add_argument(
            "--predict-steps",
            type=int,
            default=16,
            help="Number of autoregressive decoding steps to run.",
        )
        parser.add_argument(
            "--pretrained-encoder",
            action="store_true",
            help="Load pretrained encoder weights. Disabled by default to avoid downloads.",
        )
        parser.add_argument("--seed", type=int, default=42, help="Random seed.")
        return parser.parse_args()

    def resolve_device(device_arg):
        if device_arg == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device_arg == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device(device_arg)

    def seed_everything(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def build_perm_matrix(rings, n_vertices):
        # Helper used only by the smoke test.
        # It builds the same GT connectivity target as the dataset:
        # each vertex points to the next vertex in its ring, and padding vertices
        # get a self-loop on the diagonal.
        perm = torch.zeros((n_vertices, n_vertices), dtype=torch.float32)
        v_count = 0
        for ring in rings:
            for i in range(len(ring)):
                j = (i + 1) % len(ring)
                perm[v_count + i, v_count + j] = 1.0
            v_count += len(ring)

        for i in range(v_count, n_vertices):
            perm[i, i] = 1.0

        for i in range(n_vertices):
            row_sum = perm[i, :].sum()
            col_sum = perm[:, i].sum()
            if row_sum == 0 or col_sum == 0:
                perm[i, i] = 1.0

        return perm

    def build_synthetic_sample():
        image = np.zeros((CFG.INPUT_HEIGHT, CFG.INPUT_WIDTH, 3), dtype=np.uint8)
        image[20:96, 28:104, 0] = 255
        image[118:196, 120:198, 1] = 255

        rings = [
            np.array([[20, 28], [20, 104], [96, 104], [96, 28]], dtype=np.float32),
            np.array([[118, 120], [118, 198], [196, 198], [196, 120]], dtype=np.float32),
        ]
        coords = np.concatenate(rings, axis=0)
        perm_matrix = build_perm_matrix(rings, CFG.N_VERTICES)
        sample_desc = "synthetic two-building sample"
        return image, coords, perm_matrix, sample_desc

    def load_dataset_sample(dataset_dir, sample_index):
        from datasets.dataset_inria_coco import InriaCocoDataset

        dataset = InriaCocoDataset(
            dataset_dir=dataset_dir,
            transform=None,
            tokenizer=None,
            shuffle_tokens=False,
        )
        if sample_index < 0 or sample_index >= len(dataset):
            raise IndexError(
                f"sample-index {sample_index} out of range for dataset of size {len(dataset)}"
            )
        image, _, _, coords, perm_matrix, _ = dataset[sample_index]
        image_id = dataset.image_ids[sample_index]
        file_name = dataset.coco.loadImgs(image_id)[0]["file_name"]
        sample_desc = f"dataset sample index={sample_index}, image_id={image_id}, file_name={file_name}"
        return image, coords, perm_matrix.float(), sample_desc

    def prepare_example(args):
        if args.dataset_dir:
            image, coords, perm_matrix, sample_desc = load_dataset_sample(
                args.dataset_dir, args.sample_index
            )
        else:
            image, coords, perm_matrix, sample_desc = build_synthetic_sample()

        image_tensor = (
            torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        )
        coords = np.asarray(coords, dtype=np.float32)
        return image_tensor, coords, perm_matrix, sample_desc

    def pad_token_sequence(tokens, pad_idx, max_len):
        seq = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
        if seq.size(1) > max_len:
            raise ValueError(
                f"Token sequence length {seq.size(1)} exceeds configured max_len {max_len}"
            )
        if seq.size(1) < max_len:
            pad = torch.full((1, max_len - seq.size(1)), pad_idx, dtype=torch.long)
            seq = torch.cat([seq, pad], dim=1)
        return seq

    args = parse_args()
    device = resolve_device(args.device)
    CFG.DEVICE = device
    seed_everything(args.seed)

    tokenizer = Tokenizer(
        num_classes=1,
        num_bins=CFG.NUM_BINS,
        width=CFG.INPUT_WIDTH,
        height=CFG.INPUT_HEIGHT,
        max_len=CFG.MAX_LEN,
    )
    CFG.PAD_IDX = tokenizer.PAD_code

    image, coords, gt_perm, sample_desc = prepare_example(args)
    tokenized, _ = tokenizer(coords.copy(), shuffle=False)
    token_sequence = pad_token_sequence(tokenized, tokenizer.PAD_code, CFG.MAX_LEN)
    # Teacher forcing uses a right-shifted GT sequence:
    # - y_input: what the decoder sees
    # - y_expected: what the model should predict next
    y_input = token_sequence[:, :-1].to(device)
    y_expected = token_sequence[:, 1:].to(device)

    image = image.to(device)
    gt_perm = gt_perm.unsqueeze(0).to(device)

    encoder = Encoder(
        model_name=CFG.MODEL_NAME,
        pretrained=args.pretrained_encoder,
        out_dim=256,
    )
    decoder = Decoder(
        cfg=CFG,
        vocab_size=tokenizer.vocab_size,
        encoder_len=CFG.NUM_PATCHES,
        dim=256,
        num_heads=8,
        num_layers=6,
    )
    model = EncoderDecoder(cfg=CFG, encoder=encoder, decoder=decoder).to(device)
    model.eval()

    print("== Pix2Poly Smoke Test ==")
    print(f"sample: {sample_desc}")
    print(f"device: {device}")
    print(f"image shape: {tuple(image.shape)}")
    print(f"coords shape: {coords.shape}")
    print(f"coords head (yx): {coords[:min(8, len(coords))].tolist()}")
    print(f"tokenized length (without pad): {len(tokenized)}")
    print(f"tokenized head: {tokenized[:min(24, len(tokenized))]}")
    print(f"gt permutation shape: {tuple(gt_perm.shape)}")
    print(
        f"gt permutation off-diagonal edges: "
        f"{int((gt_perm.sum() - torch.diagonal(gt_perm[0]).sum()).item())}"
    )

    with torch.inference_mode():
        # Teacher-forced forward:
        # decoder_feats here are conditioned on GT history from y_input.
        # This mirrors the training path and is not the same as pure inference.
        encoder_out = model.encoder(image)
        decoder_logits, decoder_feats = model.decoder(encoder_out, y_input)
        perm_scores_1 = model.scorenet1(decoder_feats)
        perm_scores_2 = model.scorenet2(decoder_feats)
        perm_logits = perm_scores_1 + torch.transpose(perm_scores_2, 1, 2)
        perm_transport = log_optimal_transport(
            perm_logits, model.bin_score, CFG.SINKHORN_ITERATIONS
        )[:, :perm_logits.shape[1], :perm_logits.shape[2]]
        perm_probs = F.softmax(perm_transport, dim=-1)

        forward_logits, forward_perm = model(image, y_input)

    print("\n== Teacher-Forced Forward ==")
    print(f"encoder_out shape: {tuple(encoder_out.shape)}")
    print(f"decoder_logits shape: {tuple(decoder_logits.shape)}")
    print(f"decoder_feats shape: {tuple(decoder_feats.shape)}")
    print(f"perm_logits shape: {tuple(perm_logits.shape)}")
    print(f"perm_probs shape: {tuple(perm_probs.shape)}")
    print(
        f"model.forward logits match manual path: "
        f"{torch.allclose(forward_logits, decoder_logits)}"
    )
    print(
        f"model.forward perm match manual path: "
        f"{torch.allclose(forward_perm, perm_probs)}"
    )
    print(
        f"teacher-forced argmax head: "
        f"{forward_logits.argmax(dim=-1)[0, :16].detach().cpu().tolist()}"
    )
    print(
        f"expected token head: "
        f"{y_expected[0, :16].detach().cpu().tolist()}"
    )

    # Recover polygons directly from the GT permutation matrix to show what
    # `gt_perm` means geometrically: it is the connectivity pattern between
    # vertices, not another token sequence.
    padded_coords = torch.full(
        (1, CFG.N_VERTICES, 2), tokenizer.PAD_code, dtype=torch.float32, device=device
    )
    if len(coords) > 0:
        padded_coords[0, : len(coords)] = torch.from_numpy(coords).to(device)
    gt_polygons = permutations_to_polygons(
        gt_perm, padded_coords, out="torch"
    )
    print(f"gt polygons recovered from gt perm: {len(gt_polygons[0])}")
    print(
        f"gt polygon vertex counts: {[len(poly) for poly in gt_polygons[0][:8]]}"
    )

    print("\n== Autoregressive Decode ==")
    # This is the true inference path. The model starts from BOS and repeatedly
    # predicts the next token from its own previous predictions.
    generated = torch.full(
        (1, 1), tokenizer.BOS_code, dtype=torch.long, device=device
    )
    step_tokens = [tokenizer.BOS_code]
    final_feats = None
    for step in range(args.predict_steps):
        with torch.inference_mode():
            step_logits, final_feats = model.predict(image, generated)
        next_token = torch.argmax(step_logits, dim=-1, keepdim=True)
        next_token_int = int(next_token.item())
        step_tokens.append(next_token_int)
        generated = torch.cat([generated, next_token], dim=1)
        print(
            f"step={step:02d} next_token={next_token_int} "
            f"top_logit={float(step_logits.max().item()):.4f}"
        )
        if next_token_int == tokenizer.EOS_code:
            break

    print(f"generated tokens: {step_tokens}")

    valid_decode = (
        step_tokens[-1] == tokenizer.EOS_code
        and ((len(step_tokens) - 2) % 2 == 0)
    )
    if valid_decode:
        decoded_coords = tokenizer.decode(torch.tensor(step_tokens, dtype=torch.long))
        print(f"decoded coords shape: {decoded_coords.shape}")
        print(f"decoded coords head (yx): {decoded_coords[:min(8, len(decoded_coords))].tolist()}")
    else:
        print(
            "generated sequence is not decodable yet. "
            "This is expected for an untrained/randomly initialized model."
        )

    if final_feats is not None:
        with torch.inference_mode():
            # At inference time, the permutation branch is driven by decoder features
            # produced from the generated token prefix, not from GT tokens.
            pred_perm_scores = model.scorenet1(final_feats) + torch.transpose(
                model.scorenet2(final_feats), 1, 2
            )
            pred_perm_hard = scores_to_permutations(pred_perm_scores).to(device)
        print(f"hard predicted permutation shape: {tuple(pred_perm_hard.shape)}")
        print(
            f"hard predicted off-diagonal edges: "
            f"{int((pred_perm_hard.sum() - torch.diagonal(pred_perm_hard[0]).sum()).item())}"
        )
