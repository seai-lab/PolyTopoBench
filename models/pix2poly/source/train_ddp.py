import os
from os import path as osp
import torch
from torch.cuda.amp import GradScaler
from torch import nn
from torch import optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import (
    get_linear_schedule_with_warmup,
)
from torch.utils.tensorboard import SummaryWriter

from config import CFG
from tokenizer import Tokenizer
from utils import (
    seed_everything,
    load_checkpoint,
)
from ddp_utils import (
    get_inria_loaders,
)

from models.model import (
    Encoder,
    Decoder,
    EncoderDecoder
)

from engine import train_eval

from torch import distributed as dist
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy("file_system")


def init_distributed():

    # Initializes the distributed backend which will take care of synchronizing nodes/GPUs.
    dist_url = "env://"  # default

    # only works with torch.distributed.launch or torch.run.
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(
        backend="nccl",
        init_method=dist_url,
        world_size=world_size,
        rank=rank
    )

    # this will make all .cuda() calls work properly.
    torch.cuda.set_device(local_rank)

    # synchronizes all threads to reach this point before moving on.
    dist.barrier()


def main():
    # setup the process groups
    init_distributed()
    seed_everything(42)
    torch.backends.cuda.matmul.allow_tf32 = CFG.ALLOW_TF32
    torch.backends.cudnn.allow_tf32 = CFG.ALLOW_TF32

    # Define tensorboard for logging.
    writer = SummaryWriter(os.path.join(CFG.RUN_DIR, "logs", "tensorboard"))
    attrs = vars(CFG)
    os.makedirs(CFG.RUN_DIR, exist_ok=True)
    with open(os.path.join(CFG.RUN_DIR, "config.txt"), "w") as f:
        print("\n".join("%s: %s" % item for item in attrs.items()), file=f)

    train_transform_list = []
    if CFG.AFFINE_P > 0:
        train_transform_list.append(
            A.Affine(
                rotate=[-CFG.AFFINE_MAX_ROT_DEG, CFG.AFFINE_MAX_ROT_DEG],
                fit_output=True,
                p=CFG.AFFINE_P,
            )
        )
    train_transform_list.append(A.Resize(height=CFG.INPUT_HEIGHT, width=CFG.INPUT_WIDTH))
    if CFG.RANDOM_ROTATE90_P > 0:
        train_transform_list.append(A.RandomRotate90(p=CFG.RANDOM_ROTATE90_P))
    if CFG.BRIGHTNESS_CONTRAST_P > 0:
        train_transform_list.append(A.RandomBrightnessContrast(p=CFG.BRIGHTNESS_CONTRAST_P))
    if CFG.COLOR_JITTER_P > 0:
        train_transform_list.append(A.ColorJitter(p=CFG.COLOR_JITTER_P))
    if CFG.TO_GRAY_P > 0:
        train_transform_list.append(A.ToGray(p=CFG.TO_GRAY_P))
    if CFG.GAUSS_NOISE_P > 0:
        train_transform_list.append(A.GaussNoise(p=CFG.GAUSS_NOISE_P))
    train_transform_list.extend(
        [
            # ToTensorV2 of albumentations doesn't divide by 255 like in PyTorch,
            # it is done inside Normalize function.
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ]
    )
    train_transforms = A.Compose(
        train_transform_list,
        keypoint_params=A.KeypointParams(format='yx', remove_invisible=False)
    )

    valid_transforms = A.Compose(
        [
            A.Resize(height=CFG.INPUT_HEIGHT, width=CFG.INPUT_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(format='yx', remove_invisible=False)
    )

    if "debug" in CFG.EXPERIMENT_NAME:
        train_transforms = valid_transforms

    tokenizer = Tokenizer(
        num_classes=1,
        num_bins=CFG.NUM_BINS,
        width=CFG.INPUT_WIDTH,
        height=CFG.INPUT_HEIGHT,
        max_len=CFG.MAX_LEN
    )
    CFG.PAD_IDX = tokenizer.PAD_code

    train_loader, val_loader, _ = get_inria_loaders(
        CFG.TRAIN_DATASET_DIR,
        CFG.VAL_DATASET_DIR,
        CFG.TEST_IMAGES_DIR,
        tokenizer,
        CFG.MAX_LEN,
        tokenizer.PAD_code,
        CFG.SHUFFLE_TOKENS,
        CFG.BATCH_SIZE,
        train_transforms,
        valid_transforms,
        CFG.NUM_WORKERS,
        CFG.PIN_MEMORY,
    )

    encoder = Encoder(model_name=CFG.MODEL_NAME, pretrained=CFG.PRETRAINED_ENCODER, out_dim=256)
    decoder = Decoder(
        cfg=CFG,
        vocab_size=tokenizer.vocab_size,
        encoder_len=CFG.NUM_PATCHES,
        dim=256,
        num_heads=8,
        num_layers=6
    )
    model = EncoderDecoder(cfg=CFG, encoder=encoder, decoder=decoder)
    model.to(CFG.DEVICE)

    weight = torch.ones(CFG.PAD_IDX + 1, device=CFG.DEVICE)
    weight[tokenizer.num_bins:tokenizer.BOS_code] = 0.0
    vertex_loss_fn = nn.CrossEntropyLoss(ignore_index=CFG.PAD_IDX, label_smoothing=CFG.LABEL_SMOOTHING, weight=weight)
    perm_loss_fn = nn.BCELoss()

    optimizer = optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY, betas=(0.9, 0.95))
    scaler = GradScaler(enabled=CFG.USE_AMP and CFG.DEVICE.type == "cuda")

    num_training_steps = CFG.NUM_EPOCHS * (len(train_loader.dataset) // CFG.BATCH_SIZE // torch.cuda.device_count())
    num_warmup_steps = int(0.05 * num_training_steps)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_training_steps=num_training_steps,
        num_warmup_steps=num_warmup_steps
    )

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    CFG.START_EPOCH = 0
    if CFG.LOAD_MODEL:
        checkpoint_name = osp.basename(osp.realpath(CFG.CHECKPOINT_PATH))
        map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
        ckpt_path = os.path.join(CFG.RUN_DIR, "logs", "checkpoints", checkpoint_name)
        if os.environ.get("PIX2POLY_WARM_INIT", "0") == "1":
            # Weights-only warm-start: load model state_dict from a legacy ckpt but
            # keep the freshly-built optimizer/scheduler/scaler so LR/schedule reflect
            # the current fine-tune config. START_EPOCH stays 0 so the full NUM_EPOCHS
            # budget is iterated.
            ck = torch.load(ckpt_path, map_location=map_location)
            sd = ck["state_dict"] if "state_dict" in ck else ck
            missing, unexpected = model.load_state_dict(sd, strict=False)
            if local_rank == 0:
                print(f"[warm-init] loaded weights from {ckpt_path}; missing={len(missing)} unexpected={len(unexpected)}")
        else:
            start_epoch = load_checkpoint(
                torch.load(ckpt_path, map_location=map_location),
                model,
                optimizer,
                lr_scheduler,
                scaler=scaler
            )
            CFG.START_EPOCH = start_epoch + 1
        dist.barrier()

    # SyncBatchNorm is only useful when batch statistics are synchronized across
    # multiple ranks. On single-GPU runs it adds an unnecessary numerical
    # difference, and the ScoreNet branch was observed to become unstable there.
    if world_size > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # Wrap model with distributed data parallel.
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    train_eval(
        model,
        train_loader,
        val_loader,
        val_loader,
        tokenizer,
        vertex_loss_fn,
        perm_loss_fn,
        optimizer,
        lr_scheduler=lr_scheduler,
        step='batch',
        writer=writer,
        scaler=scaler
    )


if __name__ == "__main__":
    main()
