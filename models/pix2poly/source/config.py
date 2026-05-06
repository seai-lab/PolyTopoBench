import os
import torch


def _get_bool_env(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _get_float_env(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return float(value)


def _get_int_env(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return int(value)


class CFG:
    IMG_PATH = ''
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    DATASET = os.environ.get("PIX2POLY_DATASET", "inria_building_building")
    DATA_ROOT = os.environ.get("PIX2POLY_DATA_ROOT", "./data")
    TRAIN_DATASET_DIR = os.environ.get("PIX2POLY_TRAIN_DATASET_DIR", f"{DATA_ROOT}/{DATASET}/train")
    VAL_DATASET_DIR = os.environ.get("PIX2POLY_VAL_DATASET_DIR", f"{DATA_ROOT}/{DATASET}/val")
    TEST_IMAGES_DIR = os.environ.get("PIX2POLY_TEST_IMAGES_DIR", f"{DATA_ROOT}/{DATASET}/val/images")


    TRAIN_DDP = _get_bool_env("PIX2POLY_TRAIN_DDP", True)
    NUM_WORKERS = int(os.environ.get("PIX2POLY_NUM_WORKERS", 2))
    PIN_MEMORY = _get_bool_env("PIX2POLY_PIN_MEMORY", True)
    LOAD_MODEL = _get_bool_env("PIX2POLY_LOAD_MODEL", False)
    PRETRAINED_ENCODER = _get_bool_env("PIX2POLY_PRETRAINED_ENCODER", False)
    USE_AMP = _get_bool_env("PIX2POLY_USE_AMP", True)
    ALLOW_TF32 = _get_bool_env("PIX2POLY_ALLOW_TF32", True)

    N_VERTICES = _get_int_env("PIX2POLY_N_VERTICES", 224)

    SINKHORN_ITERATIONS = 100
    MAX_LEN = (N_VERTICES*2) + 2
    IMG_SIZE = _get_int_env("PIX2POLY_IMG_SIZE", 512)
    INPUT_SIZE = _get_int_env("PIX2POLY_INPUT_SIZE", IMG_SIZE)
    PATCH_SIZE = 8
    INPUT_HEIGHT = INPUT_SIZE
    INPUT_WIDTH = INPUT_SIZE
    NUM_BINS = INPUT_HEIGHT*1
    LABEL_SMOOTHING = 0.0
    vertex_loss_weight = 1.0
    perm_loss_weight = 10.0
    SHUFFLE_TOKENS = False  # order gt vertex tokens randomly every time

    BATCH_SIZE = int(os.environ.get("PIX2POLY_BATCH_SIZE", 28))  # batch size per gpu; effective batch size = BATCH_SIZE * NUM_GPUs
    START_EPOCH = 0
    NUM_EPOCHS = int(os.environ.get("PIX2POLY_NUM_EPOCHS", 200))
    MILESTONE = int(os.environ.get("PIX2POLY_MILESTONE", 0))
    SAVE_BEST = _get_bool_env("PIX2POLY_SAVE_BEST", True)
    SAVE_LATEST = _get_bool_env("PIX2POLY_SAVE_LATEST", True)
    SAVE_EVERY = int(os.environ.get("PIX2POLY_SAVE_EVERY", 10))
    VAL_EVERY = int(os.environ.get("PIX2POLY_VAL_EVERY", 5))
    DEBUG_MAX_TRAIN_STEPS = int(os.environ.get("PIX2POLY_DEBUG_MAX_TRAIN_STEPS", 0))
    DEBUG_MAX_VAL_STEPS = int(os.environ.get("PIX2POLY_DEBUG_MAX_VAL_STEPS", 0))

    # timm expects a canonical model id (for example `vit_small_patch8_224`)
    # while the runtime input size is passed separately when the encoder is built.
    MODEL_NAME = os.environ.get("PIX2POLY_MODEL_NAME", f'vit_small_patch{PATCH_SIZE}_224')
    NUM_PATCHES = int((INPUT_SIZE // PATCH_SIZE) ** 2)

    # Optimizer / training stability.
    LR = _get_float_env("PIX2POLY_LR", 2e-4)
    WEIGHT_DECAY = 1e-4
    MAX_GRAD_NORM = float(os.environ.get("PIX2POLY_MAX_GRAD_NORM", 0.1))

    # Reported setting: 512 input, 224 maximum vertices, no affine rotation.
    AFFINE_P = _get_float_env("PIX2POLY_AFFINE_P", 0.0)
    AFFINE_MAX_ROT_DEG = _get_float_env("PIX2POLY_AFFINE_MAX_ROT_DEG", 30.0)
    RANDOM_ROTATE90_P = _get_float_env("PIX2POLY_RANDOM_ROTATE90_P", 0.0)
    BRIGHTNESS_CONTRAST_P = _get_float_env("PIX2POLY_BRIGHTNESS_CONTRAST_P", 0.2)
    COLOR_JITTER_P = _get_float_env("PIX2POLY_COLOR_JITTER_P", 0.15)
    TO_GRAY_P = _get_float_env("PIX2POLY_TO_GRAY_P", 0.05)
    GAUSS_NOISE_P = _get_float_env("PIX2POLY_GAUSS_NOISE_P", 0.05)

    generation_steps = (N_VERTICES * 2) + 1  # sequence length during prediction. Should not be more than max_len
    run_eval = False

    EXPERIMENT_NAME = os.environ.get(
        "PIX2POLY_EXPERIMENT_NAME",
        f"train_Pix2Poly_{DATASET}_run1_{MODEL_NAME}_NoAffineRot_LinearWarmupLRS_{vertex_loss_weight}xVertexLoss_{perm_loss_weight}xPermLoss__2xScoreNet_initialLR_{LR}_bs_{BATCH_SIZE}_Nv_{N_VERTICES}_Nbins{NUM_BINS}_{NUM_EPOCHS}epochs"
    )
    RUNS_DIR = os.environ.get("PIX2POLY_RUNS_DIR", "runs")
    RUN_DIR = os.path.join(RUNS_DIR, EXPERIMENT_NAME)

    if "debug" in EXPERIMENT_NAME:
        BATCH_SIZE = 10
        NUM_WORKERS = 0
        SAVE_BEST = False
        SAVE_LATEST = False
        SAVE_EVERY = NUM_EPOCHS
        VAL_EVERY = 50

    if LOAD_MODEL:
        CHECKPOINT_PATH = os.path.join(RUN_DIR, "logs", "checkpoints", "latest.pth")
    else:
        CHECKPOINT_PATH = ""
