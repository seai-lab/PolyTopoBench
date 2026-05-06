from yacs.config import CfgNode as CN
from .models import MODELS
from .dataset import DATASETS
from .solver import SOLVER
cfg = CN()

cfg.MODEL = MODELS
cfg.DATASETS = DATASETS
cfg.SOLVER = SOLVER

cfg.DATALOADER = CN()
cfg.DATALOADER.NUM_WORKERS = 8
cfg.DATALOADER.TEST_MAX_IMAGES = 0

cfg.OUTPUT_DIR = "outputs/default"
cfg.TEST_OUTPUT_DIR = ""
cfg.DDPM_TIMESTEPS = 1000

cfg.USE_EMA = False
cfg.EMA_DECAY = 0.9999
cfg.EMA_USE_NUM_UPDATES = True

