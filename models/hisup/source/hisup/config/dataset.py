from yacs.config import CfgNode as CN
# ---------------------------------------------------------------------------- #
# Dataset options
# ---------------------------------------------------------------------------- #
DATASETS = CN()
DATASETS.TRAIN = ("custom_hisup_train",)
DATASETS.VAL = ("custom_hisup_val",)
DATASETS.TEST = ("custom_hisup_val",)
DATASETS.ROTATE_F = False
DATASETS.IMAGE = CN()
DATASETS.IMAGE.HEIGHT = 512
DATASETS.IMAGE.WIDTH  = 512

DATASETS.IMAGE.PIXEL_MEAN = [109.730, 103.832, 98.681]
DATASETS.IMAGE.PIXEL_STD  = [22.275, 22.124, 23.229]
DATASETS.IMAGE.TO_255 = True
DATASETS.TARGET = CN()
DATASETS.TARGET.HEIGHT= 128
DATASETS.TARGET.WIDTH = 128

DATASETS.ORIGIN = CN()
DATASETS.ORIGIN.HEIGHT = 512
DATASETS.ORIGIN.WIDTH = 512
