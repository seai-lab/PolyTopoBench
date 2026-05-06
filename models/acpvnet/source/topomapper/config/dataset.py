from yacs.config import CfgNode as CN
# ---------------------------------------------------------------------------- #
# Dataset options
# ---------------------------------------------------------------------------- #
DATASETS = CN()
DATASETS.TRAIN = ("custom_acpv_train_with_latent_vertex_heatmap",)
DATASETS.VAL = ("custom_acpv_val_with_latent_vertex_heatmap",)
DATASETS.TEST = ("custom_acpv_test_with_latent_vertex_heatmap",)
DATASETS.FACTORY = "LatentVertexHeatmapTrainDataset"
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

DATASETS.NUM_CLASSES = 2
DATASETS.AUGMENTATIONS = ['rot0']
