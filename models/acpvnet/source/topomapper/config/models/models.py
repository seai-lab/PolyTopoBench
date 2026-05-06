from yacs.config import CfgNode as CN

MODELS = CN()

MODELS.NAME = "HRNet32v2"
MODELS.DEVICE = "cuda"
MODELS.HEAD_SIZE  = [[2]]
MODELS.OUT_FEATURE_CHANNELS = 256

# loss type
MODELS.LOSS_WEIGHTS = CN(new_allowed=True)
MODELS.LOSS = CN()
MODELS.LOSS.MASK_LOSS_TYPE = "ce"          # "ce" or "weighted_ce"
MODELS.LOSS.MASK_AUX_LOSS_TYPE = "ce"      # "ce" or "weighted_ce"
MODELS.LOSS.W_BG = 1.0
MODELS.LOSS.W_OBJ = 1.0
MODELS.LOSS.PARCEL_LOSS_TYPE = "ce"
MODELS.LOSS.PARCEL_AUX_LOSS_TYPE = "ce"
MODELS.LOSS.PARCEL_W_BG = 1.0
MODELS.LOSS.PARCEL_W_OBJ = 1.0
# UPerHead parameters
MODELS.DECODE_HEAD = CN()
MODELS.DECODE_HEAD.IN_CHANNELS = [32, 64, 128, 256]
MODELS.DECODE_HEAD.IN_INDEX = [0, 1, 2, 3]
MODELS.DECODE_HEAD.POOL_SCALES = [1, 2, 3, 6]
MODELS.DECODE_HEAD.CHANNELS = 512
MODELS.DECODE_HEAD.DROPOUT_RATIO = 0.1
MODELS.DECODE_HEAD.NUM_CLASSES = 2
MODELS.DECODE_HEAD.ALIGN_CORNERS = False

# FCNHead parameters
MODELS.AUX_HEAD = CN()
MODELS.AUX_HEAD.IN_CHANNELS = 128
MODELS.AUX_HEAD.IN_INDEX = 2
MODELS.AUX_HEAD.CHANNELS = 256
MODELS.AUX_HEAD.NUM_CONVS = 1
MODELS.AUX_HEAD.CONCAT_INPUT = False
MODELS.AUX_HEAD.DROPOUT_RATIO = 0.1
MODELS.AUX_HEAD.NUM_CLASSES = 2
MODELS.AUX_HEAD.ALIGN_CORNERS = False

# Denoising UNet parameters
MODELS.unet_config = CN()
MODELS.unet_config.target = None

MODELS.unet_config.params = CN()
MODELS.unet_config.params.image_size = 128
MODELS.unet_config.params.in_channels = 3
MODELS.unet_config.params.out_channels = 3
MODELS.unet_config.params.model_channels = 224
MODELS.unet_config.params.attention_resolutions = [8, 4, 2]
MODELS.unet_config.params.num_res_blocks = 2
MODELS.unet_config.params.channel_mult = [1, 2, 3, 4]
MODELS.unet_config.params.num_head_channels = 32
MODELS.unet_config.params.context_dim = None
MODELS.unet_config.params.use_spatial_transformer = False

MODELS.conditioning_key = None

