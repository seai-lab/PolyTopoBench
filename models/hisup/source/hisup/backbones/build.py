import os

from .registry import MODELS
from .hrnet48v2 import HighResolutionNet as HRNet48v2
from .hrnet32v2 import HighResolutionNet as HRNet32v2
from .multi_task_head import MultitaskHead

@MODELS.register("HRNet48v2")
def build_hrnet48(cfg):
    head_size = cfg.MODEL.HEAD_SIZE
    num_class = sum(sum(head_size, []))

    model = HRNet48v2(cfg,
                      head=lambda c_in, c_out: MultitaskHead(c_in, c_out, head_size=head_size),
                      num_class = num_class)
    pretrained = 'hisup/backbones/hrnet_imagenet/hrnetv2_w48_imagenet_pretrained.pth'
    model.init_weights(pretrained=pretrained)
    if not os.path.isfile(pretrained):
        print('WARNING: ImageNet weights {} not found (cwd {}); the HRNet backbone is randomly '
              'initialised.'.format(pretrained, os.getcwd()))
    print('INFO:build hrnet-w48-v2 backbone')
    return model

@MODELS.register("HRNet32v2")
def build_hrnet32(cfg):
    head_size = cfg.MODEL.HEAD_SIZE
    num_class = sum(sum(head_size, []))

    model = HRNet32v2(cfg,
                      head=lambda c_in, c_out: MultitaskHead(c_in, c_out, head_size=head_size),
                      num_class = num_class)

    pretrained = 'hisup/backbones/hrnet_imagenet/hrnetv2_w32_imagenet_pretrained.pth'
    model.init_weights(pretrained=pretrained)
    print('INFO:build hrnet-w32-v2 backbone')
    return model

def build_backbone(cfg):
    assert cfg.MODEL.NAME in MODELS,  \
        "cfg.MODELS.NAME: {} is not registered in registry".format(cfg.MODELS.NAME)

    return MODELS[cfg.MODEL.NAME](cfg)
