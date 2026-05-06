# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import DATASETS
from .coco import CocoDataset


@DATASETS.register_module()
class PolyTopoBenchVectorDataset(CocoDataset):
    """COCO-format polygon dataset used by the GCP release configs."""

    METAINFO = dict(
        classes=('building'),
        palette=[(0, 0, 255)])
