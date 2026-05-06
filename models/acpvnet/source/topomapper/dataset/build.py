#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset builder utilities.

This module resolves dataset factories from `DatasetCatalog`, attaches the
appropriate transform pipeline, and returns the corresponding dataloaders.
"""

import torch
from .transforms import *
from . import latent_vertexheatmap_dataset
from topomapper.config.paths_catalog import DatasetCatalog


def build_train_dataset(cfg):
    assert len(cfg.DATASETS.TRAIN) == 1
    name = cfg.DATASETS.TRAIN[0]
    dargs = DatasetCatalog.get(name)
    factory_name = dargs['factory']
    args = dargs['args']

    if factory_name != "LatentVertexHeatmapTrainDataset":
        raise ValueError(f"Unsupported ACPV-Net release train dataset: {factory_name}")

    transform = Compose([
        ResizeImage(cfg.DATASETS.IMAGE.HEIGHT, cfg.DATASETS.IMAGE.WIDTH),
        ToTensor(),
        Normalize(cfg.DATASETS.IMAGE.PIXEL_MEAN,
                  cfg.DATASETS.IMAGE.PIXEL_STD,
                  cfg.DATASETS.IMAGE.TO_255)
    ])

    args['transform'] = transform
    args['augmentations'] = cfg.DATASETS.AUGMENTATIONS

    factory = getattr(latent_vertexheatmap_dataset, factory_name)
    collate = latent_vertexheatmap_dataset.collate_fn

    dataset = factory(**args)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.SOLVER.IMS_PER_BATCH,
        collate_fn=collate,
        shuffle=True,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        drop_last=True,
    )


def build_test_dataset(cfg):
    name = cfg.DATASETS.TEST[0]
    dargs = DatasetCatalog.get(name)
    factory_name = dargs['factory']
    args = dargs['args']

    # ==== Configure transforms ====
    transforms = Compose(
        [ResizeImage(cfg.DATASETS.IMAGE.HEIGHT,
                     cfg.DATASETS.IMAGE.WIDTH),
         ToTensor(),
         Normalize(cfg.DATASETS.IMAGE.PIXEL_MEAN,
                   cfg.DATASETS.IMAGE.PIXEL_STD,
                   cfg.DATASETS.IMAGE.TO_255)
         ]
    )

    args['transform'] = transforms
    if getattr(cfg.DATALOADER, "TEST_MAX_IMAGES", 0) > 0:
        args['max_images'] = int(cfg.DATALOADER.TEST_MAX_IMAGES)

    print("factory_name: ", factory_name)

    if factory_name != "LatentVertexHeatmapTestDataset":
        raise ValueError(f"Unsupported ACPV-Net release test dataset: {factory_name}")

    factory = getattr(latent_vertexheatmap_dataset, factory_name)
    collate = latent_vertexheatmap_dataset.collate_fn

    dataset = factory(**args)
    dataset = torch.utils.data.DataLoader(
        dataset, 
        batch_size=cfg.SOLVER.IMS_PER_BATCH,
        collate_fn=collate,
        shuffle=False,  # No shuffling is needed during testing.
        num_workers=cfg.DATALOADER.NUM_WORKERS,
    )
    return dataset
