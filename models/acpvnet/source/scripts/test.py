#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run model inference and evaluation under a given configuration.

This script resolves the correct detector implementation from the config,
loads the latest checkpoint from `cfg.OUTPUT_DIR`, and executes the testing
pipeline.
"""

import os
import argparse
import logging

import torch

from topomapper.config import cfg
from topomapper.utils.logger import setup_logger
from topomapper.utils.checkpoint import DetectronCheckpointer
from tools.test_pipelines import TestPipeline

torch.multiprocessing.set_sharing_strategy('file_system')


def parse_args():
    parser = argparse.ArgumentParser(description='Testing')

    parser.add_argument("--config-file",
                        metavar="FILE",
                        help="path to config file",
                        type=str,
                        default=None,
                        )

    parser.add_argument("--eval-type",
                        type=str,
                        help="Evalutation type for the test results",
                        default="coco_iou",
                        choices=["coco_iou",  "boundary_iou", "polis"]
                        )

    parser.add_argument("--sampler",
                        type=str,
                        help="Sampling method to use",
                        default="direct",
                        choices=["direct", "ddim", "ddpm"]
                        )

    parser.add_argument("--ddim-steps",
                        type=int,
                        help="Number of DDIM sampling steps",
                        default=200)

    parser.add_argument("opts",
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER
                        )

    return parser.parse_args()


def resolve_model_class(cfg):
    if cfg.MODEL.NAME != "HRNet32v2":
        raise ValueError(f"ACPV-Net release uses HRNet32v2, got {cfg.MODEL.NAME}")
    if not bool(getattr(cfg.MODEL.unet_config, "target", None)):
        raise ValueError("ACPV-Net release expects the latent diffusion vertex branch.")
    from topomapper.upernet_detector_vh_m_ldm import BuildingUPerNetDetector
    return BuildingUPerNetDetector


def test(cfg, args):
    logger = logging.getLogger("testing")
    device = cfg.MODEL.DEVICE

    # Build the model from the configuration.
    ModelCls = resolve_model_class(cfg)
    model = ModelCls(cfg).to(device)

    # Log EMA usage if enabled.
    inner = model.module if hasattr(model, "module") else model
    if getattr(inner, "use_ema", False):
        logger.info(f"EMA enabled for inference (decay={getattr(inner, 'ema_decay', None)}).")

    # Load the checkpoint.
    if args.config_file is not None:
        checkpointer = DetectronCheckpointer(cfg,
                                         model,
                                         save_dir=cfg.OUTPUT_DIR,
                                         save_to_disk=True,
                                         logger=logger)
        _ = checkpointer.load()        
        model = model.eval()

    # Run the testing pipeline.
    logger.info(f"Checkpoint load dir (cfg.OUTPUT_DIR): {cfg.OUTPUT_DIR}")
    logger.info(f"Test output dir: {TestPipeline(cfg, args.eval_type).output_dir}")
    test_pipeline = TestPipeline(cfg, args.eval_type)
    test_pipeline.test(model)
    # test_pipeline.eval()


if __name__ == "__main__":
    args = parse_args()

    # Merge the config file if provided.
    if args.config_file is not None:
        cfg.merge_from_file(args.config_file)
    else:
        cfg.OUTPUT_DIR = 'outputs/default'
        os.makedirs(cfg.OUTPUT_DIR,exist_ok=True)

    # Inject sampler settings from the command line into cfg.
    if args.sampler is not None:
        cfg.SAMPLER = args.sampler
    if args.ddim_steps is not None:
        cfg.DDIM_STEPS = args.ddim_steps

    # Merge additional command-line overrides and freeze the config.
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    
    output_dir = cfg.OUTPUT_DIR
    logger = setup_logger('testing', output_dir)

    logger.info(args)
    if args.config_file is not None:
        logger.info("Loaded configuration file {}".format(args.config_file))
    else:
        logger.info("Loaded the default configuration for testing")
    logger.info("CLI config overrides: {}".format(args.opts if args.opts else []))
    logger.info("Running with merged config (after CLI overrides):\n{}".format(cfg))

    test(cfg, args)
