#
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
SparseRCNN Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""

import os
import itertools
import time
from typing import Any, Dict, List, Set
import logging
from collections import OrderedDict

import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.data.datasets.coco import load_coco_json
from detectron2.engine import AutogradProfiler, DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import COCOEvaluator, verify_results
from detectron2.solver.build import maybe_add_gradient_clipping

from sparsercnn import SparseRCNNDatasetMapper, add_sparsercnn_config, SparseRCNNWithTTA


class Trainer(DefaultTrainer):
#     """
#     Extension of the Trainer class adapted to SparseRCNN.
#     """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, tasks=("bbox",), distributed=True, output_dir=output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = SparseRCNNDatasetMapper(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_optimizer(cls, cfg, model):
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for key, value in model.named_parameters(recurse=True):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "backbone" in key:
                lr = lr * cfg.SOLVER.BACKBONE_MULTIPLIER
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        def maybe_add_full_model_gradient_clipping(optim):  # optim: the optimizer class
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # Only support Sparse R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = SparseRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

    def build_hooks(self):
        hook_list = super().build_hooks()
        if comm.is_main_process() and self.cfg.TEST.EVAL_PERIOD > 0 and len(self.cfg.DATASETS.TEST) > 0:
            for index, hook in enumerate(hook_list):
                if isinstance(hook, hooks.EvalHook):
                    hook_list.insert(
                        index + 1,
                        hooks.BestCheckpointer(
                            self.cfg.TEST.EVAL_PERIOD,
                            self.checkpointer,
                            "bbox/AP",
                            mode="max",
                            file_prefix="model_best",
                        ),
                    )
                    break
        return hook_list

def setup(args):
    """
    Create configs and perform basic setups.
    """
    register_datasets_from_args(args)
    cfg = get_cfg()
    add_sparsercnn_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.defrost()
    if args.train_dataset:
        cfg.DATASETS.TRAIN = (args.train_dataset,)
    if args.val_dataset:
        cfg.DATASETS.TEST = (args.val_dataset,)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def build_argument_parser():
    parser = default_argument_parser()
    parser.add_argument("--train-dataset", default=None, help="Dataset name to register for training")
    parser.add_argument("--train-json", default=None, help="COCO-format training annotation JSON")
    parser.add_argument("--train-path", default=None, help="Training image root")
    parser.add_argument("--val-dataset", default=None, help="Dataset name to register for validation")
    parser.add_argument("--val-json", default=None, help="COCO-format validation annotation JSON")
    parser.add_argument("--val-path", default=None, help="Validation image root")
    return parser


def register_coco_dataset(dataset_name, annotation_json, image_root):
    if not dataset_name or not annotation_json or not image_root:
        return

    if dataset_name in DatasetCatalog.list():
        return

    DatasetCatalog.register(
        dataset_name,
        lambda ann=annotation_json, root=image_root, name=dataset_name: load_coco_json(ann, root, name),
    )
    # Let load_coco_json derive thing_classes from the annotation file so the
    # detector can reuse different single-class exports without metadata clashes.
    MetadataCatalog.get(dataset_name).set(
        json_file=annotation_json,
        image_root=image_root,
        evaluator_type="coco",
    )


def register_datasets_from_args(args):
    register_coco_dataset(args.train_dataset, args.train_json, args.train_path)
    register_coco_dataset(args.val_dataset, args.val_json, args.val_path)


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = build_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
