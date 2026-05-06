import os
import time
import argparse
import logging
import random
import numpy as np
import datetime

from hisup.config import cfg
from hisup.detector import BuildingDetector
from hisup.dataset import build_train_dataset
from hisup.utils.comm import to_single_device
from hisup.solver import make_lr_scheduler, make_optimizer
from hisup.utils.logger import setup_logger
from hisup.utils.miscellaneous import save_config
from hisup.utils.metric_logger import MetricLogger
from hisup.utils.checkpoint import DetectronCheckpointer

import torch
torch.multiprocessing.set_sharing_strategy('file_system')

class LossReducer(object):
    def __init__(self, cfg):
        self.loss_weights = dict(cfg.MODEL.LOSS_WEIGHTS)

    def __call__(self, loss_dict):
        total_loss = sum([self.loss_weights[k] * loss_dict[k]
                          for k in self.loss_weights.keys()])

        return total_loss

def parse_args():
    parser = argparse.ArgumentParser(description='Testing')

    parser.add_argument("--config-file",
                        metavar="FILE",
                        help="path to config file",
                        type=str,
                        default=None,
                        )
    
    parser.add_argument("--clean",
                        default=False,
                        action='store_true')

    parser.add_argument("--seed",
                        default=2,
                        type=int)

    parser.add_argument("opts",
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER
                        )

    args = parser.parse_args()
    
    return args

def set_random_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train(cfg):
    logger = logging.getLogger("training")
    device = cfg.MODEL.DEVICE
    model = BuildingDetector(cfg)
    model = model.to(device)

    train_dataset = build_train_dataset(cfg)
    
    optimizer = make_optimizer(cfg,model)
    scheduler = make_lr_scheduler(cfg,optimizer)
    
    loss_reducer = LossReducer(cfg)
    
    arguments = {}
    arguments["epoch"] = 0
    max_epoch = cfg.SOLVER.MAX_EPOCH
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    arguments["max_epoch"] = max_epoch
    max_train_steps = int(os.environ.get("POLYTOPOBENCH_MAX_TRAIN_STEPS", "0") or 0)

    checkpointer = DetectronCheckpointer(cfg,
                                        model,
                                        optimizer,
                                        scheduler,
                                        save_dir=cfg.OUTPUT_DIR,
                                        save_to_disk=True,
                                        logger=logger)

    # Auto-resume from last_checkpoint if HISUP_RESUME=1 and the file exists.
    # Parses epoch number from the filename (model_XXXXX.pth) to continue
    # training from the correct epoch. Optimizer (and scheduler if present) are
    # reloaded by DetectronCheckpointer.load(). Older paper_v2 checkpoints were
    # saved without scheduler state, so we also step the scheduler forward
    # manually to match the resumed epoch so the LR schedule is correct.
    if os.environ.get("HISUP_RESUME", "0") == "1":
        last_ckpt_file = os.path.join(cfg.OUTPUT_DIR, "last_checkpoint")
        if os.path.exists(last_ckpt_file):
            checkpoint_extra = checkpointer.load()
            scheduler_restored = bool(checkpoint_extra) and "scheduler" in checkpoint_extra
            try:
                with open(last_ckpt_file, "r") as _f:
                    _last = _f.read().strip()
                _base = os.path.splitext(os.path.basename(_last))[0]
                _num = _base.split("_")[-1]
                arguments["epoch"] = int(_num)
                logger.info("HISUP_RESUME: resumed from epoch {}".format(arguments["epoch"]))
                if not scheduler_restored and arguments["epoch"] > 0:
                    # Advance scheduler to match resumed epoch (older ckpt had no scheduler state)
                    for _ in range(arguments["epoch"]):
                        scheduler.step()
                    logger.info("HISUP_RESUME: stepped scheduler forward {} times".format(arguments["epoch"]))
            except Exception as _exc:
                logger.warning("HISUP_RESUME: failed to parse epoch from last_checkpoint ({})".format(_exc))
        else:
            logger.info("HISUP_RESUME=1 but no last_checkpoint found; training from scratch")

    start_training_time = time.time()
    end = time.time()

    start_epoch = arguments['epoch']
    epoch_size = len(train_dataset)

    global_iteration = epoch_size*start_epoch

    for epoch in range(start_epoch+1, arguments['max_epoch']+1):
        meters = MetricLogger(" ")
        model.train()
        arguments['epoch'] = epoch

        for it, (images, annotations) in enumerate(train_dataset):
            if max_train_steps and it >= max_train_steps:
                break
            data_time = time.time() - end
            images = images.to(device)
            annotations = to_single_device(annotations,device)
            
            loss_dict, _ = model(images,annotations)
            total_loss = loss_reducer(loss_dict)

            with torch.no_grad():
                loss_dict_reduced = {k:v.item() for k,v in loss_dict.items()}
                loss_reduced = total_loss.item()
                meters.update(loss=loss_reduced, **loss_dict_reduced)
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            global_iteration +=1
            
            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time, data=data_time)

            eta_batch = epoch_size*(max_epoch-epoch+1) - it +1
            eta_seconds = meters.time.global_avg*eta_batch
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if it % 20 == 0 or it+1 == len(train_dataset):
                logger.info(
                    meters.delimiter.join(
                        [
                            "eta: {eta}",
                            "epoch: {epoch}",
                            "iter: {iter}",
                            "{meters}",
                            "lr: {lr:.6f}",
                            "max mem: {memory:.0f}\n",
                        ]
                    ).format(
                        eta=eta_string,
                        epoch=epoch,
                        iter=it,
                        meters=str(meters),
                        lr=optimizer.param_groups[0]["lr"],
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    )
                )
        
        if epoch % checkpoint_period == 0 or epoch == max_epoch:
            checkpointer.save('model_{:05d}'.format(epoch))
        scheduler.step()
    
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / epoch)".format(
            total_time_str, total_training_time / (max_epoch)
        )
    )

if __name__ == "__main__":
    args = parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    
    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        if os.path.isdir(output_dir) and args.clean:
            import shutil
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)

    logger = setup_logger('training', output_dir, out_file='train.log')
    logger.info(args)
    logger.info("Loaded configuration file {}".format(args.config_file))

    with open(args.config_file,"r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)

    logger.info("Running with config:\n{}".format(cfg))
    output_config_path = os.path.join(cfg.OUTPUT_DIR, 'config.yml')
    logger.info("Saving config into: {}".format(output_config_path))

    save_config(cfg, output_config_path)
    set_random_seed(args.seed, True)
    train(cfg)
