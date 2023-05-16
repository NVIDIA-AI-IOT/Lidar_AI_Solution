# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import argparse
import json
import os
import sys

from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaWarning
import warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaWarning)

import numpy as np
import torch
import yaml
from det3d.datasets import build_dataloader, build_dataset
from det3d.torchie.parallel import collate, collate_kitti
from torch.utils.data import DataLoader
from det3d.models import build_detector
from det3d.torchie import Config
from det3d.torchie.apis import (
    build_optimizer,
    get_root_logger,
    init_dist,
    set_random_seed,
    train_detector,
    train_detector_QAT,
    batch_processor
)

import torch.distributed as dist
import subprocess

import onnx_export.funcs as funcs
import sparseconv_quantization as quantize
from det3d.torchie.trainer import load_checkpoint, save_checkpoint

 
# quantize.quant_add_module() -> Solving the problem of multiple inputs with different quantization factors
# funcs.layer_fusion_bn       -> PTQ/PTQ using fuseBn mode reduces quantification errors
def load_model(cfg, checkpoint_path = None, use_fuse =False, use_quant_spconv= False, use_quant_add=False):
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    if checkpoint_path != None:
        checkpoint = load_checkpoint(model, checkpoint_path, map_location="cpu")
    model.eval()
    model = model.cuda()
    if use_quant_spconv:
        quantize.quant_sparseconv_module(model.backbone)

    if use_quant_add:
        quantize.quant_add_module(model.backbone)

    if use_fuse:
        model.backbone = funcs.layer_fusion_bn(model.backbone)          
    return model

def disable_quant_layer(model_finetune):
    print("🔥Disable Conv1FirstSparseBasicBlock Mode🔥") 
    quantize.disable_quantization(model_finetune.backbone.conv_input).apply()
    quantize.disable_quantization(model_finetune.backbone.conv1[0].conv1).apply()
    quantize.disable_quantization(model_finetune.backbone.conv1[0].conv2).apply()
    quantize.disable_quantization(model_finetune.backbone.conv1[0].quant_add).apply()
    return model_finetune

def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--work_dir", help="the dir to save logs and models")
    parser.add_argument("--resume_from", help="the checkpoint file to resume from")
    parser.add_argument("--calibrate_batch", type=int, default=300, help="calibrate batch")
    parser.add_argument("--ptq_mode", action="store_true", help="use ptq mode")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="whether to evaluate the checkpoint during training",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="number of gpus to use " "(only applicable to non-distributed training)",
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument(
        "--launcher",
        choices=["pytorch", "slurm"],
        default="pytorch",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--autoscale-lr",
        action="store_true",
        help="automatically scale lr with the number of gpus",
    )
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args


def main():
    quantize.initialize()
    # torch.manual_seed(0)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # np.random.seed(0)

    args = parse_args()

    cfg = Config.fromfile(args.config)

    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from

    distributed = False
    if "WORLD_SIZE" in os.environ:
        distributed = int(os.environ["WORLD_SIZE"]) > 1

    if distributed:
        if args.launcher == "pytorch":
            torch.cuda.set_device(args.local_rank)
            torch.distributed.init_process_group(backend="nccl", init_method="env://")
            cfg.local_rank = args.local_rank
        elif args.launcher == "slurm":
            proc_id = int(os.environ["SLURM_PROCID"])
            ntasks = int(os.environ["SLURM_NTASKS"])
            node_list = os.environ["SLURM_NODELIST"]
            num_gpus = torch.cuda.device_count()
            cfg.gpus = num_gpus
            torch.cuda.set_device(proc_id % num_gpus)
            addr = subprocess.getoutput(
                f"scontrol show hostname {node_list} | head -n1")
            # specify master port
            port = None
            if port is not None:
                os.environ["MASTER_PORT"] = str(port)
            elif "MASTER_PORT" in os.environ:
                pass  # use MASTER_PORT in the environment variable
            else:
                # 29500 is torch.distributed default port
                os.environ["MASTER_PORT"] = "29501"
            # use MASTER_ADDR in the environment variable if it already exists
            if "MASTER_ADDR" not in os.environ:
                os.environ["MASTER_ADDR"] = addr
            os.environ["WORLD_SIZE"] = str(ntasks)
            os.environ["LOCAL_RANK"] = str(proc_id % num_gpus)
            os.environ["RANK"] = str(proc_id)

            dist.init_process_group(backend="nccl")
            cfg.local_rank = int(os.environ["LOCAL_RANK"])

        cfg.gpus = dist.get_world_size()
    else:
        cfg.local_rank = args.local_rank 

    if args.autoscale_lr:
        cfg.lr_config.lr_max = cfg.lr_config.lr_max * cfg.gpus

    # init logger before other steps
    logger = get_root_logger(cfg.log_level)
    logger.info("Distributed training: {}".format(distributed))
    logger.info(f"torch.backends.cudnn.benchmark: {torch.backends.cudnn.benchmark}")

    if args.local_rank == 0:
        # copy important files to backup
        backup_dir = os.path.join(cfg.work_dir, "det3d")
        os.makedirs(backup_dir, exist_ok=True)
        # os.system("cp -r * %s/" % backup_dir)
        # logger.info(f"Backup source files to {cfg.work_dir}/det3d")

    # set random seeds
    if args.seed is not None:
        logger.info("Set random seed to {}".format(args.seed))
        set_random_seed(args.seed)

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        datasets.append(build_dataset(cfg.data.val))

    data_loaders = [
        build_dataloader(
            ds, cfg.data.samples_per_gpu, cfg.data.workers_per_gpu, dist=distributed
        )
        for ds in datasets
    ]
 
    print(cfg.data.samples_per_gpu, cfg.data.workers_per_gpu)
    model  = load_model(cfg, checkpoint_path = cfg.resume_from, use_fuse = True, use_quant_spconv=True, use_quant_add=True)
    quantize.set_quantizer_fast(model)
    print("🔥 start calibrate 🔥 ")
    with torch.no_grad():
        quantize.calibrate_model(model, data_loaders[0], 0, batch_processor, args.calibrate_batch)
    disable_quant_layer(model)
    quantize.print_quantizer_status(model)
    if args.ptq_mode:
        save_checkpoint(model, args.work_dir + '/ptq.pth')
        exit(0)
        
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    train_detector_QAT(
        model,
        datasets,
        data_loaders,
        cfg,
        distributed=distributed,
        validate=args.validate,
        logger=logger,
    )


if __name__ == "__main__":
    main()