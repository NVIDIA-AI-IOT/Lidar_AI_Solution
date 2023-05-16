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
import copy
import os
try:
    import apex
except:
    print("No APEX!")

from det3d import torchie
from det3d.datasets import build_dataloader, build_dataset
from det3d.models import build_detector
from det3d.torchie import Config
from det3d.torchie.apis import (
    batch_processor,
    set_random_seed,
)

from det3d.torchie.parallel import collate, collate_kitti
from det3d.torchie.trainer import load_checkpoint, save_checkpoint
from det3d.torchie.trainer.utils import all_gather, synchronize

import torch
from torch.utils.data import DataLoader

import time 
import numpy as np
from copy import deepcopy
import matplotlib.cm as cm
from matplotlib import pyplot as plt 

import onnx_export.funcs as funcs
import sparseconv_quantization as quantize

def eval_model(model, data_loader_test, dataset_test):
    model.eval()
    local_rank = 0
    work_dir = 'work_dirs/tmp'

    detections = {}
    cpu_device = torch.device("cpu")
    for i, data_batch in enumerate(data_loader_test):
        with torch.no_grad():
            outputs = batch_processor(
                model, data_batch, train_mode=False, local_rank=local_rank,
            )
        for output in outputs:
            token = output["metadata"]["token"]
            for k, v in output.items():
                if k not in [
                    "metadata",
                ]:
                    output[k] = v.to(cpu_device)
            detections.update(
                {token: output,}
            )

    all_predictions = all_gather(detections)
    if local_rank != 0:
        return

    predictions = {}
    for p in all_predictions:
        predictions.update(p)

    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    
    mAP = 0.0
    result_dict, mAP = dataset_test.evaluation(copy.deepcopy(predictions), output_dir=work_dir, testset=None)

    del detections
    del predictions
    del all_predictions
    return mAP

def disable_quant_layer(model_finetune):
    print("🔥Disable Conv1FirstSparseBasicBlock Mode🔥") 
    quantize.disable_quantization(model_finetune.backbone.conv_input).apply()
    quantize.disable_quantization(model_finetune.backbone.conv1[0].conv1).apply()
    quantize.disable_quantization(model_finetune.backbone.conv1[0].conv2).apply()
    quantize.disable_quantization(model_finetune.backbone.conv1[0].quant_add).apply()
    return model_finetune

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

def centerpoint_qat_test(nuScense_config, weight_path, use_quantization):
    quantize.initialize()
    cfg = Config.fromfile(nuScense_config)    

    checkpoint_path = weight_path 
    if use_quantization:
        print("🔥 Evaluate QAT/PTQ Model 🔥")
        model_QAT  = load_model(cfg, checkpoint_path = None, use_fuse = True, use_quant_spconv=True, use_quant_add=True)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model_QAT.load_state_dict(checkpoint["state_dict"], strict=True)   
        funcs.layer_fusion_relu(model_QAT.backbone)
        disable_quant_layer(model_QAT)
    else:
        print("🔥 Evaluate Original Model 🔥")
        model_QAT  = load_model(cfg, checkpoint_path = weight_path, use_fuse = False, use_quant_spconv=False, use_quant_add=False)       

    dataset_val   = build_dataset(cfg.data.val)
    print('val nums:{}'.format(len(dataset_val)))

    data_loader_val = DataLoader(
        dataset_val,
        batch_size=4,
        sampler=None, 
        shuffle=False,
        num_workers=4,
        collate_fn=collate_kitti,  
        pin_memory=False,
    )

    print("🔥 Evaluate orin model 🔥")
    model_QAT.eval()
    mAP = eval_model(model_QAT, data_loader_val, dataset_val)
    return mAP


def quant_sensitivity_profile(nuScense_config, calibrate_batch, eval_origin, weight):
    quantize.initialize()
    cfg = Config.fromfile(nuScense_config)     
    
    checkpoint_path = weight
    model_original = load_model(cfg, checkpoint_path = checkpoint_path, use_fuse = True, use_quant_spconv=True, use_quant_add=True)
   
    print_ptq_status = True
    quantize.set_quantizer_fast(model_original)
    dataset_train = build_dataset(cfg.data.train)
    dataset_val   = build_dataset(cfg.data.val)
    print('train nums:{} val nums:{}'.format(len(dataset_train), len(dataset_val)))
    data_loader_train = DataLoader(
        dataset_train,
        batch_size=4,
        sampler=None,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_kitti,
        pin_memory=False, 
    )

    data_loader_val = DataLoader(
        dataset_val,
        batch_size=4,
        sampler=None, 
        shuffle=False,
        num_workers=4,
        collate_fn=collate_kitti,  
        pin_memory=False,
    )

    model_original.eval()
    print("🔥 Start Calibrate 🔥") 
    with torch.no_grad():
        quantize.calibrate_model(model_original, data_loader_train, 0, batch_processor, calibrate_batch)

    if eval_origin:
        print("🔥 Evaluate Origin Model 🔥") 
        with quantize.disable_quantization(model_original):
            eval_model(model_original, data_loader_val, dataset_val)  

    print("🔥 Sensitivity Profile 🔥")
    quantize.build_sensitivity_profile(model_original, data_loader_val, dataset_val,eval_model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='eval.py')
    subps  = parser.add_subparsers(dest="cmd")
    
    sensitivity_cmd = subps.add_parser("sensitivity", help="sensitivity profile...")
    sensitivity_cmd.add_argument("--nuScense-config", type=str, default=None, help="nuScense type")
    sensitivity_cmd.add_argument("--weight", type=str, default=None, help="nuScense type")
    sensitivity_cmd.add_argument("--calibrate_batch", type=int, default=400, help="calibrate batch")
    sensitivity_cmd.add_argument("--eval-origin", action="store_true", help="do eval for origin model")

    test_cmd = subps.add_parser("test", help="Do evaluate")
    test_cmd.add_argument("--nuScense-config", type=str, default=None, help="nuScense type")
    test_cmd.add_argument("--weight", type=str, default=None, help="weight file")
    test_cmd.add_argument("--use-quantization", action="store_true", help="enabel quantization")
    args = parser.parse_args()

    set_random_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if args.cmd == "test":
        centerpoint_qat_test(args.nuScense_config, args.weight, args.use_quantization)
    elif args.cmd == 'sensitivity':
        quant_sensitivity_profile(args.nuScense_config, args.calibrate_batch, args.eval_origin, args.weight)