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

import sys

from torchpack.utils.config import configs
from mmcv.cnn import fuse_conv_bn
from mmcv import Config
from mmdet3d.models import build_model
from mmdet3d.utils import recursive_eval
from mmcv.runner import wrap_fp16_model
import torch
import argparse
import os

# custom functional package
import lean.funcs as funcs
import lean.exptool as exptool
import lean.quantize as quantize 

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Export scn to onnx file")
    parser.add_argument("--in-channel", type=int, default=5, help="SCN num of input channels")
    parser.add_argument("--ckpt", type=str, default="qat/ckpt/bevfusion_ptq.pth", help="SCN Checkpoint (scn backbone checkpoint)")
    parser.add_argument("--save", type=str, default="qat/onnx/lidar.backbone.onnx", help="output onnx")
    parser.add_argument("--inverse", action="store_true", help="Transfer the coordinate order of the index from xyz to zyx")
   
    args = parser.parse_args()
    inverse_indices = args.inverse
    if inverse_indices:
        args.save = os.path.splitext(args.save)[0] + ".zyx.onnx"
    else:
        args.save = os.path.splitext(args.save)[0] + ".xyz.onnx"
    
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    model = torch.load(args.ckpt).module
    model.eval().cuda().half()
    model = model.encoders.lidar.backbone

    quantize.disable_quantization(model).apply()

    # Set layer attributes
    for name, module in model.named_modules():
        module.precision = "int8"
        module.output_precision = "int8"
    
    model.conv_input.precision = "fp16"
    model.conv_out.output_precision = "fp16"

    voxels = torch.zeros(1, args.in_channel).cuda().half()
    coors  = torch.zeros(1, 4).int().cuda()
    batch_size = 1
    
    exptool.export_onnx(model, voxels, coors, batch_size, inverse_indices, args.save)
