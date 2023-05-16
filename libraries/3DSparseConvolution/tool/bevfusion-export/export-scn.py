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

import sys; sys.path.insert(0, "./")

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
import funcs
import exptool
 
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Export scn to onnx file")
    parser.add_argument("--in-channel", type=int, default=5, help="SCN num of input channels")
    parser.add_argument("--config", type=str, default="configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml", help="Config yaml file")
    parser.add_argument("--ckpt", type=str, default="pretrained/bevfusion-det.pth", help="SCN Checkpoint (scn backbone checkpoint)")
    parser.add_argument("--input", type=str, default="deploy/data/input.pth", help="input pickle data, random if there have no input")
    parser.add_argument("--save-onnx", type=str, default="deploy/bevfusion/bevfusion.scn.onnx", help="output onnx")
    parser.add_argument("--save-tensor", type=str, default="deploy/bevfusion/infer", help="Save input/output tensor to file. The purpose of this operation is to verify the inference result of c++")
    parser.add_argument("--origin", action="store_true", help="Avoid inverse indices and use the original configuration(xyz) instead. According to my experiments, zyx format input has better performance. The difference is about 7ms on ORIN. You should select inverse indices as default configuration😊!")
    parser.add_argument("--inferonly", action="store_true", help="Avoid fusing the conv and bn of the model. Because the latter will result in abnormal inference due to the missing relu. If you wish to store the post-inference results, you should use this option😊!")
    args = parser.parse_args()

    # default inverse xyz to zyx
    inverse_indices = True
    if args.origin:
        inverse_indices = False

    if inverse_indices:
        args.save_onnx = os.path.splitext(args.save_onnx)[0] + ".zyx.onnx"
        args.save_tensor += ".zyx"
    else:
        args.save_onnx = os.path.splitext(args.save_onnx)[0] + ".xyz.onnx"
        args.save_tensor += ".xyz"

    os.makedirs(os.path.dirname(args.save_onnx), exist_ok=True)
    os.makedirs(os.path.dirname(args.save_tensor), exist_ok=True)
    configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(configs), filename=args.config)

    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    model.eval().cuda().half()
    model = model.encoders.lidar.backbone

    if args.ckpt:
        funcs.load_checkpoint(model, args.ckpt, startsname="encoders.lidar.backbone")

    if args.inferonly:
        print("layer_fusion will only be valid when exporting onnx. Otherwise it will lead to wrong inference results")
    else:
        model = funcs.layer_fusion(model)

    if args.input:
        voxels, coors, spatial_shape, batch_size = torch.load(args.input, map_location="cpu")
        voxels = voxels.detach().cuda().half()
        coors  = coors.detach().int().cuda()
    else:
        voxels = torch.zeros(1, args.in_channel).cuda().half()
        coors  = torch.zeros(1, 4).int().cuda()
        batch_size = 1
    
    if args.inferonly:
        exptool.inference_and_save_tensor(model, voxels, coors, batch_size, inverse_indices, args.save_tensor)
    else:
        exptool.export_onnx(model, voxels, coors, batch_size, inverse_indices, args.save_onnx)
        if not inverse_indices:
            funcs.cprint("🙈 According to my experiments, zyx format input has <yellow>better performance</yellow>. The difference is about 7ms on orin. You should select it as default configuration.")
            funcs.cprint("If the <yellow>--origin</yellow> argument is set, I will export the original model structure of bevfusion for you, which you should know is in xyz format input(<red>lower performance</red>)!")