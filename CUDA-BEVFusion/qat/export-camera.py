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


import warnings
warnings.filterwarnings("ignore")

import argparse
import os

import onnx
import torch
from onnxsim import simplify
from torchpack.utils.config import configs
from mmcv import Config
from mmdet3d.models import build_model
from mmdet3d.utils import recursive_eval

from torch import nn
from pytorch_quantization.nn.modules.tensor_quantizer import TensorQuantizer
import lean.quantize as quantize

def parse_args():
    parser = argparse.ArgumentParser(description="Export bevfusion model")
    parser.add_argument('--ckpt', type=str, default='qat/ckpt/bevfusion_ptq.pth')
    parser.add_argument('--fp16', action= 'store_true')
    args = parser.parse_args()
    return args

class SubclassCameraModule(nn.Module):
    def __init__(self, model):
        super(SubclassCameraModule, self).__init__()
        self.model = model

    def forward(self, img, depth):
        B, N, C, H, W = img.size()
        img = img.view(B * N, C, H, W)

        feat = self.model.encoders.camera.backbone(img)
        feat = self.model.encoders.camera.neck(feat)
        if not isinstance(feat, torch.Tensor):
            feat = feat[0]

        BN, C, H, W = map(int, feat.size())
        feat = feat.view(B, int(BN / B), C, H, W)

        def get_cam_feats(self, x, d):
            B, N, C, fH, fW = map(int, x.shape)
            d = d.view(B * N, *d.shape[2:])
            x = x.view(B * N, C, fH, fW)

            d = self.dtransform(d)
            x = torch.cat([d, x], dim=1)
            x = self.depthnet(x)

            depth = x[:, : self.D].softmax(dim=1)
            feat  = x[:, self.D : (self.D + self.C)].permute(0, 2, 3, 1)
            return feat, depth
        
        return get_cam_feats(self.model.encoders.camera.vtransform, feat, depth)

def main():
    args = parse_args()

    model  = torch.load(args.ckpt).module
    suffix = "int8"
    if args.fp16:
        suffix = "fp16"
        quantize.disable_quantization(model).apply()
        
    data = torch.load("example-data/example-data.pth")
    img = data["img"].data[0].cuda()
    points = [i.cuda() for i in data["points"].data[0]]

    camera_model = SubclassCameraModule(model)
    camera_model.cuda().eval()
    depth = torch.zeros(len(points), img.shape[1], 1, img.shape[-2], img.shape[-1]).cuda()

    downsample_model = model.encoders.camera.vtransform.downsample
    downsample_model.cuda().eval()
    downsample_in = torch.zeros(1, 80, 360, 360).cuda()

    save_root = f"qat/onnx_{suffix}"
    os.makedirs(save_root, exist_ok=True)

    with torch.no_grad():
        camera_backbone_onnx = f"{save_root}/camera.backbone.onnx"
        camera_vtransform_onnx = f"{save_root}/camera.vtransform.onnx"
        TensorQuantizer.use_fb_fake_quant = True
        torch.onnx.export(
            camera_model,
            (img, depth),
            camera_backbone_onnx,
            input_names=["img", "depth"],
            output_names=["camera_feature", "camera_depth_weights"],
            opset_version=13,
            do_constant_folding=True,
        )

        onnx_orig = onnx.load(camera_backbone_onnx)
        onnx_simp, check = simplify(onnx_orig)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(onnx_simp, camera_backbone_onnx)
        print(f"🚀 The export is completed. ONNX save as {camera_backbone_onnx} 🤗, Have a nice day~")

        torch.onnx.export(
            downsample_model,
            downsample_in,
            camera_vtransform_onnx,
            input_names=["feat_in"],
            output_names=["feat_out"],
            opset_version=13,
            do_constant_folding=True,
        )
        print(f"🚀 The export is completed. ONNX save as {camera_vtransform_onnx} 🤗, Have a nice day~")

if __name__ == "__main__":
    main()
