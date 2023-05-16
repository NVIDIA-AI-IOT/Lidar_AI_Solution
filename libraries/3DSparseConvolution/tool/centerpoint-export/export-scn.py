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

import sys; sys.path.insert(0, ".")

from det3d.models.backbones.scn import SpMiddleResNetFHD
import torch
import pickle
import argparse

# custom functional package
import funcs
import exptool
 
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Export scn to onnx file")
    parser.add_argument("--in-channel", type=int, default=5, help="SCN num of input channels")
    parser.add_argument("--ckpt", type=str, default=None, help="SCN Checkpoint (scn backbone checkpoint)")
    parser.add_argument("--input", type=str, default=None, help="input pickle data, random if there have no input")
    parser.add_argument("--save-onnx", type=str, default="centerpoint.scn.onnx", help="output onnx")
    parser.add_argument("--save-tensor", type=str, default=None, help="Save input/output tensor to file. The purpose of this operation is to verify the inference result of c++")
    args = parser.parse_args()

    model = SpMiddleResNetFHD(args.in_channel).cuda().eval().half()
    if args.ckpt:
        model = funcs.load_scn_backbone_checkpoint(model, args.ckpt)

    model = funcs.layer_fusion(model)

    print("Fusion model:")
    print(model)

    if args.input:
        with open(args.input, "rb") as f:
            voxels, coors, spatial_shape, batch_size = pickle.load(f)
            voxels = torch.tensor(voxels).half().cuda()
            coors  = torch.tensor(coors).int().cuda()
    else:
        voxels = torch.zeros(1, args.in_channel).half().cuda()
        coors  = torch.zeros(1, 4).int().cuda()
        batch_size    = 1
        spatial_shape = [1440, 1440, 40]

    exptool.export_onnx(model, voxels, coors, batch_size, spatial_shape, args.save_onnx, args.save_tensor)