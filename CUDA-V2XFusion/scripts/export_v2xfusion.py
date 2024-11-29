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
import os
import time
import warnings
import tensor
import mmcv
import onnx
import torch
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint, wrap_fp16_model
from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet.apis import multi_gpu_test, set_random_seed
from mmdet.datasets import replace_ImageToTensor
from tqdm import tqdm
from torchpack.utils.config import configs
from mmdet3d.utils import get_root_logger, convert_sync_batchnorm, recursive_eval
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.datasets.v2x_dataset import collate_fn
from functools import partial
import sys
from torch import nn
from pytorch_quantization.nn.modules.tensor_quantizer import TensorQuantizer
from pytorch_quantization.nn.modules.tensor_quantizer import TensorQuantizer
import tensor
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="MMDet test (and eval) a model")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument("--precision", choices=["int8", "fp16"],help="Select the precision mode (int8 or fp16)")
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    args = parser.parse_args()
    return args


class no_jit_trace:
    def __enter__(self):
        # pylint: disable=protected-access
        self.state = torch._C._get_tracing_state()
        torch._C._set_tracing_state(None)

    def __exit__(self, *args):
        torch._C._set_tracing_state(self.state)
        self.state = None

class SubclassCenterHeadBBox(nn.Module):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent

    @staticmethod
    def get_bboxes(self, preds_dicts):
        rets = []
        for task_id, preds_dict in enumerate(preds_dicts):
            batch_heatmap = preds_dict["heatmap"].sigmoid()

            batch_reg = preds_dict["reg"]
            batch_hei = preds_dict["height"]

            if self.norm_bbox:
                batch_dim = torch.exp(preds_dict["dim"])
            else:
                batch_dim = preds_dict["dim"]

            if "vel" in preds_dict:
                batch_vel = preds_dict["vel"]
            else:
                batch_vel = None
            rets.extend([batch_heatmap, preds_dict["rot"], batch_hei, batch_dim, batch_vel, batch_reg])
        return rets

    def forward(self, x):
        for type, head in self.parent.heads.items():
            if type == "object":
                ret_dicts = []
                x = head.shared_conv(x)
                for task in head.task_heads:
                    ret_dicts.append(task(x))
                return SubclassCenterHeadBBox.get_bboxes(head, ret_dicts)
            else:
                raise ValueError(f"unsupported head: {type}")


class Voxelization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points0, points1, points2, points3, bsizes):
        with no_jit_trace():
            return Voxelization.voxelize([points0, points1, points2, points3])
    
    @staticmethod
    def symbolic(g, points0, points1, points2, points3, bsizes):
        return g.op("Voxelization", points0, points1, points2, points3, bsizes, outputs=3)

class PillarsScatter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, feats, coords, N):
        batch_size = N.shape[0].item()
        with no_jit_trace():
            N = N[0, 0].item()
            return PillarsScatter.pts_middle_encoder(feats[:N], coords.view(-1, 4)[:N], batch_size)
    
    @staticmethod
    def symbolic(g, feats, coords, N):
        return g.op("PillarsScatter", feats, coords, N, W_i=128, H_i=128)


class SubclassBEVFusion(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.head = SubclassCenterHeadBBox(model)

    # images: 1,N,C,H,W
    # feats:  (A+B)xMxC
    # coords: (A+B)x4     BXYZ
    # sizes:  (A+B)x1
    # camera_intrinsics: Nx3x3
    # camera2lidar:      Nx4x4
    def forward(self, images, feats, coords, N, intervals, geometry, num_intervals, camera_intrinsics, camera2lidar, denorms, camera2ego, img_aug_matrix):
        lidar_backbone = self.model.encoders["lidar"]["backbone"]
        feats = lidar_backbone.pts_voxel_encoder.pfn_layers[0](feats.view(-1, 10, 9), export=True)
        PillarsScatter.pts_middle_encoder = lidar_backbone.pts_middle_encoder
        lidar_bev = PillarsScatter.apply(feats, coords, N)
        
        with no_jit_trace():
            N = 1
            B, C, H, W = map(int, images.size())

        x = self.model.encoders["camera"]["backbone"](images)
        x = self.model.encoders["camera"]["neck"](x)

        if not isinstance(x, torch.Tensor):
            x = x[1]

        BN, C, H, W = map(int, x.size())
        with no_jit_trace():
            lidar2ego = torch.eye(4).view(1, 4, 4).repeat(B, 1, 1).to(x.device)

        x = self.model.encoders["camera"]["vtransform"].export(
            x,
            None,
            camera2ego,
            lidar2ego,
            None,
            None,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            None,
            None,
            intervals,
            geometry,
            num_intervals,
            denorms=denorms
        )
        
        x = self.model.fuser([x, lidar_bev])
        x = self.model.decoder["backbone"](x)
        x = self.model.decoder["neck"](x)
        output = self.head(x)
        return output 

def main():
    args = parse_args()
    print(args)
    if args.precision is None:
        print("Select the precision mode (int8 or fp16)")
        exit()
        
    export_fp16 = True
    if args.precision == 'int8':
        export_fp16 = False
    
    configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(configs), filename=args.config)

    # set cudnn_benchmark
    torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True

    # build the dataloader
    bacth_size = 4
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=bacth_size,
        workers_per_gpu=bacth_size,#cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )
    data_loader.collate_fn = partial(collate_fn, is_return_depth=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model  = torch.load(args.checkpoint).module
    if export_fp16:
        for name, item in model.named_modules():
            if isinstance(item, TensorQuantizer):
                item.disable()

    model = SubclassBEVFusion(model).eval().cuda()
    with torch.no_grad():
        for data in data_loader:
            break

        for key, value in data.items():
            if isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, torch.Tensor):
                        value[i] = item.cuda()
            elif isinstance(value, torch.Tensor):
                data[key] = value.cuda()
                
        global metas
        img = data["img"].squeeze(1)
        camera_intrinsics = data["camera_intrinsics"]
        img_aug_matrix = data["img_aug_matrix"]
        camera2ego = data["camera2ego"]
        lidar2ego = data["lidar2ego"]
        camera2lidar = data["camera2lidar"]
        denorms = data["denorms"]
        lidar_aug_matrix = data["lidar_aug_matrix"]

        metas = data["metas"][0]
        points = data["points"]
        feats, coords, sizes = model.model.voxelize(points)
        
        # max_effective_voxel_count, max_intervals, max_geometry are statistics from the DAIR-V2X-I dataset
        # intervals and geometry are the numbers after filtering. The geometry is related to the number of batches. Here we take 4 as an example.
        max_effective_voxel_count = 8000  
        max_intervals = 10499
        max_geometry  = 1086935
        MAX = bacth_size*max_effective_voxel_count 
        
        lidar_backbone = model.model.encoders["lidar"]["backbone"]
        features = lidar_backbone.pts_voxel_encoder(feats, sizes, coords, do_pfn=False)
        def pad(x):
            t = torch.zeros(MAX, *x.shape[1:], dtype=x.dtype, device=x.device)
            t[:x.size(0)] = x
            return t

        N = torch.tensor([features.size(0)], dtype=torch.int32).view(1, 1).repeat(bacth_size, 1).cuda()
        num_intervals = torch.tensor([1], dtype=torch.int32).view(1, 1).repeat(bacth_size, 1).cuda()
        
        features = pad(features).view(bacth_size, max_effective_voxel_count, cfg.model.encoders.lidar.voxelize.max_num_points, 9)
        coords = pad(coords).view(bacth_size, max_effective_voxel_count, 4)
        
        TensorQuantizer.use_fb_fake_quant = True
        intervals = torch.zeros(bacth_size, int(max_intervals), 3, dtype=torch.int32)
        geometry  = torch.zeros(bacth_size, int(max_geometry), dtype=torch.int32)

        export_file_name = "onemodel-fp16-seq.onnx" if export_fp16 else "onemodel-int8-seq.onnx"
        print(f"Exporting...")

        try:
            torch.onnx.export(
                model,
                (img, features, coords, N, intervals, geometry, num_intervals, camera_intrinsics, camera2lidar, denorms, camera2ego, img_aug_matrix),
                export_file_name,
                input_names=["images", "feats", "coords", "N", "intervals", "geometry", "num_intervals"],
                output_names=["heatmap", "rotation", "height", "dim", "vel", "reg", "heatmap2", "rotation2", "height2", "dim2", "vel2", "reg2"],
                opset_version=13,
                do_constant_folding=True,
                enable_onnx_checker=False,
                verbose=False,
                dynamic_axes={
                    "images": {0: "batch"},
                    "feats": {0: "batch"},
                    "coords": {0: "batch"},
                    "N": {0: "batch"},
                    "intervals": {0: "batch"},
                    "geometry": {0: "batch"},
                    "num_intervals": {0: "batch"},
                }
            )
        except torch.onnx.utils.ONNXCheckerError as e:
            print(f"Ignore checker error: {e}")

        print(f"Export onnx to: {export_file_name} done.")

if __name__ == "__main__":
    main()