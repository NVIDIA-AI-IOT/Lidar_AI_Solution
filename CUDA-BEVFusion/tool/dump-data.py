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

import os
import shutil
import tensor
import argparse
import numpy as np
from torchpack.utils.config import configs
from mmcv import Config
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.utils import recursive_eval
import torch
from mmdet3d.core.bbox.structures import LiDARInstance3DBoxes

def arg_parser():
    parser = argparse.ArgumentParser(description='For bevfusion evaluation on nuScenes dataset.')
    parser.add_argument('--config', dest='config', type=str, default='bevfusion/configs/nuscenes/det/transfusion/secfpn/camera+lidar/resnet50/convfuser.yaml')
    # parser.add_argument('--checkpoint', dest='checkpoint', type=str, default='model/resnet50/bevfusion-det.pth')
    args = parser.parse_args()
    return args

def dump_tensor(args):
    configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(configs), filename=args.config)
    dataset = build_dataset(cfg.data.test)

    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=1,
        dist=False,
        shuffle=False,
    )
    
    data_iter = iter(data_loader)
    for i in range(200):
        data = next(data_iter)
        
        root = f"dump/{i:05d}"
        if not os.path.exists(root):
            os.makedirs(root)

        torch.save(data, f"{root}/example-data.pth")
        metas = data["metas"].data[0]
        token = metas[0]['token']

        with open(f"{root}/token.txt", "w") as f:
            f.write(token)

        names = ["FRONT", "FRONT_RIGHT", "FRONT_LEFT", "BACK", "BACK_LEFT", "BACK_RIGHT"]
        for idx, path in enumerate(data['metas']._data[0][0]['filename']):
            name = os.path.basename(path)
            shutil.copyfile(path, f"{root}/{idx}-{names[idx]}.jpg")

        tensor.save(data["img"].data[0].half(), f"{root}/images.tensor", True)
        tensor.save(data["points"].data[0][0].half(), f"{root}/points.tensor", True)
        tensor.save(data["gt_bboxes_3d"].data[0][0].tensor, f"{root}/gt_bboxes_3d.tensor", True)
        tensor.save(data["gt_labels_3d"].data[0][0], f"{root}/gt_labels_3d.tensor", True)
        tensor.save(data["camera_intrinsics"].data[0], f"{root}/camera_intrinsics.tensor", True)
        tensor.save(data["camera2ego"].data[0], f"{root}/camera2ego.tensor", True)
        tensor.save(data["lidar2ego"].data[0], f"{root}/lidar2ego.tensor", True)
        tensor.save(data["lidar2camera"].data[0], f"{root}/lidar2camera.tensor", True)
        tensor.save(data["camera2lidar"].data[0], f"{root}/camera2lidar.tensor", True)
        tensor.save(data["lidar2image"].data[0], f"{root}/lidar2image.tensor", True)
        tensor.save(data["img_aug_matrix"].data[0], f"{root}/img_aug_matrix.tensor", True)
        tensor.save(data["lidar_aug_matrix"].data[0], f"{root}/lidar_aug_matrix.tensor", True)


if __name__ == "__main__":
    args = arg_parser()

    # if args.dump:
    dump_tensor(args)
