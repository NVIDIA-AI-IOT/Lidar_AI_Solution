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

import sys; sys.path.insert(0, "./CenterPoint")

import os
import copy
import torch
import argparse
import numpy as np

from os import walk
from det3d.torchie import Config
from det3d.torchie.apis.train import example_to_device
from det3d.datasets import build_dataloader, build_dataset

def arg_parser():
    parser = argparse.ArgumentParser(description='For voxelnet evaluation on nuScenes dataset.')
    parser.add_argument('--checkpoint', dest='checkpoint', default='tool/checkpoint/epoch_20.pth',
                        action='store', type=str, help='default : "tool/checkpoint/epoch_20.pth"')
    parser.add_argument('--config', dest='config', type=str, help='default : "nusc_centerpoint_voxelnet_0075voxel_fix_bn_z.py"', action='store',
                        default='CenterPoint/configs/nusc/voxelnet/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z.py')
    parser.add_argument('--work_dir', dest='work_dir', default='data/eval/', action='store', type=str, help='default : "data/eval/"')
    parser.add_argument("--dump", action="store_true")
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()
    return args

def dump_bin(args):
    cfg = Config.fromfile(args.config)
    dataset = build_dataset(cfg.data.val)

    data_loader = build_dataloader(
        dataset,
        batch_size=1,
        workers_per_gpu=1,
        dist=False,
        shuffle=False,
    )
    data_iter = iter(data_loader)

    for i in range(0, 6020):
        print(i)
        data_batch = next(data_iter)
        example = example_to_device(data_batch, torch.device("cuda"), non_blocking=False)

        assert(len(example.keys()) > 0)
        assert(len(example["points"]) == 1)

        points = example["points"][0].cpu().numpy()
        print(points.shape)

        assert(len(example["metadata"]) == 1)
        print(example["metadata"][0]["token"])

        with open("data/nusc_bin/" + example["metadata"][0]["token"] + ".bin", "wb") as f:
            f.write(points.tobytes())

def eval_metrics(args):
    cfg = Config.fromfile(args.config)
    dataset = build_dataset(cfg.data.val)
    pred_dir = "data/prediction"
    filenames = next(walk(pred_dir), (None, None, []))[2]  # [] if no file
    predictions = {}

    for f in filenames:
        token = f[0:-4]
        pred = np.loadtxt(os.path.join(pred_dir, f))

        box3d_lidar = torch.from_numpy(pred[:, 0:9]).to(torch.float32)
        label_preds = torch.from_numpy(pred[:, 9:10]).to(torch.int64)
        scores = torch.from_numpy(pred[:, 10:11]).to(torch.float32)

        predictions[token] = { 'box3d_lidar':box3d_lidar, 'label_preds':label_preds, 'scores':scores, 'metadata': {'num_point_features': 5, 'token': token}}

    # print(predictions)

    result_dict, _ = dataset.evaluation(copy.deepcopy(predictions), output_dir=args.work_dir, testset=False)

    if result_dict is not None:
        for k, v in result_dict["results"].items():
            print(f"Evaluation {k}: {v}")

if __name__ == "__main__":
    args = arg_parser()

    if args.dump:
        dump_bin(args)

    if args.eval:
        eval_metrics(args)