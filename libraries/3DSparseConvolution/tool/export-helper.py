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

import sys; sys.path.insert(0, "./tool/CenterPoint")

import pickle
import torch
import argparse
import tensor

from det3d.datasets import build_dataloader, build_dataset
from det3d.models import build_detector
from det3d.torchie import Config
from det3d.torchie.apis.train import example_to_device
from det3d.torchie.trainer import load_checkpoint

def arg_parser():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--checkpoint', type=str, default='tool/checkpoint/epoch_20.pth', help='checkpoint')
    parser.add_argument('--config', type=str, default='tool/CenterPoint/configs/nusc/voxelnet/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z.py', help='config')
    parser.add_argument('--pkl-in', type=str, default='workspace/centerpoint/input.pkl', help='input pickle')
    parser.add_argument('--in-features', type=str, default='workspace/centerpoint/in_features.torch.fp16.tensor', help='input features')
    parser.add_argument('--in-indices', type=str, default='workspace/centerpoint/in_indices.torch.int32.tensor', help='input indices')
    parser.add_argument('--out-dense', type=str, default='workspace/centerpoint/out_dense.torch.fp16.tensor', help='dense output')
    args = parser.parse_args()
    return args

def main(args):
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
    data_batch = next(data_iter)
    example = example_to_device(data_batch, torch.device("cuda"), non_blocking=False)

    assert(len(example.keys()) > 0)
    print("Token: ", example["metadata"][0]["token"])
    assert(len(example["points"]) == 1)

    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    load_checkpoint(model, args.checkpoint, map_location="cpu")

    model.eval().half().cuda()

    with torch.no_grad():
        example = dict(
            features=example['voxels'],
            num_voxels=example["num_points"],
            coors=example["coordinates"],
            batch_size=len(example['points']),
            input_shape=example["shape"][0],
        )

        input_features = model.reader(example["features"], example['num_voxels'])
        input_indices = example["coors"]

        with open(args.pkl_in, 'wb') as handle:
            pickle.dump([ input_features, input_indices, example["input_shape"], example["batch_size"]], handle)

        input_features = input_features.half()
        dense_out, _ = model.backbone(
            input_features, input_indices, example["batch_size"], example["input_shape"]
            )

        tensor.save(input_features, args.in_features)
        tensor.save(input_indices, args.in_indices)
        tensor.save(dense_out, args.out_dense)

if __name__ == "__main__":
    args = arg_parser()
    main(args)