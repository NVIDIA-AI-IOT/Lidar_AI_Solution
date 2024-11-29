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

import platform
from mmcv.utils import Registry, build_from_cfg

from mmdet.datasets import DATASETS
from mmdet.datasets.builder import _concat_dataset

if platform.system() != "Windows":
    # https://github.com/pytorch/pytorch/issues/973
    import resource

    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    base_soft_limit = rlimit[0]
    hard_limit = rlimit[1]
    soft_limit = min(max(4096, base_soft_limit), hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))

OBJECTSAMPLERS = Registry("Object sampler")


def build_dataset(cfg, default_args=None):
    from mmdet3d.datasets.dataset_wrappers import CBGSDataset
    from mmdet.datasets.dataset_wrappers import ClassBalancedDataset, ConcatDataset, RepeatDataset

    if "type" in cfg and cfg["type"] == "V2XDataset":
        from mmdet3d.datasets.v2x_dataset import V2XDataset

        ida_aug_conf = {
            "final_dim": cfg["dataset"].final_dim,
            "H": cfg["dataset"].H,
            "W": cfg["dataset"].W,
            "bot_pct_lim": (0.0, 0.0),
            "cams": ["CAM_FRONT"],
            "Ncams": 1,
        }

        img_conf = dict(
            img_mean=cfg["dataset"].img_mean,
            img_std=cfg["dataset"].img_std,
            to_rgb=cfg["dataset"].to_rgb,
        )

        dataset = V2XDataset(
            ida_aug_conf=ida_aug_conf,
            classes=cfg["dataset"].object_classes,
            data_root=cfg["dataset"].dataset_root,
            kitti_root=cfg["dataset"].dataset_kitti_root,
            result_root=cfg["dataset"].result_root,
            info_path=cfg["dataset"].ann_file,
            is_train=cfg["dataset"].is_train,
            use_cbgs=cfg["dataset"].use_cbgs,
            img_conf=img_conf,
            num_sweeps=cfg["dataset"].num_sweeps,
            sweep_idxes=list(),
            key_idxes=list(),
            return_depth=False,
        )
        return dataset
    elif isinstance(cfg, (list, tuple)):
        dataset = ConcatDataset([build_dataset(c, default_args) for c in cfg])
    elif cfg["type"] == "ConcatDataset":
        dataset = ConcatDataset(
            [build_dataset(c, default_args) for c in cfg["datasets"]],
            cfg.get("separate_eval", True),
        )
    elif cfg["type"] == "RepeatDataset":
        dataset = RepeatDataset(build_dataset(cfg["dataset"], default_args), cfg["times"])
    elif cfg["type"] == "ClassBalancedDataset":
        dataset = ClassBalancedDataset(
            build_dataset(cfg["dataset"], default_args), cfg["oversample_thr"]
        )
    elif cfg["type"] == "CBGSDataset":
        dataset = CBGSDataset(build_dataset(cfg["dataset"], default_args))
    elif isinstance(cfg.get("ann_file"), (list, tuple)):
        dataset = _concat_dataset(cfg, default_args)
    else:
        dataset = build_from_cfg(cfg, DATASETS, default_args)

    return dataset