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

import torch
import collections
from mmdet3d.ops import SparseBasicBlock
from mmdet3d.ops import spconv as spconv
from mmdet3d.ops.spconv.modules import SparseSequential
from mmdet3d.ops.spconv.conv import SubMConv3d, SparseConv3d
from mmdet3d.ops.spconv.structure import SparseConvTensor
import numpy as np

def text_format_to_color(text):
    text = text.replace("<red>", "\033[31m")
    text = text.replace("</red>", "\033[0m")
    text = text.replace("<green>", "\033[32m")
    text = text.replace("</green>", "\033[0m")
    text = text.replace("<yellow>", "\033[33m")
    text = text.replace("</yellow>", "\033[0m")
    text = text.replace("<blue>", "\033[34m")
    text = text.replace("</blue>", "\033[0m")
    text = text.replace("<mag>", "\033[35m")
    text = text.replace("</mag>", "\033[0m")
    text = text.replace("<cyan>", "\033[36m")
    text = text.replace("</cyan>", "\033[0m")
    return text

def cprint(*args, **kwargs):
    args = list(args)
    for i, item in enumerate(args):
        if isinstance(item, str):
            args[i] = text_format_to_color(item)
    print(*args, *kwargs)

def fuse_bn_weights(conv_w_OKI, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b):
    NDim = conv_w_OKI.ndim - 2
    permute = [0, NDim+1] + [i+1 for i in range(NDim)]
    conv_w_OIK = conv_w_OKI.permute(*permute)
    # OIDHW
    if conv_b is None:
        conv_b = torch.zeros_like(bn_rm)
    if bn_w is None:
        bn_w = torch.ones_like(bn_rm)
    if bn_b is None:
        bn_b = torch.zeros_like(bn_rm)
    bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)

    conv_w_OIK = conv_w_OIK * (bn_w * bn_var_rsqrt).reshape([-1] + [1] * (len(conv_w_OIK.shape) - 1))
    conv_b = (conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b
    permute = [0,] + [i+2 for i in range(NDim)] + [1,]
    conv_w_OKI = conv_w_OIK.permute(*permute).contiguous()
    return torch.nn.Parameter(conv_w_OKI), torch.nn.Parameter(conv_b)

def fuse_bn(conv, bn):
    """
    Given a conv Module `A` and an batch_norm module `B`, returns a conv
    module `C` such that C(x) == B(A(x)) in inference mode.
    """
    assert(not (conv.training or bn.training)), "Fusion only for eval!"
    conv.weight, conv.bias = fuse_bn_weights(conv.weight.permute(4, 0, 1, 2, 3), conv.bias, bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias)
    conv.weight.data = conv.weight.data.permute(1, 2, 3, 4, 0)

def load_checkpoint(model, file, startsname=None):

    device   = next(model.parameters()).device
    ckpt     = torch.load(file, map_location=device)["state_dict"]
    new_ckpt = ckpt
    
    if startsname is not None:
        new_ckpt = collections.OrderedDict()
        for key, val in ckpt.items():
            if key.startswith(startsname):
                newkey = key[len(startsname)+1:]
                new_ckpt[newkey] = val

    model.load_state_dict(new_ckpt)

def replace_feature(self, feature: torch.Tensor):
    """we need to replace x.features = F.relu(x.features) with x = x.replace_feature(F.relu(x.features))
    due to limit of torch.fx
    """
    # assert feature.shape[0] == self.indices.shape[0], "replaced num of features not equal to indices"
    new_spt = SparseConvTensor(feature, self.indices, self.spatial_shape,
                                self.batch_size, self.grid)
    return new_spt

def new_sparse_basic_block_forward(self):
    def sparse_basic_block_forward(x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))
        return out
    return sparse_basic_block_forward

def fuse_sparse_basic_block(self):
    self.forward = new_sparse_basic_block_forward(self)
    self.conv1.act_type = "ReLU"
    fuse_bn(self.conv1, self.bn1)
    fuse_bn(self.conv2, self.bn2)
    delattr(self, "bn1")
    delattr(self, "bn2")

def layer_fusion(model):

    def set_attr_by_path(m, path, newval):

        def set_attr_by_array(parent, arr):
            if len(arr) == 1: 
                setattr(parent, arr[0], newval)
                return parent

            parent = getattr(parent, arr[0])
            return set_attr_by_array(parent, arr[1:])

        return set_attr_by_array(m, path.split("."))


    for name, module in model.named_modules():
        if isinstance(module, SparseSequential):
            if isinstance(module[0], SubMConv3d) or isinstance(module[0], SparseConv3d):
                c, b, r = [module[i] for i in range(3)]
                fuse_bn(c, b)
                c.act_type = "ReLU"
                set_attr_by_path(model, name, c)
        elif isinstance(module, SparseBasicBlock):
            fuse_sparse_basic_block(module)
        elif isinstance(module, torch.nn.ReLU): 
            module.inplace = False
    return model