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
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.modules.pooling import _MaxPoolNd, _AdaptiveAvgPoolNd, _AvgPoolNd
from torch.nn.modules.conv import _ConvNd, _ConvTransposeNd
from torch.nn.modules.batchnorm import _BatchNorm
from torch.autograd import Function
from enum import Enum
from typing import Callable, List, Optional, Union
import numpy as np
import inspect
import os
import re
import warnings

g_logger_verbose = False

def set_verbose(enable=True):
    global g_logger_verbose
    g_logger_verbose = enable

def log(*msg):
    if g_logger_verbose:
        stack = inspect.stack()[1]
        name = os.path.basename(stack.filename)
        msg = " ".join(msg)
        print(f"[{name}:{stack.lineno}]: {msg}")


class QuantizationImpl(Function):
    @staticmethod
    def forward(ctx, x, scale, bound):
        scaled_x = x / scale
        ctx.bound = bound
        ctx.save_for_backward(scaled_x)
        return scaled_x.round_().clamp_(-bound, +bound) * scale

    @staticmethod
    def backward(ctx, grad):
        scaled_x = ctx.saved_tensors[0]
        bound    = ctx.bound
        zero = grad.new_zeros(1)
        grad = torch.where((scaled_x > -bound) | (scaled_x < +bound), grad, zero)
        return grad, None, None, None


class Algorithm:
    def __init__(self, name, **kwargs):
        self.name = name
        self.keys = kwargs
        self.keys["name"] = name
        for key in kwargs:
            setattr(self, key, kwargs[key])

    def __repr__(self):
        pack_args = []
        for key in self.keys:
            pack_args.append(f"{key}={self.keys[key]}")
        return "(" + ", ".join(pack_args) + ")"

class QuantizationStyle(Enum):
    PerTensor  = 0
    PerChannel = 1

class QuantizationMethod:
    def __init__(self, bits:int, algo:Algorithm, dim:int, style:QuantizationStyle, once=False):
        self.bits = bits
        self.style = style
        self.algo = algo
        self.dim = dim
        self.once = once
        self.bitbound = int(2**(self.bits - 1) - 1)

    @staticmethod
    def per_tensor(bits:int, algo:Algorithm):
        return QuantizationMethod(bits, algo, 0, QuantizationStyle.PerTensor)
    
    @staticmethod
    def per_channel(bits:int, algo:Algorithm, dim:int, once:bool=True):
        return QuantizationMethod(bits, algo, dim, QuantizationStyle.PerChannel, once)

    def __repr__(self):
        return f"[style={self.style.name}, bits={self.bits}, algo={self.algo}, dim={self.dim}, once={self.once}]"


class CalibMax:
    def __init__(self, method : QuantizationMethod):
        self.collect_datas = []
        self.method = method

    def collect(self, x):
        if self.method.style == QuantizationStyle.PerChannel:
            if len(self.collect_datas) > 0 and self.method.once:
                return
            
            x = x.abs()
            reduce_shape = list(range(len(x.shape)-1, -1, -1))
            del reduce_shape[reduce_shape.index(self.method.dim)]

            for i in reduce_shape:
                if x.shape[i] > 1:
                    x = torch.max(x, dim=i, keepdim=True)[0]

            self.collect_datas.append(x)
        else:
            self.collect_datas.append(x.abs().max())

    def compute(self, eps=1 / (1 << 24)):
        if self.method.style == QuantizationStyle.PerChannel:
            return (self.collect_datas[0] / self.method.bitbound).clamp(eps)
        else:
            return (torch.stack(self.collect_datas).mean() / self.method.bitbound).clamp(eps)


class CalibHistogram:
    def __init__(self, method : QuantizationMethod):
        self.collect_datas = None
        self.method = method

    def collect(self, x):
        if self.method.style == QuantizationStyle.PerChannel:
            raise NotImplementedError("Not implemented")
        else:
            x    = x.float().abs()
            xmax = x.max().item()
            if self.collect_datas is None:
                hist = torch.histc(x, self.method.algo.bins, min=0, max=xmax)
                self.collect_datas = hist, xmax / self.method.algo.bins, xmax, self.method.algo.bins
            else:
                prev_hist, width, prev_xmax, prev_bins = self.collect_datas
                new_xmax = max(prev_xmax, xmax)
                new_bins = max(prev_bins, int(math.ceil(xmax / width)))
                hist = torch.histc(x, new_bins, min=0, max=new_xmax)
                hist[:prev_hist.numel()] += prev_hist
                self.collect_datas = hist, width, new_xmax, new_bins

    def compute(self, eps=1 / (1 << 24)):
        if self.method.style == QuantizationStyle.PerChannel:
            raise NotImplementedError("Not implemented")
        
        assert self.collect_datas is not None, f"maybe not run on collect"
        hist, width, xmax, num_bins = self.collect_datas
        device    = hist.device
        centers   = torch.linspace(width / 2, xmax - width / 2, num_bins, device=device)
        start_bin = 128
        scaled_centers = centers[start_bin:] / self.method.bitbound
        mses      = torch.zeros(len(centers) - start_bin)
        centers   = centers.unsqueeze(1)
        hist      = hist.unsqueeze(1)
        scaled_centers = scaled_centers.unsqueeze(0)
        quant_centers = (centers / scaled_centers).round().clamp_(-self.method.bitbound, +self.method.bitbound) * scaled_centers
        mses = ((quant_centers - centers)**2 * hist).mean(0)
        index = torch.argmin(mses).item() + start_bin
        return (centers[index] / self.method.bitbound).clamp(eps)


class CalibHistogramFast:
    def __init__(self, method : QuantizationMethod):
        self.collect_datas = None
        self.method = method

    def collect(self, x):
        if self.method.style == QuantizationStyle.PerChannel:
            raise NotImplementedError("Not implemented")
        else:
            x    = x.abs()
            xmax = x.max().item()
            if self.collect_datas is None:
                x    = (x / xmax * (self.method.algo.bins - 1)).int()
                hist = torch.histc(x, self.method.algo.bins, min=0, max=self.method.algo.bins - 1)
                self.collect_datas = hist, xmax / (self.method.algo.bins - 1), xmax, self.method.algo.bins
            else:
                prev_hist, width, prev_xmax, prev_bins = self.collect_datas
                new_xmax = max(prev_xmax, xmax)
                new_bins = max(prev_bins, int(math.ceil(xmax / width)))
                x    = (x / new_xmax * (new_bins - 1)).int()
                hist = torch.histc(x, new_bins, min=0, max=new_bins - 1)
                hist[:prev_hist.numel()] += prev_hist
                self.collect_datas = hist, width, new_xmax, new_bins

    def compute(self, eps=1 / (1 << 24)):
        if self.method.style == QuantizationStyle.PerChannel:
            raise NotImplementedError("Not implemented")
        
        assert self.collect_datas is not None, f"maybe not run on collect"
        hist, width, xmax, num_bins = self.collect_datas
        device    = hist.device
        centers   = torch.linspace(width / 2, xmax - width / 2, num_bins, device=device)
        start_bin = 128
        scaled_centers = centers[start_bin:] / self.method.bitbound
        mses      = torch.zeros(len(centers) - start_bin)
        centers   = centers.unsqueeze(1)
        hist      = hist.unsqueeze(1)
        scaled_centers = scaled_centers.unsqueeze(0)
        quant_centers = (centers / scaled_centers).round().clamp_(-self.method.bitbound, +self.method.bitbound) * scaled_centers
        mses = ((quant_centers - centers)**2 * hist).mean(0)
        index = torch.argmin(mses).item() + start_bin
        return (centers[index] / self.method.bitbound).clamp(eps)


class Quantizer(nn.Module):
    def __init__(self, method : QuantizationMethod):
        super().__init__()

        self.enable   = True
        self.collect  = False
        self.method   = method
        self.use_torch_quantize = False
        self.quant    = True

    @property
    def collect(self):
        return self._collect

    @collect.setter
    def collect(self, new_value):
        self._collect = new_value
        if new_value:
            if self.method.algo.name == "max":
                self.collector = CalibMax(self.method)
            elif self.method.algo.name == "histogram":
                self.collector = CalibHistogram(self.method)
            elif self.method.algo.name == "histogram_fast":
                self.collector = CalibHistogramFast(self.method)
        else:
            self.collector = None

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        path = f"{prefix}_scale"
        if path in state_dict:
            self.register_parameter("_scale", nn.Parameter(state_dict[path].cuda(), False))
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def compute(self):
        assert self.collector is not None, "self.collector is None, please run collect data first."
        self.register_parameter("_scale", nn.Parameter(self.collector.compute(), False))

    def forward(self, x):
        if not self.enable:
            return x

        if self._collect:
            self.collector.collect(x.detach())
            return x
        
        if not hasattr(self, "_scale"):
            return x

        if self.use_torch_quantize:
            assert x.dtype == torch.float32, f"Failed to export onnx for {x.dtype}(not supported). Please convert the model to float32 first."
            if self.method.style == QuantizationStyle.PerTensor:
                return torch.fake_quantize_per_tensor_affine(x, self._scale.item(), 0, -self.method.bitbound - 1, self.method.bitbound)
            elif self.method.style == QuantizationStyle.PerChannel:
                scale_sequeeze = self._scale.view(self._scale.numel()).data
                return torch.fake_quantize_per_channel_affine(
                    x, scale_sequeeze, scale_sequeeze.new_zeros(scale_sequeeze.shape, dtype=torch.int32), self.method.dim, -self.method.bitbound - 1, self.method.bitbound)

        if self.quant:
            return QuantizationImpl.apply(x, self._scale.to(x.dtype), self.method.bitbound)
        else:
            return x

    def __repr__(self):
        scale = "NoScale"
        if hasattr(self, "_scale"):
            if self.method.style == QuantizationStyle.PerChannel:
                scale = f"scale=[min={self._scale.min().item()}, max={self._scale.max().item()}]"
            else:
                scale = f"scale={self._scale.item()}"
        return f"Quantizer({scale}, enable={self.enable}, collect={self.collect}, method={self.method}, quant={self.quant})"


class _linker(object):
    def __init__(self, model:nn.Module, module_class):
        self.model = model
        self.module_class = module_class

    def apply(self, fn):
        if isinstance(self.model, self.module_class):
            fn(self.model)

        for name, m in self.model.named_modules():
            if isinstance(m, self.module_class):
                fn(m)
        return self

    def __getattribute__(self, name: str):
        if name == "model":     return object.__getattribute__(self, "model")
        if name == "module_class":    return object.__getattribute__(self, "module_class")

        apply = object.__getattribute__(self, "apply")
        module_class = object.__getattribute__(self, "module_class")
        if module_class == Quantizer:
            if name == "enable":    return apply(lambda m: setattr(m, "enable", True))
            if name == "disable":   return apply(lambda m: setattr(m, "enable", False))
            if name == "collect":   return apply(lambda m: setattr(m, "collect", True))
            if name == "nocollect": return apply(lambda m: setattr(m, "collect", False))
            if name == "compute":   return apply(lambda m: m.compute())
            if name == "export":    return apply(lambda m: setattr(m, "use_torch_quantize", True))
            if name == "noexport":  return apply(lambda m: setattr(m, "use_torch_quantize", False))
            if name == "quant":     return apply(lambda m: setattr(m, "quant", True))
            if name == "noquant":   return apply(lambda m: setattr(m, "quant", False))
        elif module_class == Sparser:
            if name == "sparsity":   return apply(lambda m: setattr(m, "sparsity", True))
            if name == "nosparsity": return apply(lambda m: setattr(m, "sparsity", False))
            if name == "sparsify_weight":   return apply(lambda m: m.sparsify_weight())
            if name == "recompute_mask": return apply(lambda m: m.recompute_mask())
            if name == "every_forward":  return apply(lambda m: setattr(m, "frequency", SparsityMaskFrequency.EveryForward))
            if name == "never_update":   return apply(lambda m: setattr(m, "frequency", SparsityMaskFrequency.NeverUpdate))
        raise AttributeError(f"Can not found attribute: {name} for {module_class}")


# quantizer linker
class linker(_linker):
    enable:  callable = None
    disable: callable = None
    collect: callable = None
    nocollect: callable = None
    compute: callable = None
    export: callable = None
    noexport: callable = None
    quant: callable = None
    noquant: callable = None

    def __init__(self, model:nn.Module):
        super().__init__(model, Quantizer)


# sparsity linker
class slinker(_linker):
    sparsity: callable = None
    nosparsity: callable = None
    sparsify_weight: callable = None
    recompute_mask: callable = None
    every_forward: callable = None
    never_update: callable = None

    def __init__(self, model:nn.Module):
        super().__init__(model, Sparser)


class early:
    def __init__(self, dataloader, num_iter):
        self.dataloader = dataloader

        if hasattr(dataloader, "__len__"):
            self.num_iter = min(num_iter, len(dataloader))
        else:
            self.num_iter = num_iter
    
    def __len__(self):
        return self.num_iter
        
    def __iter__(self):
        for i, obj in enumerate(self.dataloader):
            yield obj

            if i+1 >= self.num_iter:
                break


class _action(object):
    def __init__(self, model, __linker, **actions):
        self.linker = __linker(model)
        self.actions = actions

        # quantization linker
        if __linker == linker:
            self.allow_actions = ["enable", "collect", "export", "quant"]
            self.inverse_action = {"enable": "disable", "collect": "nocollect", "export":"noexport", "quant":"noquant"}
        # sparsity linker
        elif __linker == slinker:
            self.allow_actions = ["sparsity"]
            self.inverse_action = {"sparsity": "nosparsity"}
        else:
            raise NotImplementedError(f"Invalid linker class: {__linker}")
            
        for key in actions:
            assert key in self.allow_actions, f"Unknow action name: {key}, allow actions is: {self.allow_actions}"
        for key in list(self.inverse_action.keys()):
            self.inverse_action[self.inverse_action[key]] = key
    
    def __enter__(self):
        for name in self.actions:
            if self.actions[name]:
                log(f"Do action: linker(model).{name}")
                getattr(self.linker, name)
            else:
                log(f"Do action: linker(model).{self.inverse_action[name]}")
                getattr(self.linker, self.inverse_action[name])
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_value is not None:
            raise exc_value

        for name in self.actions:
            if self.actions[name]:
                log(f"Do action: linker(model).{self.inverse_action[name]}")
                getattr(self.linker, self.inverse_action[name])
            else:
                log(f"Do action: linker(model).{name}")
                getattr(self.linker, name)


# quantization action
class action(_action):
    def __init__(self, model, **actions):
        super().__init__(model, linker, **actions)
        

# sparsity action
class saction(_action):
    def __init__(self, model, **actions):
        super().__init__(model, slinker, **actions)


class collect(torch.no_grad):
    def __init__(self, model):
        super().__init__()
        self.linker = linker(model)
        
    def __enter__(self):
        super().__enter__()
        self.prev_training = self.linker.model.training
        self.linker.model.eval()
        self.linker.enable.collect
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_value is not None:
            raise exc_value
        
        super().__exit__(exc_type, exc_value, exc_traceback)
        self.linker.compute.nocollect

        if self.prev_training:
            self.linker.model.train()

class will_export(torch.no_grad):
    def __init__(self, model):
        super().__init__()
        self.model   = model
        self.linker  = linker(model) if has_quantizer(model) else None
        self.slinker = slinker(model) if has_sparser(model) else None

    def __enter__(self):
        super().__enter__()
        self.prev_training = self.model.training
        self.model.eval()

        if self.linker is not None:
            self.linker.export

        if self.slinker is not None:
            self.slinker.nosparsity

        warnings.filterwarnings("ignore")
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_value is not None:
            raise exc_value
        
        super().__exit__(exc_type, exc_value, exc_traceback)
        if self.linker is not None:
            self.linker.noexport

        if self.slinker is not None:
            self.slinker.sparsity

        if self.prev_training:
            self.model.train()


PER_TENSOR_HISTOGRAM_8BITS = QuantizationMethod.per_tensor(8, Algorithm("histogram", bins=2048))
PER_TENSOR_HISTOGRAM_FAST_8BITS = QuantizationMethod.per_tensor(8, Algorithm("histogram_fast", bins=2048))
PER_CHANNEL_MAX_8BITS      = QuantizationMethod.per_channel(8, Algorithm("max"), 0)

class QTypeInputAndWeight:
    def init_quantizer(self):
        self.input_quantizer_  = Quantizer(PER_TENSOR_HISTOGRAM_FAST_8BITS)
        self.weight_quantizer_ = Quantizer(PER_CHANNEL_MAX_8BITS)

class QTypeInputOnly:
    def init_quantizer(self):
        self.input_quantizer_  = Quantizer(PER_TENSOR_HISTOGRAM_FAST_8BITS)

class QuantConvNd(_ConvNd, QTypeInputAndWeight):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_quantizer()

class QuantConv1d(QuantConvNd, nn.Conv1d):
    def forward(self, x):
        return self._conv_forward(self.input_quantizer_(x), self.weight_quantizer_(self.weight), self.bias)

class QuantConv2d(QuantConvNd, nn.Conv2d):
    def forward(self, x):
        return self._conv_forward(self.input_quantizer_(x), self.weight_quantizer_(self.weight), self.bias)

class QuantConv3d(QuantConvNd, nn.Conv3d):
    def forward(self, x):
        return self._conv_forward(self.input_quantizer_(x), self.weight_quantizer_(self.weight), self.bias)

class QuantConvTransposeNd(_ConvTransposeNd, QTypeInputAndWeight):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_quantizer()
    
    def _conv_forward(self, fn, x, output_size: Optional[List[int]] = None):
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose1d')

        assert isinstance(self.padding, tuple)
        x = self.input_quantizer_(x)
        w = self.weight_quantizer_(self.weight)
        output_padding = self._output_padding(x, output_size, self.stride, self.padding, self.kernel_size, self.dilation)  # type: ignore[arg-type]
        return fn(x, w, self.bias, self.stride, self.padding, output_padding, self.groups, self.dilation)

class QuantConvTranspose1d(QuantConvTransposeNd, nn.ConvTranspose1d):
    def forward(self, x, output_size: Optional[List[int]] = None):
        return super()._conv_forward(F.conv_transpose1d, x, output_size)

class QuantConvTranspose2d(QuantConvTransposeNd, nn.ConvTranspose2d):
    def forward(self, x, output_size: Optional[List[int]] = None):
        return super()._conv_forward(F.conv_transpose2d, x, output_size)

class QuantConvTranspose3d(QuantConvTransposeNd, nn.ConvTranspose3d):
    def forward(self, x, output_size: Optional[List[int]] = None):
        return super()._conv_forward(F.conv_transpose3d, x, output_size)

class QuantLinear(nn.Linear, QTypeInputAndWeight):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_quantizer()
    
    def forward(self, x):
        x = self.input_quantizer_(x)
        w = self.weight_quantizer_(self.weight)
        return F.linear(x, w, self.bias)

class QuantAdd(nn.Module, QTypeInputOnly):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_quantizer()
    
    def forward(self, a, b):
        return torch.add(self.input_quantizer_(a), self.input_quantizer_(b))

class QuantAdaptiveAvgPoolNd(_AdaptiveAvgPoolNd, QTypeInputOnly):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_quantizer()

class QuantAdaptiveAvgPool1d(QuantAdaptiveAvgPoolNd, nn.AdaptiveAvgPool1d):
    def forward(self, x):
        return F.adaptive_avg_pool1d(self.input_quantizer_(x), self.output_size)
    
class QuantAdaptiveAvgPool2d(QuantAdaptiveAvgPoolNd, nn.AdaptiveAvgPool2d):
    def forward(self, x):
        return F.adaptive_avg_pool2d(self.input_quantizer_(x), self.output_size)

class QuantAdaptiveAvgPool3d(QuantAdaptiveAvgPoolNd, nn.AdaptiveAvgPool3d):
    def forward(self, x):
        return F.adaptive_avg_pool3d(self.input_quantizer_(x), self.output_size) 


class QuantMaxPoolNd(_MaxPoolNd, QTypeInputOnly):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_quantizer()
    
class QuantMaxPool1d(QuantMaxPoolNd, nn.MaxPool1d): 
    def forward(self, x):
        x = self.input_quantizer_(x)
        return F.max_pool1d(x, self.kernel_size, self.stride, self.padding, self.dilation, self.ceil_mode, self.return_indices)
    
class QuantMaxPool2d(QuantMaxPoolNd, nn.MaxPool2d): 
    def forward(self, x):
        x = self.input_quantizer_(x)
        return F.max_pool2d(x, self.kernel_size, self.stride, self.padding, self.dilation, self.ceil_mode, self.return_indices)
    
class QuantMaxPool3d(QuantMaxPoolNd, nn.MaxPool3d): 
    def forward(self, x):
        x = self.input_quantizer_(x)
        return F.max_pool3d(x, self.kernel_size, self.stride, self.padding, self.dilation, self.ceil_mode, self.return_indices)
    

class QuantAvgPoolNd(_AvgPoolNd, QTypeInputOnly):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_quantizer()

class QuantAvgPool1d(QuantAvgPoolNd, nn.AvgPool1d): 
    def forward(self, x):
        x = self.input_quantizer_(x)
        return F.avg_pool1d(x, self.kernel_size, self.stride, self.padding, self.ceil_mode, self.count_include_pad)
    
class QuantAvgPool2d(QuantAvgPoolNd, nn.AvgPool2d): 
    def forward(self, x):
        x = self.input_quantizer_(x)
        return F.avg_pool2d(x, self.kernel_size, self.stride, self.padding, self.ceil_mode, self.count_include_pad)
    
class QuantAvgPool3d(QuantAvgPoolNd, nn.AvgPool3d): 
    def forward(self, x):
        x = self.input_quantizer_(x)
        return F.avg_pool3d(x, self.kernel_size, self.stride, self.padding, self.ceil_mode, self.count_include_pad)


quantization_modules_map = [
    (torch.nn, "Conv1d", QuantConv1d),
    (torch.nn, "Conv2d", QuantConv2d),
    (torch.nn, "Conv3d", QuantConv3d),
    (torch.nn, "ConvTranspose1d", QuantConvTranspose1d),
    (torch.nn, "ConvTranspose2d", QuantConvTranspose2d),
    (torch.nn, "ConvTranspose3d", QuantConvTranspose3d),
    (torch.nn, "Linear", QuantLinear),
    (torch.nn, "AdaptiveAvgPool1d", QuantAdaptiveAvgPool1d),
    (torch.nn, "AdaptiveAvgPool2d", QuantAdaptiveAvgPool2d),
    (torch.nn, "AdaptiveAvgPool3d", QuantAdaptiveAvgPool3d),
    (torch.nn, "AvgPool1d", QuantAvgPool1d),
    (torch.nn, "AvgPool2d", QuantAvgPool2d),
    (torch.nn, "AvgPool3d", QuantAvgPool3d),
]

def add_replace_quantization_module(module, name, target):
    global quantization_modules_map
    quantization_modules_map.append((module, name, target))


class quantization_replacement:
    def __init__(self, enable=True):
        self.old_instance = []
        self.enable = enable

    def __enter__(self):
        if self.enable:
            for m, name, target in quantization_modules_map:
                self.old_instance.append((m, name, getattr(m, name)))
                setattr(m, name, target)
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_value is not None:
            raise exc_value
        
        if self.enable:
            for m, name, target in self.old_instance:
                setattr(m, name, target)
        

def ignore_match(ignore_policy : Union[str, List[str], Callable], path : str, target_module: nn.Module=None) -> bool:

    if ignore_policy is None: return False
    if isinstance(ignore_policy, Callable):
        return ignore_policy(path, target_module)

    if isinstance(ignore_policy, str) or isinstance(ignore_policy, List):
        if isinstance(ignore_policy, str):
            ignore_policy = [ignore_policy]

        if path in ignore_policy: return True
        for item in ignore_policy:
            if re.match(item, path):
                return True
    return False


def _replace_modules(model: nn.Module, modules_map: list, ignore_policy: [Callable, List[str]]=None, replace_type=None):
    select_modules = []
    for target, target_module in model.named_modules():
        if ignore_match(ignore_policy, target, target_module):
            continue

        for old_cls, new_cls in modules_map:
            if isinstance(target_module, old_cls):
                select_modules.append([target_module, target, new_cls])
                break

    for target_module, target, new_cls in select_modules:
        new_module = new_cls.__new__(new_cls)        
        if replace_type == "Quantization":
            for k, val in vars(target_module).items():   # Conv2d to QuantConv2d
                setattr(new_module, k, val) 
            new_module.init_quantizer()
        elif replace_type == "Sparsity":
            for k, val in vars(target_module).items():   # Conv2d to SparseConv2d
                setattr(new_module, k, val)
            new_module.init_sparse()
        elif replace_type == "InverseQuantization":      # QuantConv2d to Conv2d
            for k, val in vars(target_module).items():
                if k in ["input_quantizer_", "weight_quantizer_"]:
                    continue

                if k == "_modules":
                    for subkey in ["input_quantizer_", "weight_quantizer_"]:
                        if subkey in val:
                            del val[subkey]

                setattr(new_module, k, val)
        elif replace_type == "InverseSparsity":          # SparseConv2d to Conv2d
            for k, val in vars(target_module).items():
                if k in ["frequency", "sparsity", "N", "M", "mask"]:
                    continue

                setattr(new_module, k, val)
        else:
            raise NotImplementedError(f"Unknow replace type: {replace_type}")

        atoms = target.split(".")
        parent = model.get_submodule(".".join(atoms[:-1]))
        item  = atoms[-1]
        setattr(parent, item, new_module)
    return model


def replace_quantization_modules(model: nn.Module, ignore_policy: [Callable, List[str]]=None):
    preper_modules_map = []
    for m, name, target in quantization_modules_map:
        preper_modules_map.append([getattr(m, name), target])
    return _replace_modules(model, preper_modules_map, ignore_policy, "Quantization")


def _has_some_module(model : nn.Module, module_class):
    for name, m in model.named_modules():
        if isinstance(m, module_class):
            return True
    return False


def has_quantizer(model : nn.Module):
    return _has_some_module(model, Quantizer)


def has_sparser(model : nn.Module):
    return _has_some_module(model, Sparser)


def ismodule(model, node, m):
    if node.op != "call_module":
        return False
    
    return isinstance(model.get_submodule(node.target), m)

def get_call_conv(model, node):
    if node.op == "call_function" and node.target.__name__ in ["conv2d", "conv1d", "conv3d"]:
        conv_target = ".".join(node.args[1].target.split(".")[:-1])
        conv_module = model.get_submodule(conv_target)
        return conv_module

def isconv_bn_or_conv(model, node):
    if ismodule(model, node, _ConvNd):
        return True

    if ismodule(model, node, _BatchNorm):
        return ismodule(model, node.args[0], _ConvNd)
    return False


def quantization_transform_by_fx(model : nn.Module, concrete_args=dict(), ignore_policy: [Callable, List[str]]=None) -> nn.Module:
    import torch.fx
    traced : torch.fx.GraphModule = torch.fx.symbolic_trace(model, concrete_args=concrete_args)
    
    preper_modules_map = []
    for m, name, target in quantization_modules_map:
        preper_modules_map.append([getattr(m, name), target])

    select_modules = []
    for node in traced.graph.nodes:
        if node.op == 'call_module':
            if ignore_match(ignore_policy, node.target):
                continue

            target_module = traced.get_submodule(node.target)
            for old_cls, new_cls in preper_modules_map:
                if isinstance(target_module, old_cls):
                    select_modules.append([target_module, node.target, new_cls, node])
                    break

    for target_module, target, new_cls, node in select_modules:
        quant_module = new_cls.__new__(new_cls)
        for k, val in vars(target_module).items():
            setattr(quant_module, k, val)
        
        quant_module.init_quantizer()
        with traced.graph.inserting_after(node):
            traced.add_submodule(target, quant_module)
            new_node = traced.graph.call_module(target, args=node.args)
            node.replace_all_uses_with(new_node)
            traced.graph.erase_node(node)

    # x = bn(conv(x)) + quantizer(identity)
    for node in traced.graph.nodes:
        if node.op == 'call_function':
            if node.target.__name__ == "add":
                ia = isconv_bn_or_conv(model, node.args[0])
                ib = isconv_bn_or_conv(model, node.args[1])
                if ia or ib:
                    if ia:
                        qinput = node.args[1]
                    else:
                        qinput = node.args[0]

                    with traced.graph.inserting_before(node):
                        new_node_name = f"additional_quants.{node.name}.input_quantizer_"
                        traced.add_submodule(new_node_name, Quantizer(PER_TENSOR_HISTOGRAM_8BITS))
                        new_node = traced.graph.call_module(new_node_name, args=(qinput,))
                        node.replace_input_with(qinput, new_node)

    traced.graph.lint()
    traced.recompile()
    return traced


def apply_connect_rules(model : nn.Module, concrete_args=dict()):

    # Rule1: For some operators[MaxPool, AvgPool, ], they don't modify the distribution of values. Their inputs and outputs should use the same quantizer object.
    # Rule2: Also need to ensure that the same tensor is used at different operator inputs, using the same quantizer.

    import torch.fx
    with action(model, enable=False):
        log(f"Do torch.fx.symbolic_trace")
        traced : torch.fx.GraphModule = torch.fx.symbolic_trace(model, concrete_args=concrete_args)

    module_name_map = dict()
    for name, m in model.named_modules():
        module_name_map[m] = name

    def connect_io_quantizer(node, enum_args, rule_name):
        major = None
        for qnode in list(node.users.keys()):
            qnode_module = get_call_conv(model, qnode)
            if qnode_module is None or not isinstance(qnode_module, QuantConvNd):
                continue
            
            if major is None:
                major = qnode_module
            else:
                log(f"Apply rule({rule_name}) for: {module_name_map[qnode_module]}, {module_name_map[major]}")
                qnode_module.input_quantizer_ = major.input_quantizer_

        for subnode in enum_args:
            for subnode_user in subnode.users.keys():
                if subnode_user == node:
                    continue

                subconv_module = get_call_conv(model, subnode_user)
                if subconv_module is None or not isinstance(subconv_module, QuantConvNd):
                    continue

                log(f"Apply rule({rule_name}) for: {module_name_map[subconv_module]}, {module_name_map[major]}")
                subconv_module.input_quantizer_ = major.input_quantizer_

    for node in traced.graph.nodes:
        if node.op == "call_module":
            if ismodule(model, node, _MaxPoolNd):
                connect_io_quantizer(node, node.args, "MaxPool")
            elif ismodule(model, node, _AvgPoolNd):
                connect_io_quantizer(node, node.args, "AvgPool")
        elif node.op == "call_function":
            if node.target.__name__ == "cat":
                connect_io_quantizer(node, node.args[0], "torch.cat")


### Sparsity
def obtain_sparsity_mask(weight, N=2, M=4):

    if len(weight.shape) == 2:
        O, I = weight.shape
        weight = weight.detach().reshape(-1, M)
        index  = torch.argsort(weight.abs(), dim=1)[:, :int(M-N)]
        mask = torch.ones(weight.shape, device=weight.device, dtype=weight.dtype)
        return mask.scatter_(dim=1, index=index, value=0).reshape(O, I)

    O, I, H, W = weight.shape
    weight = weight.detach().permute(0, 2, 3, 1).reshape(-1, M)
    index  = torch.argsort(weight.abs(), dim=1)[:, :int(M-N)]

    mask = torch.ones(weight.shape, device=weight.device, dtype=weight.dtype)
    mask = mask.scatter_(dim=1, index=index, value=0).reshape(O, H, W, I)
    return mask.permute(0, 3, 1, 2).contiguous()

class SparsifyImpl(Function):
    @staticmethod
    def forward(ctx, x, mask, coeff=2e-4):
        ctx.coeff = coeff
        ctx.mask  = mask
        ctx.save_for_backward(x)
        return x * mask

    @staticmethod
    def backward(ctx, grad):
        return grad + ctx.coeff * (1 - ctx.mask) * ctx.saved_tensors[0], None, None

class SparsityMaskFrequency(Enum):
    EveryForward = 0
    NeverUpdate  = 1

class Sparser(nn.Module):
    def __init__(self):
        super().__init__()
        self.init_sparse()
    
    def init_sparse(self):
        self.frequency = SparsityMaskFrequency.EveryForward
        self.sparsity  = True
        self.N = 2
        self.M = 4
        self.recompute_mask()

    def recompute_mask(self):
        self.mask = obtain_sparsity_mask(self.weight, N=self.N, M=self.M)
        #self.mask = obtain_sparsity_mask(self.weight.cuda(), N=self.N, M=self.M).to(self.weight.device)
    def set_mask(self, mask):
        self.mask = mask

    def sparsify_weight(self):
        self.weight.data = SparsifyImpl.apply(self.weight.data, self.mask)

    def forward(self, x):
        if not self.sparsity:
            return self._conv_forward(x, self.weight, self.bias)

        if self.frequency == SparsityMaskFrequency.EveryForward:
            self.recompute_mask()

        return self._conv_forward(x, SparsifyImpl.apply(self.weight, self.mask), self.bias)

class SparseConvNd(_ConvNd, Sparser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.recompute_mask()

    def __repr__(self):
        return super().__repr__()[:-1] + f", frequency={self.frequency.name}, sparsity={self.sparsity})"

class SparseConv1d(SparseConvNd, nn.Conv1d): 
    def forward(self, x):
        return Sparser.forward(self, x)

class SparseConv2d(SparseConvNd, nn.Conv2d): 
    def forward(self, x):
        return Sparser.forward(self, x)
    
class SparseConv3d(SparseConvNd, nn.Conv3d): 
    def forward(self, x):
        return Sparser.forward(self, x)

class SparseLinear(nn.Linear, Sparser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.recompute_mask()

    def __repr__(self):
        return super().__repr__()[:-1] + f", frequency={self.frequency.name}, sparsity={self.sparsity})"
    
    def forward(self, x):
        if not self.sparsity:
            return F.linear(x, self.weight, self.bias)
        
        if self.frequency == SparsityMaskFrequency.EveryForward:
            self.recompute_mask()
            
        w = SparsifyImpl.apply(self.weight, self.mask)
        return F.linear(x, w, self.bias)


def allow_sparsity_with_best_practice(name, weight_shape):
    shape_str = " x ".join(list(map(str, weight_shape)))
    if weight_shape[1] % 4 != 0:
        log(f"Ingore sparsity for {name}, {shape_str}, due to weight.shape(1)[{int(weight_shape[1])} % 4 != 0]")
        return False

    shape = list(weight_shape)
    if len(shape) == 2:
        shape += [1, 1]

    RS = np.prod(shape[2:])
    if RS > 32:
        log(f"Ingore sparsity for {name}, {shape_str} due to RS [{RS}] > 32")
        return False

    CRS = np.prod(shape[1:])
    if RS > 1 and CRS < 512 or RS == 1 and CRS < 4096:
        log(f"Ingore sparsity for {name}, {shape_str} due to RS[{RS}] > 1 and CRS[{CRS}] < 512 or RS == 1 and CRS < 4096")
        return False

    if name.find("rbr_reparam") != -1:
        log(f"Warning💡, Suspected re-parameterization module found({name}, {shape_str}). For re-parameterization modules, you must handle them specially and use set_mask/SparsityMaskFrequency to manage masks manually.")

    log(f"Enable sparsity: {name}, {shape_str}")
    return True

sparsity_modules_map = [
    (torch.nn.Conv1d, SparseConv1d),
    (torch.nn.Conv2d, SparseConv2d),
    (torch.nn.Conv3d, SparseConv3d),
    (torch.nn.Linear, SparseLinear),
]
def replace_sparsity_modules(model: nn.Module, ignore_policy: [Callable, List[str]]=None):
    def ignore_policy_internal(path, module):
        if not isinstance(module, _ConvNd) and not isinstance(module, nn.Linear):
            return True
        return not allow_sparsity_with_best_practice(path, module.weight.shape) and not ignore_match(ignore_policy, path, module)

    return _replace_modules(model, sparsity_modules_map, ignore_policy_internal, "Sparsity")


def remove_sparsity_modules(model: nn.Module):
    inverse_sparsity_modules_map = [(b, a) for a, b in sparsity_modules_map]
    return _replace_modules(model, inverse_sparsity_modules_map, None, "InverseSparsity")


def remove_quant_modules(model: nn.Module):
    inverse_quant_modules_map = [(c, getattr(a, b)) for a, b, c in quantization_modules_map]
    return _replace_modules(model, inverse_quant_modules_map, None, "InverseQuantization")