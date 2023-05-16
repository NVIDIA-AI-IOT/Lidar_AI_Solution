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
import torch.optim as optim
from torch.cuda import amp

from tqdm import tqdm
from typing import Callable, Dict
from copy import deepcopy
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from pytorch_quantization import nn as quant_nn
from pytorch_quantization.nn.modules import _utils as quant_nn_utils
from pytorch_quantization import calib
from pytorch_quantization import tensor_quant
from pytorch_quantization import quant_modules
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization.nn.modules import _utils
from absl import logging as quant_logging

from det3d.models.backbones.scn import SparseBasicBlock
from spconv.pytorch.conv import SparseConvolution, SparseConvTensor
from spconv.core import ConvAlgo
from typing import List, Optional, Tuple, Union
from cumm import tensorview as tv
import spconv.pytorch as spconv

# Replace add to quant_add
class QuantAdd(nn.Module, _utils.QuantInputMixin):
    default_quant_desc_input  = tensor_quant.QUANT_DESC_8BIT_PER_TENSOR  
    def __init__(self):
        super().__init__()
        self.init_quantizer(self.default_quant_desc_input)
        
    def forward(self, input1, input2):
        quant_input1 = self._input_quantizer(input1)
        quant_input2 = self._input_quantizer(input2)
        return torch.add(quant_input1, quant_input2)

def insert_quant_add_to_block(self):
    self.quant_add = QuantAdd()
    self.quant_add.__init__()

def quant_add_module(model):
    for name, block in model.named_modules():
        if isinstance(block, SparseBasicBlock):
            insert_quant_add_to_block(block)
# Replace add to quant_add

# Replace spconv to spconv_quant
class SparseConvolutionQunat(SparseConvolution, _utils.QuantMixin):
    default_quant_desc_input  = tensor_quant.QuantDescriptor(num_bits=8, calib_method = 'histogram')
    default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_CONV2D_WEIGHT_PER_CHANNEL
    
    def __init__(self,
                ndim: int,
                in_channels: int,
                out_channels: int,
                kernel_size: Union[int, List[int], Tuple[int, ...]] = 3,
                stride: Union[int, List[int], Tuple[int, ...]] = 1,
                padding: Union[int, List[int], Tuple[int, ...]] = 0,
                dilation: Union[int, List[int], Tuple[int, ...]] = 1,
                groups: int = 1,
                bias: bool = True,
                subm: bool = False,
                output_padding: Union[int, List[int], Tuple[int, ...]] = 0,
                transposed: bool = False,
                inverse: bool = False,
                indice_key: Optional[str] = None,
                algo: Optional[ConvAlgo] = None,
                fp32_accum: Optional[bool] = None,
                record_voxel_count: bool = False,
                act_type: tv.gemm.Activation = tv.gemm.Activation.None_,
                act_alpha: float = 0,
                act_beta: float = 0,
                name=None,
                device=None, 
                dtype=None):
                 
        SparseConvolutionQunat.__init__(self, ndim, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, 
                    bias=bias,
                    subm=subm,
                    output_padding=output_padding,
                    transposed=transposed,
                    inverse=inverse,
                    indice_key=indice_key,
                    algo=algo,
                    fp32_accum=fp32_accum,
                    record_voxel_count=record_voxel_count,
                    act_type=act_type,
                    act_alpha=act_alpha,
                    act_beta=act_beta)

    def _quant(self, input, add_input):
        if input!=None:
            input._features  = self._input_quantizer(input._features)

        if add_input !=None:
            add_input._feature  = self._input_quantizer(add_input._feature)
        

        quant_weight = None
        if self.weight !=None:
            quant_weight = self._weight_quantizer(self.weight)
            
        return (input ,add_input, quant_weight)

    def forward(self, input: SparseConvTensor, add_input: Optional[SparseConvTensor] = None):
        input, add_input, quant_weight = self._quant(input, add_input)
        return self._conv_forward(self.training, input, quant_weight, self.bias, add_input,
            name=self.name, sparse_unique_name=self._sparse_unique_name, act_type=self.act_type,
            act_alpha=self.act_alpha, act_beta=self.act_beta)
    
def transfer_spconv_to_quantization(nninstance : torch.nn.Module, quantmodule):
    quant_instance = quantmodule.__new__(quantmodule)
    for k, val in vars(nninstance).items():
        setattr(quant_instance, k, val)
    def __init__(self):
        if isinstance(self, SparseConvolutionQunat):
            quant_desc_input, quant_desc_weight = quant_instance.default_quant_desc_input, quant_instance.default_quant_desc_weight
            self.init_quantizer(self.default_quant_desc_input, self.default_quant_desc_weight)

    __init__(quant_instance)
    return quant_instance

def quant_sparseconv_module(model):
    def replace_module(module, prefix=""):
        for name in module._modules:
            submodule = module._modules[name]
            submodule_name = name if prefix == "" else prefix + "." + name
            replace_module(submodule, submodule_name)

            if isinstance(submodule,  spconv.SubMConv3d) or isinstance(submodule, spconv.SparseConv3d):
                module._modules[name]  = transfer_spconv_to_quantization(submodule, SparseConvolutionQunat)
    replace_module(model)
# Replace spconv to spconv_quant

# Set the log level
def initialize():
    quant_logging.set_verbosity(quant_logging.ERROR)    

# Calibration of the quantization layer 
def calibrate_model(model : torch.nn.Module, dataloader, device, batch_processor_callback: Callable = None, num_batch=25):

    def compute_amax(model, **kwargs):
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    if isinstance(module._calibrator, calib.MaxCalibrator):
                        module.load_calib_amax()
                    else:
                        module.load_calib_amax(**kwargs)

                    module._amax = module._amax.to(device)
        
    def collect_stats(model, data_loader, device, num_batch=200):
        """Feed data to the network and collect statistics"""
        # Enable calibrators
        model.eval()
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.disable_quant()
                    module.enable_calib()
                else:
                    module.disable()

        # Feed data to the network for collecting stats
        with torch.no_grad():
            for i, datas in tqdm(enumerate(data_loader), total=num_batch, desc="Collect stats for calibrating"):
                batch_processor_callback(model, datas, train_mode=False, return_preds=True, local_rank=0)
                if i >= num_batch:
                    break

        # Disable calibrators
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.enable_quant()
                    module.disable_calib()
                else:
                    module.enable()

    collect_stats(model, dataloader, device, num_batch=num_batch)
    compute_amax(model, method="mse")

# Using the fast calibration mode 
def set_quantizer_fast(module): 
    for name, module in module.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
             if isinstance(module._calibrator, calib.HistogramCalibrator):
                module._calibrator._torch_hist = True 

def print_quantizer_status(module):
    for name, module in module.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            print('TensorQuantizer name:{} disabled staus:{} module:{}'.format(name, module._disabled, module))

def have_quantizer(module):
    for name, module in module.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            return True
    return False

class disable_quantization:
    def __init__(self, model):
        self.model  = model

    def apply(self, disabled=True):
        for name, module in self.model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                module._disabled = disabled

    def __enter__(self):
        self.apply(True)

    def __exit__(self, *args, **kwargs):
        self.apply(False)

class enable_quantization:
    def __init__(self, model):
        self.model  = model

    def apply(self, enabled=True):
        for name, module in self.model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                module._disabled = not enabled

    def __enter__(self):
        self.apply(True)

    def __exit__(self, *args, **kwargs):
        self.apply(False)

def build_sensitivity_profile(model, data_loader_val, dataset_val, eval_model_callback : Callable = None):
    quant_layer_names = []
    for name, module in model.named_modules():
        if name.endswith("_quantizer"):
            print('use quant layer:{}',name)
            module.disable()
            layer_name = name.replace("._input_quantizer", "").replace("._weight_quantizer", "")
            if layer_name not in quant_layer_names:
                quant_layer_names.append(layer_name)
    for i, quant_layer in enumerate(quant_layer_names):
        print("Enable", quant_layer)
        for name, module in model.named_modules():
            if name.endswith("_quantizer") and quant_layer in name:
                module.enable()
                print(F"{name:40}: {module}")
        with torch.no_grad():
            eval_model_callback(model,data_loader_val, dataset_val) 
        for name, module in model.named_modules():
            if name.endswith("_quantizer") and quant_layer in name:
                module.disable()
                print(F"{name:40}: {module}")