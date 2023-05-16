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
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm
from typing import Callable
from absl import logging as quant_logging

from torch.cuda import amp
from torch.nn.parameter import Parameter
from pytorch_quantization import nn as quant_nn
from pytorch_quantization.nn.modules import _utils as quant_nn_utils
from pytorch_quantization import calib
from pytorch_quantization import tensor_quant
from pytorch_quantization import quant_modules
from pytorch_quantization.tensor_quant import QuantDescriptor
from mmdet3d.ops import spconv, SparseBasicBlock
import mmcv.cnn.bricks.wrappers

class QuantConcat(torch.nn.Module):
    def __init__(self, quantization =True):
        super().__init__()

        if quantization:
            self._input_quantizer = quant_nn.TensorQuantizer(QuantDescriptor(num_bits=8, calib_method="histogram"))
            self._input_quantizer._calibrator._torch_hist = True
            self._fake_quant = True
        self.quantization = quantization

    def forward(self, x,  y):
        if self.quantization:
            return torch.cat([self._input_quantizer(x), self._input_quantizer(y)], dim=1)
        return torch.cat([x, y], dim=1)

class QuantAdd(torch.nn.Module):
    def __init__(self, quantization = True):
        super().__init__()
  
        if quantization:
            self._input_quantizer = quant_nn.TensorQuantizer(QuantDescriptor(num_bits=8, calib_method="histogram"))
            self._input_quantizer._calibrator._torch_hist = True
            self._fake_quant = True
        self.quantization = quantization
              
    def forward(self, input1, input2):
        if self.quantization:
            return torch.add(self._input_quantizer(input1), self._input_quantizer(input2))
        return torch.add(input1, input2)

class SparseConvolutionQunat(spconv.conv.SparseConvolution, quant_nn_utils.QuantMixin):
    default_quant_desc_input  = tensor_quant.QuantDescriptor(num_bits=8, calib_method = 'histogram')
    default_quant_desc_weight = tensor_quant.QuantDescriptor(num_bits=8, axis=(4))  
    def __init__(
        self,
        ndim,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        subm=False,
        output_padding=0,
        transposed=False,
        inverse=False,
        indice_key=None,
        fused_bn=False,
    ):
                 
        super(spconv.conv.SparseConvolution, self).__init__(self,
                                        ndim,
                                        in_channels,
                                        out_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=0,
                                        dilation=1,
                                        groups=1,
                                        bias=True,
                                        subm=False,
                                        output_padding=0,
                                        transposed=False,
                                        inverse=False,
                                        indice_key=None,
                                        fused_bn=False,)

    def forward(self, input):
        if input!=None:
            input.features  = self._input_quantizer(input.features)

        if self.weight !=None:
            quant_weight = self._weight_quantizer(self.weight)

        self.weight = Parameter(quant_weight)
        return super().forward(input) 
 
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

def quantize_sparseconv_module(model):
    def replace_module(module, prefix=""):
        for name in module._modules:
            submodule = module._modules[name]
            submodule_name = name if prefix == "" else prefix + "." + name
            replace_module(submodule, submodule_name)

            if isinstance(submodule,  spconv.SubMConv3d) or isinstance(submodule, spconv.SparseConv3d):
                module._modules[name]  = transfer_spconv_to_quantization(submodule, SparseConvolutionQunat)
    replace_module(model)
   

def quantize_add_module(model):
    for name, block in model.named_modules():
        if isinstance(block, SparseBasicBlock):
            block.quant_add = QuantAdd()

'''
Quantize the lidar backbone

'''
def quantize_encoders_lidar_branch(model_lidar_backbone):
    quantize_sparseconv_module(model_lidar_backbone)
    quantize_add_module(model_lidar_backbone)
   
    
'''
Quantize the camera branches

'''
def quantize_encoders_camera_branch(model_camera_branch):
    quantize_camera_backbone(model_camera_branch.backbone)  
    quantize_camera_neck(model_camera_branch.neck)
    quantize_camera_vtransform(model_camera_branch.vtransform)
    
    '''
    Make all inputs of each concat have the same scale
    Improved performance when using TensorRT forward
    '''
    major = model_camera_branch.backbone.layer3[0].conv1._input_quantizer
    model_camera_branch.neck.quant_concat1._input_quantizer = major
    model_camera_branch.neck.lateral_convs[0].conv._input_quantizer = major
    model_camera_branch.backbone.layer3[0].downsample[0]._input_quantizer = major
    
    major = model_camera_branch.backbone.layer4[0].conv1._input_quantizer
    model_camera_branch.neck.quant_concat0._input_quantizer = major
    model_camera_branch.neck.lateral_convs[1].conv._input_quantizer = major
    model_camera_branch.backbone.layer4[0].downsample[0]._input_quantizer = major
    
    
def transfer_torch_to_quantization(nninstance : torch.nn.Module, quantmodule):

    quant_instance = quantmodule.__new__(quantmodule)
    for k, val in vars(nninstance).items():
        setattr(quant_instance, k, val)

    def __init__(self):
        quant_desc_input, quant_desc_weight = quant_nn_utils.pop_quant_desc_in_kwargs(self.__class__)
        if isinstance(self, quant_nn_utils.QuantInputMixin):
            #quant_desc_input = quant_nn_utils.pop_quant_desc_in_kwargs(self.__class__, input_only=True)
            self.init_quantizer(quant_desc_input)

            # Turn on torch_hist to enable higher calibration speeds
            if isinstance(self._input_quantizer._calibrator, calib.HistogramCalibrator):
                self._input_quantizer._calibrator._torch_hist = True
        else:
            self.init_quantizer(quant_desc_input, quant_desc_weight)

            # Turn on torch_hist to enable higher calibration speeds
            if isinstance(self._input_quantizer._calibrator, calib.HistogramCalibrator):
                self._input_quantizer._calibrator._torch_hist = True
                self._weight_quantizer._calibrator._torch_hist = True

    __init__(quant_instance)
    return quant_instance

def replace_to_quantization_module(model : torch.nn.Module):

    module_dict = {}
    for entry in quant_modules._DEFAULT_QUANT_MAP:
        module = getattr(entry.orig_mod, entry.mod_name)
        module_dict[id(module)] = entry.replace_mod

    def recursive_and_replace_module(module, prefix=""):
        for name in module._modules:
            submodule = module._modules[name]
            path      = name if prefix == "" else prefix + "." + name
            recursive_and_replace_module(submodule, path)

            submodule_id = id(type(submodule))
            if submodule_id in module_dict:  
                module._modules[name] = transfer_torch_to_quantization(submodule, module_dict[submodule_id])

    recursive_and_replace_module(model)
    
def quantize_camera_vtransform(model_camera_vtreansform):
    replace_to_quantization_module(model_camera_vtreansform.dtransform) 
    replace_to_quantization_module(model_camera_vtreansform.depthnet) 
    
def quantize_decoder(model_decoder):
    replace_to_quantization_module(model_decoder) 

class hook_generalized_lss_fpn_forward:
    def __init__(self, obj):
        self.obj = obj
    
    def __call__(self, inputs):
        self = self.obj

        # upsample -> cat -> conv1x1 -> conv3x3
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [inputs[i + self.start_level] for i in range(len(inputs))]

        # build top-down path
        used_backbone_levels = len(laterals) - 1
        ic = 0
        for i in range(used_backbone_levels - 1, -1, -1):
            x = F.interpolate(
                laterals[i + 1],
                size=list(map(int, laterals[i].shape[2:])),
                **self.upsample_cfg,
            )
            if ic == 0:
                laterals[i] = self.quant_concat0(laterals[i], x) #torch.cat([laterals[i], x], dim=1)
            else:
                laterals[i] = self.quant_concat1(laterals[i], x)
            ic += 1
            laterals[i] = self.lateral_convs[i](laterals[i])
            laterals[i] = self.fpn_convs[i](laterals[i])

        # build outputs
        outs = [laterals[i] for i in range(used_backbone_levels)]
        return tuple(outs)

def quantize_camera_neck(model_camera_neck):  
    replace_to_quantization_module(model_camera_neck)    
    model_camera_neck.quant_concat0   =  QuantConcat()
    model_camera_neck.quant_concat1   =  QuantConcat()
    model_camera_neck.forward = hook_generalized_lss_fpn_forward(model_camera_neck)
    
        
class hook_bottleneck_forward:
    def __init__(self, obj):
        self.obj = obj

    def __call__(self, x):

        self = self.obj
        identity = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        
        if hasattr(self, "residual_quantizer"):
            identity = self.residual_quantizer(identity)
        
        out += identity
        out = self.relu(out)
        return out

def quantize_camera_backbone(model_camera_backbone):
    replace_to_quantization_module(model_camera_backbone)
    for name, bottleneck in model_camera_backbone.named_modules():
        if bottleneck.__class__.__name__ == "Bottleneck":
            print(f"Add QuantAdd to {name}")
            if bottleneck.downsample is not None:
                bottleneck.downsample[0]._input_quantizer = bottleneck.conv1._input_quantizer
                bottleneck.residual_quantizer = quant_nn.TensorQuantizer(quant_nn.QuantConv2d.default_quant_desc_input)
            else:
                bottleneck.residual_quantizer = bottleneck.conv1._input_quantizer
            bottleneck.forward = hook_bottleneck_forward(bottleneck)
  

def calibrate_model(model : torch.nn.Module, dataloader, device, batch_processor_callback: Callable = None, num_batch=1):

    def compute_amax(model, **kwargs):
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    if isinstance(module._calibrator, calib.MaxCalibrator):
                        module.load_calib_amax(strict=False)
                    else:
                        module.load_calib_amax(strict=False, **kwargs)

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

        iter_count = 0 
        for data in tqdm(data_loader, total=num_batch, desc="Collect stats for calibrating"):
            with torch.no_grad():
                result = model(return_loss=False, rescale=True, **data)
            iter_count += 1
            if iter_count >num_batch:
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

def print_quantizer_status(module):
    for name, module in module.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            print('TensorQuantizer name:{} disabled staus:{} module:{}'.format(name, module._disabled, module))

def have_quantizer(module):
    for name, module in module.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            return True
    return False

def set_quantizer_fast(module): 
    for name, module in module.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
             if isinstance(module._calibrator, calib.HistogramCalibrator):
                module._calibrator._torch_hist = True 

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
                
def initialize():
    quant_logging.set_verbosity(quant_logging.ERROR)
    quant_desc_input = QuantDescriptor(calib_method="histogram")

    quant_modules._DEFAULT_QUANT_MAP.append(
        quant_modules._quant_entry(mmcv.cnn.bricks.wrappers, "ConvTranspose2d", quant_nn.QuantConvTranspose2d)
    )

    for item in quant_modules._DEFAULT_QUANT_MAP:
        item.replace_mod.set_default_quant_desc_input(quant_desc_input)

    quant_logging.set_verbosity(quant_logging.ERROR) 
