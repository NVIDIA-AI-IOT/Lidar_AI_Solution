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

from torchpack.utils.config import configs
from mmcv import Config
from mmcv.runner.fp16_utils import auto_fp16
from mmdet3d.models import build_model
from mmdet3d.utils import recursive_eval
from mmcv.runner import wrap_fp16_model
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import lean.quantize as quantize
from pytorch_quantization.nn.modules.tensor_quantizer import TensorQuantizer


class SubclassHeadBBox(nn.Module):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent

        # ["batch_score", "batch_rot", "batch_dim", "batch_center", "batch_height", "batch_vel"]
        self.classes_eye  = nn.Parameter(torch.eye(parent.heads.object.num_classes).float())

    @staticmethod
    @auto_fp16(apply_to=("inputs", "classes_eye"))
    def head_forward(self, inputs, classes_eye):
        """Forward function for CenterPoint.
        Args:
            inputs (torch.Tensor): Input feature map with the shape of
                [B, 512, 128(H), 128(W)]. (consistent with L748)
        Returns:
            list[dict]: Output results for tasks.
        """
        batch_size = int(inputs.shape[0])
        lidar_feat = self.shared_conv(inputs)

        #################################
        # image to BEV
        #################################
        lidar_feat_flatten = lidar_feat.view(
            batch_size, int(lidar_feat.shape[1]), -1
        )  # [BS, C, H*W]
        bev_pos = self.bev_pos.to(lidar_feat.dtype).repeat(batch_size, 1, 1).to(lidar_feat.device)

        #################################
        # image guided query initialization
        #################################
        dense_heatmap = self.heatmap_head(lidar_feat)
        heatmap = dense_heatmap.detach().sigmoid()
        padding = self.nms_kernel_size // 2
        local_max = torch.zeros_like(heatmap)
        # equals to nms radius = voxel_size * out_size_factor * kenel_size
        local_max_inner = F.max_pool2d(
            heatmap, kernel_size=self.nms_kernel_size, stride=1, padding=0
        )
        local_max[:, :, padding:(-padding), padding:(-padding)] = local_max_inner
        ## for Pedestrian & Traffic_cone in nuScenes
        if self.test_cfg["dataset"] == "nuScenes":
            # local_max[
            #     :,
            #     8,
            # ] = F.max_pool2d(heatmap[:, 8], kernel_size=1, stride=1, padding=0)
            # local_max[
            #     :,
            #     9,
            # ] = F.max_pool2d(heatmap[:, 9], kernel_size=1, stride=1, padding=0)
            local_max[:, 8] = heatmap[:, 8]
            local_max[:, 9] = heatmap[:, 9]
        elif self.test_cfg["dataset"] == "Waymo":  # for Pedestrian & Cyclist in Waymo
            # local_max[
            #     :,
            #     1,
            # ] = F.max_pool2d(heatmap[:, 1], kernel_size=1, stride=1, padding=0)
            # local_max[
            #     :,
            #     2,
            # ] = F.max_pool2d(heatmap[:, 2], kernel_size=1, stride=1, padding=0)
            local_max[:, 1] = heatmap[:, 1]
            local_max[:, 2] = heatmap[:, 2]
        heatmap = heatmap * (heatmap == local_max)
        heatmap = heatmap.view(batch_size, int(heatmap.shape[1]), -1)

        # top #num_proposals among all classes
        # top_proposals = heatmap.view(batch_size, -1).argsort(dim=-1, descending=True)[
        #     ..., : self.num_proposals
        # ]
        top_proposals = heatmap.view(batch_size, -1).topk(k=self.num_proposals, dim=-1, largest=True)[1]
        top_proposals_class = top_proposals // int(heatmap.shape[-1])
        top_proposals_index = top_proposals % int(heatmap.shape[-1])
        query_feat = lidar_feat_flatten.gather(
            index=top_proposals_index[:, None, :].expand(
                -1, lidar_feat_flatten.shape[1], -1
            ),
            dim=-1,
        )
        self.query_labels = top_proposals_class

        # add category embedding
        # self.one_hot = F.one_hot(top_proposals_class, num_classes=self.num_classes).permute(
        #     0, 2, 1
        # ).half()
        self.one_hot = classes_eye.index_select(0, top_proposals_class.view(-1))[None].permute(
            0, 2, 1
        )
        query_cat_encoding = self.class_encoding(self.one_hot)
        query_feat += query_cat_encoding

        query_pos = bev_pos.gather(
            index=top_proposals_index[:, None, :]
            .permute(0, 2, 1)
            .expand(-1, -1, bev_pos.shape[-1]),
            dim=1,
        )

        #################################
        # transformer decoder layer (LiDAR feature as K,V)
        #################################
        ret_dicts = []
        for i in range(self.num_decoder_layers):
            prefix = "last_" if (i == self.num_decoder_layers - 1) else f"{i}head_"

            # Transformer Decoder Layer
            # :param query: B C Pq    :param query_pos: B Pq 3/6
            query_feat = self.decoder[i](
                query_feat, lidar_feat_flatten, query_pos, bev_pos
            )

            # Prediction
            res_layer = self.prediction_heads[i](query_feat)
            res_layer["center"] = res_layer["center"] + query_pos.permute(0, 2, 1)
            first_res_layer = res_layer
            ret_dicts.append(res_layer)

            # for next level positional embedding
            query_pos = res_layer["center"].detach().clone().permute(0, 2, 1)

        #################################
        # transformer decoder layer (img feature as K,V)
        #################################
        ret_dicts[0]["query_heatmap_score"] = heatmap.gather(
            index=top_proposals_index[:, None, :].expand(-1, self.num_classes, -1),
            dim=-1,
        )  # [bs, num_classes, num_proposals]
        ret_dicts[0]["dense_heatmap"] = dense_heatmap

        if self.auxiliary is False:
            # only return the results of last decoder layer
            return ret_dicts[-1]

        # return all the layer's results for auxiliary superivison
        new_res = {}
        for key in ret_dicts[0].keys():
            if key not in ["dense_heatmap", "dense_heatmap_old", "query_heatmap_score"]:
                new_res[key] = torch.cat(
                    [ret_dict[key] for ret_dict in ret_dicts], dim=-1
                )
            else:
                new_res[key] = ret_dicts[0][key]
        return new_res

    def get_bboxes(self, preds_dict, one_hot):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
        Returns:
            list[list[dict]]: Decoded bbox, scores and labels for each layer & each batch
        """
        # batch_score = preds_dict["heatmap"][..., -self.num_proposals :].sigmoid()
        batch_score = preds_dict["heatmap"].sigmoid()
        # if self.loss_iou.loss_weight != 0:
        #    batch_score = torch.sqrt(batch_score * preds_dict['iou'][..., -self.num_proposals:].sigmoid())
        # one_hot = F.one_hot(
        #     query_labels, num_classes=num_classes
        # ).permute(0, 2, 1)
        batch_score = batch_score * preds_dict["query_heatmap_score"] * one_hot
        # batch_center = preds_dict["center"][..., -self.num_proposals :]
        # batch_height = preds_dict["height"][..., -self.num_proposals :]
        # batch_dim = preds_dict["dim"][..., -self.num_proposals :]
        # batch_rot = preds_dict["rot"][..., -self.num_proposals :]
        # batch_vel = None
        # if "vel" in preds_dict:
        #     batch_vel = preds_dict["vel"][..., -self.num_proposals :]
        batch_center = preds_dict["center"]
        batch_height = preds_dict["height"]
        batch_dim = preds_dict["dim"]
        batch_rot = preds_dict["rot"]
        batch_vel = None
        if "vel" in preds_dict:
            batch_vel = preds_dict["vel"]

        return [batch_score, batch_rot, batch_dim, batch_center, batch_height, batch_vel]

    def forward(self, x):
        for type, head in self.parent.heads.items():
            if type == "object":
                pred_dict = self.head_forward(head, x, self.classes_eye)
                return self.get_bboxes(pred_dict, head.one_hot)
            else:
                raise ValueError(f"unsupported head: {type}")
        

class SubclassFuser(nn.Module):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent

    @auto_fp16(apply_to=("features",))
    def forward(self, features):
        if self.parent.fuser is not None:
            x = self.parent.fuser(features)
        else:
            assert len(features) == 1, features
            x = features[0]

        x = self.parent.decoder["backbone"](x)
        x = self.parent.decoder["neck"](x)
        return x[0]

class CustomLayerNormImpl(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, normalized_shape, weight, bias, eps, x_shape):
        return F.layer_norm(input, normalized_shape, weight, bias, eps)

    @staticmethod
    def symbolic(g, input, normalized_shape, weight, bias, eps, x_shape):
        y = g.op("nv::CustomLayerNormalization", input, weight, bias, axis_i=-1, epsilon_f=eps)
        y.setType(input.type().with_sizes(x_shape))
        return y

class CustomLayerNorm(nn.LayerNorm):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return CustomLayerNormImpl.apply(
            input, self.normalized_shape, self.weight, self.bias, self.eps, input.size())

    @staticmethod
    def convert(old: nn.LayerNorm):
        Custom_layernorm = CustomLayerNorm(old.normalized_shape, old.eps, old.elementwise_affine)
        if Custom_layernorm.weight is not None:
            Custom_layernorm.weight.data = old.weight.data
            Custom_layernorm.bias.data   = old.bias.data
        return Custom_layernorm

def replace_layernorm(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm):
            parent, child = name.rsplit(".", 1)
            parent = model.get_submodule(parent)
            setattr(parent, child, CustomLayerNorm.convert(module))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export transfusion to onnx file")
    parser.add_argument("--ckpt", type=str, default="qat/ckpt/bevfusion_ptq.pth", help="Pretrain model")
    parser.add_argument('--fp16', action= 'store_true')
    args = parser.parse_args()
    model = torch.load(args.ckpt).module
    
    suffix = "int8"
    if args.fp16:
        suffix = "fp16"
        quantize.disable_quantization(model).apply()
    
    save_root = f"qat/onnx_{suffix}"
    os.makedirs(save_root, exist_ok=True)

    model.eval()
    fuser    = SubclassFuser(model).cuda()
    headbbox = SubclassHeadBBox(model).cuda()
    replace_layernorm(headbbox)

    TensorQuantizer.use_fb_fake_quant = True
    with torch.no_grad():
        camera_features = torch.randn(1, 80, 180, 180).cuda()
        lidar_features  = torch.randn(1, 256, 180, 180).cuda()

        fuser_onnx_path = f"{save_root}/fuser.onnx"
        torch.onnx.export(fuser, [camera_features, lidar_features], fuser_onnx_path, opset_version=13, 
            input_names=["camera", "lidar"],
            output_names=["middle"],
        )
        print(f"🚀 The export is completed. ONNX save as {fuser_onnx_path} 🤗, Have a nice day~")

        boxhead_onnx_path = f"{save_root}/head.bbox.onnx"
        head_input = torch.randn(1, 512, 180, 180).cuda()
        torch.onnx.export(headbbox, head_input, f"{save_root}/head.bbox.onnx", opset_version=13, 
            input_names=["middle"],
            output_names=["score", "rot", "dim", "reg", "height", "vel"],
        )
        print(f"🚀 The export is completed. ONNX save as {boxhead_onnx_path} 🤗, Have a nice day~")
