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

from typing import Tuple

import torch
from mmcv.runner import force_fp32
from torch import nn
import numpy as np
from mmdet3d.ops import bev_pool
import tensor
import os
__all__ = ["BaseTransform", "BaseDepthTransform"]

class no_jit_trace:
    def __enter__(self):
        # pylint: disable=protected-access
        self.state = torch._C._get_tracing_state()
        torch._C._set_tracing_state(None)

    def __exit__(self, *args):
        torch._C._set_tracing_state(self.state)
        self.state = None

newx = None
class BEVPooling(torch.autograd.Function):
    @staticmethod
    def forward(ctx, feat, depth, intervals, geom_feats, num_intervals, C, H, W):
        return newx
    
    @staticmethod
    def symbolic(g, feat, depth, intervals, geom_feats, num_intervals, C, H, W):
        return g.op("BEVPooling", feat, depth, intervals, geom_feats, num_intervals, C_i=C, H_i=H, W_i=W)

def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor(
        [(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]]
    )
    return dx, bx, nx


class BaseTransform(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        zbound: Tuple[float, float, float],
        dbound: Tuple[float, float, float],
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.image_size = image_size
        self.feature_size = feature_size
        self.downsample_factor = int(image_size[0] / feature_size[0])
        self.xbound = xbound
        self.ybound = ybound
        self.zbound = zbound
        self.dbound = dbound

        dx, bx, nx = gen_dx_bx(self.xbound, self.ybound, self.zbound)
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.C = out_channels
        self.fp16_enabled = False
        
        frustum, rays = self.create_frustum_rays()
        self.frustum_rays = frustum
        self.rays = rays
        self.D = self.frustum_rays.shape[0]
   
        
    def create_frustum_rays(self):
            """Generate frustum"""
            # make grid in image plane
            ogfH, ogfW = self.image_size
            fH, fW = ogfH // self.downsample_factor, ogfW // self.downsample_factor

            Xs = np.linspace(0, ogfW-1, fW)
            Ys = np.linspace(0, ogfH-1, fH)
            Xs, Ys = np.meshgrid(Xs, Ys)
            Zs = np.ones_like(Xs)
            Ws = np.ones_like(Xs)

            # H x W x 4
            rays = torch.from_numpy(np.stack([Xs, Ys, Zs, Ws], axis=-1).astype(np.float32))
            rays_d_bound = [0, 1, self.dbound[2]]

            # DID
            alpha = 1.5
            d_coords = np.arange(rays_d_bound[2]) / rays_d_bound[2]
            d_coords = np.power(d_coords, alpha)
            d_coords = rays_d_bound[0] + d_coords * (rays_d_bound[1] - rays_d_bound[0])
            d_coords = torch.tensor(d_coords, dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
            
            D, _, _ = d_coords.shape
            x_coords = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(
                1, 1, fW).expand(D, fH, fW)
            y_coords = torch.linspace(0, ogfH - 1, fH,
                                    dtype=torch.float).view(1, fH,
                                                            1).expand(D, fH, fW)
            paddings = torch.ones_like(d_coords)

            # D x H x W x 3
            frustum = torch.stack((x_coords, y_coords, d_coords, paddings), -1)
            return frustum, rays
        
    def get_geometry_rays(self, sensor2ego_mat, intrin_mat, ida_mat, bda_mat, denorms):
            """Transfer points from camera coord to ego coord.

            Args:
                rots(Tensor): Rotation matrix from camera to ego.
                trans(Tensor): Translation matrix from camera to ego.
                intrins(Tensor): Intrinsic matrix.
                post_rots_ida(Tensor): Rotation matrix for ida.
                post_trans_ida(Tensor): Translation matrix for ida
                post_rot_bda(Tensor): Rotation matrix for bda.

            Returns:
                Tensors: points ego coord.
            """
            batch_size, num_cams, _, _ = sensor2ego_mat.shape
            ego2sensor_mat = sensor2ego_mat.inverse()
            device = ego2sensor_mat.device

            H, W = self.rays.shape[:2]
            B, N = intrin_mat.shape[:2]
            O = (ego2sensor_mat @ torch.tensor([0, 0, 0, 1], dtype=torch.float32, device=device).view(1, 1, 4, 1))[..., :3, 0].view(B, N, 1, 1, 3, 1)
            n = (denorms[:, :3] / torch.norm(denorms[:, :3], dim=-1, keepdim=True)).view(B, N, 1, 1, 1, 3)
            P0 = O + self.dbound[0] * n.view(B, N, 1, 1, 3, 1)  
            P1 = O + self.dbound[1] * n.view(B, N, 1, 1, 3, 1)  
            self.rays = self.rays.to(intrin_mat.device)
            self.frustum_rays = self.frustum_rays.to(intrin_mat.device)
            
            rays = (self.rays.to(intrin_mat.device).view(1, 1, H, W, 4) @ (intrin_mat.inverse() @ ida_mat.inverse()).permute(0, 1, 3, 2).reshape(B, N, 1, 4, 4))[..., :3]
            dirs = (rays / torch.norm(rays, dim=-1, keepdim=True)).unsqueeze(-1)

            tmp_0 = (n @ P0) / (n @ dirs)
            tmp_1 = (n @ P1) / (n @ dirs)

            D, H, W, _ = self.frustum_rays.shape
            tmp_diff  = tmp_0 - tmp_1
            points = self.frustum_rays.view(1, 1, D, H, W, 4).repeat(B, N, 1, 1, 1, 1)
            points[..., 2] = (tmp_0.view(B, N, 1, H, W) - points[..., 2] * tmp_diff.view(B, N, 1, H, W)) * dirs[..., 2, 0].view(B, N, 1, H, W)
            points = points @ ida_mat.inverse().permute(0, 1, 3, 2).reshape(B, N, 1, 1, 4, 4)
            points[..., :2] *= points[..., [2]]

            matrix = sensor2ego_mat @ intrin_mat.inverse()
            if bda_mat is not None:
                matrix = bda_mat.unsqueeze(1) @ matrix

            return (points @ matrix.permute(0, 1, 3, 2).reshape(B, N, 1, 1, 4, 4))[..., :3]

    def get_cam_feats(self, x):
        raise NotImplementedError

    @force_fp32()
    def bev_pool(self, geom_feats, x, export = False):
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W

        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.0)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat(
            [
                torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long)
                for ix in range(B)
            ]
        )
        if export:
            geom_feats = torch.cat((geom_feats, batch_ix, torch.arange(len(batch_ix), device=batch_ix.device, dtype=torch.int32).unsqueeze(1)), 1)
        else:
            geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = (
            (geom_feats[:, 0] >= 0)
            & (geom_feats[:, 0] < self.nx[0])
            & (geom_feats[:, 1] >= 0)
            & (geom_feats[:, 1] < self.nx[1])
            & (geom_feats[:, 2] >= 0)
            & (geom_feats[:, 2] < self.nx[2])
        )
        x = x[kept]
        geom_feats = geom_feats[kept]

        if export:
            x, intervals, geom_feats = bev_pool(x, geom_feats, B, self.nx[2], self.nx[0], self.nx[1], export = True)
        else:
            x = bev_pool(x, geom_feats, B, self.nx[2], self.nx[0], self.nx[1])

        # collapse Z
        final = torch.cat(x.unbind(dim=2), 1)

        if export:
            return final, intervals, geom_feats
        else:
            return final

    @force_fp32()
    def forward(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        denorms,
        sensor2virtual,
        reference_heights,
        **kwargs,
    ):
        geom = self.get_geometry_rays(
            camera2lidar,
            camera_intrinsics,
            img_aug_matrix,
            lidar_aug_matrix,
            denorms[:,0]
        )
        x = self.get_cam_feats(img)
        x = self.bev_pool(geom, x)
        return x
    
    # @force_fp32()
    def export(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        denorms,
        sensor2virtual,
        reference_heights,
        intervals,
        geometry,
        num_intervals,
        **kwargs,
    ):

        feat, depth, x = self.get_cam_feats(img, export=True)
        torch.save([
            camera2lidar,
            camera_intrinsics,
            img_aug_matrix,
            lidar_aug_matrix,
            denorms[:,0]
        ], "metas.pth")
        with no_jit_trace():
            camera2lidar, camera_intrinsics, img_aug_matrix, lidar_aug_matrix, denorms = torch.load("metas.pth")
            if os.path.exists('metas.pth'): 
                os.remove('metas.pth')
                
            geom = self.get_geometry_rays(
                camera2lidar,
                camera_intrinsics,
                img_aug_matrix,
                lidar_aug_matrix,
                denorms
            )
            x, local_intervals, local_geom_feats = self.bev_pool(geom, x ,export= True)
            global newx
            newx = x
            # tensor.save(local_intervals, "intervals.tensor")
            # tensor.save(local_geom_feats, "geometrys.tensor")
        # torch.save(x, "bev.pth")
        return BEVPooling.apply(feat.permute(0, 2, 3, 1), depth, intervals, geometry, num_intervals, int(x.size(1)), int(x.size(2)), int(x.size(3)))
    
class BaseDepthTransform(BaseTransform):
    @force_fp32()
    def forward(
        self,
        img,
        points,
        sensor2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        cam_intrinsic,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        denorms,
        sensor2virtual,
        reference_heights,
        **kwargs,
    ):
        rots = sensor2ego[..., :3, :3]
        trans = sensor2ego[..., :3, 3]
        intrins = cam_intrinsic[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        lidar2ego_rots = lidar2ego[..., :3, :3]
        lidar2ego_trans = lidar2ego[..., :3, 3]
        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]

        # print(img.shape, self.image_size, self.feature_size)

        batch_size = len(points)
        depth = torch.zeros(batch_size, img.shape[1], 1, *self.image_size).to(
            points[0].device
        )

        for b in range(batch_size):
            cur_coords = points[b][:, :3]
            cur_img_aug_matrix = img_aug_matrix[b]
            cur_lidar_aug_matrix = lidar_aug_matrix[b]
            cur_lidar2image = lidar2image[b]

            # inverse aug
            cur_coords -= cur_lidar_aug_matrix[:3, 3]
            cur_coords = torch.inverse(cur_lidar_aug_matrix[:3, :3]).matmul(
                cur_coords.transpose(1, 0)
            )
            # lidar2image
            cur_coords = cur_lidar2image[:, :3, :3].matmul(cur_coords)
            cur_coords += cur_lidar2image[:, :3, 3].reshape(-1, 3, 1)
            # get 2d coords
            dist = cur_coords[:, 2, :]
            cur_coords[:, 2, :] = torch.clamp(cur_coords[:, 2, :], 1e-5, 1e5)
            cur_coords[:, :2, :] /= cur_coords[:, 2:3, :]

            # imgaug
            cur_coords = cur_img_aug_matrix[:, :3, :3].matmul(cur_coords)
            cur_coords += cur_img_aug_matrix[:, :3, 3].reshape(-1, 3, 1)
            cur_coords = cur_coords[:, :2, :].transpose(1, 2)

            # normalize coords for grid sample
            cur_coords = cur_coords[..., [1, 0]]

            on_img = (
                (cur_coords[..., 0] < self.image_size[0])
                & (cur_coords[..., 0] >= 0)
                & (cur_coords[..., 1] < self.image_size[1])
                & (cur_coords[..., 1] >= 0)
            )
            for c in range(on_img.shape[0]):
                masked_coords = cur_coords[c, on_img[c]].long()
                masked_dist = dist[c, on_img[c]]
                depth[b, c, 0, masked_coords[:, 0], masked_coords[:, 1]] = masked_dist

        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]
        geom = self.get_geometry(
            camera2lidar_rots,
            camera2lidar_trans,
            intrins,
            post_rots,
            post_trans,
            extra_rots=extra_rots,
            extra_trans=extra_trans,
        )

        x = self.get_cam_feats(img, depth)
        x = self.bev_pool(geom, x)
        return x
