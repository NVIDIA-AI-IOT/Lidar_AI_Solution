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
import math
import cv2
import numpy as np
import lzf
import struct
import random
import mmcv
import numpy as np
import torch
from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes
from nuscenes.utils.data_classes import Box  
from PIL import Image
from pyquaternion import Quaternion
from torch.utils.data import Dataset
import json
import tempfile
from os import path as osp
from typing import Any, Dict
import pyquaternion
from pyquaternion import Quaternion
from evaluators.det_evaluators import RoadSideEvaluator

__all__ = ['V2XDataset']

map_name_from_general_to_detection = {
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.wheelchair': 'ignore',
    'human.pedestrian.stroller': 'ignore',
    'human.pedestrian.personal_mobility': 'ignore',
    'human.pedestrian.police_officer': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'animal': 'ignore',
    'vehicle.car': 'car',
    'vehicle.motorcycle': 'motorcycle',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.truck': 'truck',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.emergency.ambulance': 'ignore',
    'vehicle.emergency.police': 'ignore',
    'vehicle.trailer': 'trailer',
    'movable_object.barrier': 'barrier',
    'movable_object.trafficcone': 'traffic_cone',
    'movable_object.pushable_pullable': 'ignore',
    'movable_object.debris': 'ignore',
    'static_object.bicycle_rack': 'ignore',
}

def equation_plane(points): 
    x1, y1, z1 = points[0, 0], points[0, 1], points[0, 2]
    x2, y2, z2 = points[1, 0], points[1, 1], points[1, 2]
    x3, y3, z3 = points[2, 0], points[2, 1], points[2, 2]
    a1 = x2 - x1
    b1 = y2 - y1
    c1 = z2 - z1
    a2 = x3 - x1
    b2 = y3 - y1
    c2 = z3 - z1
    a = b1 * c2 - b2 * c1
    b = a2 * c1 - a1 * c2
    c = a1 * b2 - b1 * a2
    d = (- a * x1 - b * y1 - c * z1)
    return np.array([a, b, c, d])

def get_denorm(sweepego2sweepsensor):
    ground_points_lidar = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    ground_points_lidar = np.concatenate((ground_points_lidar, np.ones((ground_points_lidar.shape[0], 1))), axis=1)
    ground_points_cam = np.matmul(sweepego2sweepsensor, ground_points_lidar.T).T
    denorm = -1 * equation_plane(ground_points_cam)
    return denorm

def get_sensor2virtual(denorm):
    origin_vector = np.array([0, 1, 0])
    target_vector = -1 * np.array([denorm[0], denorm[1], denorm[2]])
    target_vector_norm = target_vector / np.sqrt(target_vector[0]**2 + target_vector[1]**2 + target_vector[2]**2) 
    sita = math.acos(np.inner(target_vector_norm, origin_vector))
    n_vector = np.cross(target_vector_norm, origin_vector) 
    n_vector = n_vector / np.sqrt(n_vector[0]**2 + n_vector[1]**2 + n_vector[2]**2)
    n_vector = n_vector.astype(np.float32)
    rot_mat, _ = cv2.Rodrigues(n_vector * sita)
    rot_mat = rot_mat.astype(np.float32)
    sensor2virtual = np.eye(4)
    sensor2virtual[:3, :3] = rot_mat
    return sensor2virtual.astype(np.float32)

def get_reference_height(denorm):
    ref_height = np.abs(denorm[3]) / np.sqrt(denorm[0]**2 + denorm[1]**2 + denorm[2]**2)
    return ref_height.astype(np.float32)

def get_rot(h):
    return torch.Tensor([
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ])

def img_intrin_extrin_transform(img, ratio, roll, transform_pitch, intrin_mat):
    center = intrin_mat[:2, 2].astype(np.int32) 
    center = (int(center[0]), int(center[1]))

    W, H = img.size[0], img.size[1]
    new_W, new_H = (int(W * ratio), int(H * ratio))
    img = img.resize((new_W, new_H), Image.LANCZOS)
    
    h_min = int(center[1] * abs(1.0 - ratio))
    w_min = int(center[0] * abs(1.0 - ratio))
    if ratio <= 1.0:
        image = Image.new(mode='RGB', size=(W, H))
        image.paste(img, (w_min, h_min,  w_min + new_W, h_min + new_H))
    else:
        image = img.crop((w_min, h_min,  w_min + W, h_min + H))
    img = image.rotate(-roll, expand=0, center=center, translate=(0, transform_pitch), fillcolor=(0,0,0), resample=Image.BICUBIC)
    return img

def img_transform(img, resize, resize_dims, crop, flip, rotate):
    ida_rot = torch.eye(2)
    ida_tran = torch.zeros(2)
    # adjust image
    img = img.resize(resize_dims)
    img = img.crop(crop)
    if flip:
        img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
    img = img.rotate(rotate)

    # post-homography transformation
    ida_rot *= resize
    ida_tran -= torch.Tensor(crop[:2])
    if flip:
        A = torch.Tensor([[-1, 0], [0, 1]])
        b = torch.Tensor([crop[2] - crop[0], 0])
        ida_rot = A.matmul(ida_rot)
        ida_tran = A.matmul(ida_tran) + b
    A = get_rot(rotate / 180 * np.pi)
    b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
    b = A.matmul(-b) + b
    ida_rot = A.matmul(ida_rot)
    ida_tran = A.matmul(ida_tran) + b
    ida_mat = ida_rot.new_zeros(4, 4)
    ida_mat[3, 3] = 1
    ida_mat[2, 2] = 1
    ida_mat[:2, :2] = ida_rot
    ida_mat[:2, 3] = ida_tran
    return img, ida_mat


def bev_transform(gt_boxes, rotate_angle, scale_ratio, flip_dx, flip_dy):
    rotate_angle = torch.tensor(rotate_angle / 180 * np.pi)
    rot_sin = torch.sin(rotate_angle)
    rot_cos = torch.cos(rotate_angle)
    rot_mat = torch.Tensor([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0],
                            [0, 0, 1]])
    scale_mat = torch.Tensor([[scale_ratio, 0, 0], [0, scale_ratio, 0],
                              [0, 0, scale_ratio]])
    flip_mat = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    if flip_dx:
        flip_mat = flip_mat @ torch.Tensor([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
    if flip_dy:
        flip_mat = flip_mat @ torch.Tensor([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
    rot_mat = flip_mat @ (scale_mat @ rot_mat)
    if gt_boxes.shape[0] > 0:
        gt_boxes[:, :3] = (rot_mat @ gt_boxes[:, :3].unsqueeze(-1)).squeeze(-1)
        gt_boxes[:, 3:6] *= scale_ratio
        gt_boxes[:, 6] += rotate_angle
        if flip_dx:
            gt_boxes[:, 6] = 2 * torch.asin(torch.tensor(1.0)) - gt_boxes[:, 6]
        if flip_dy:
            gt_boxes[:, 6] = -gt_boxes[:, 6]
        gt_boxes[:, 7:] = (
            rot_mat[:2, :2] @ gt_boxes[:, 7:].unsqueeze(-1)).squeeze(-1)
    return gt_boxes, rot_mat


def depth_transform(cam_depth, resize, resize_dims, crop, flip, rotate):
    """Transform depth based on ida augmentation configuration.

    Args:
        cam_depth (np array): Nx3, 3: x,y,d.
        resize (float): Resize factor.
        resize_dims (list): Final dimension.
        crop (list): x1, y1, x2, y2
        flip (bool): Whether to flip.
        rotate (float): Rotation value.

    Returns:
        np array: [h/down_ratio, w/down_ratio, d]
    """

    H, W = resize_dims
    cam_depth[:, :2] = cam_depth[:, :2] * resize
    cam_depth[:, 0] -= crop[0]
    cam_depth[:, 1] -= crop[1]
    if flip:
        cam_depth[:, 0] = resize_dims[1] - cam_depth[:, 0]

    cam_depth[:, 0] -= W / 2.0
    cam_depth[:, 1] -= H / 2.0

    h = rotate / 180 * np.pi
    rot_matrix = [
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ]
    cam_depth[:, :2] = np.matmul(rot_matrix, cam_depth[:, :2].T).T

    cam_depth[:, 0] += W / 2.0
    cam_depth[:, 1] += H / 2.0

    depth_coords = cam_depth[:, :2].astype(np.int16)

    depth_map = np.zeros(resize_dims)
    valid_mask = ((depth_coords[:, 1] < resize_dims[0])
                  & (depth_coords[:, 0] < resize_dims[1])
                  & (depth_coords[:, 1] >= 0)
                  & (depth_coords[:, 0] >= 0))
    depth_map[depth_coords[valid_mask, 1],
              depth_coords[valid_mask, 0]] = cam_depth[valid_mask, 2]

    return torch.Tensor(depth_map)

from mmdet.datasets import DATASETS
@DATASETS.register_module()
class V2XDataset(Dataset):
    def __init__(self,
                 ida_aug_conf,
                 classes,
                 data_root,
                 kitti_root,
                 result_root,
                 info_path,
                 is_train,
                 use_cbgs=False,
                 num_sweeps=1,
                 img_conf=dict(img_mean=[123.675, 116.28, 103.53],
                               img_std=[58.395, 57.12, 57.375],
                               to_rgb=True),
                 return_depth=False,
                 sweep_idxes=list(),
                 key_idxes=list()):
        """Dataset used for bevdetection task.
        Args:
            ida_aug_conf (dict): Config for ida augmentation.
            classes (list): Class names.
            use_cbgs (bool): Whether to use cbgs strategy,
                Default: False.
            num_sweeps (int): Number of sweeps to be used for each sample.
                default: 1.
            img_conf (dict): Config for image.
            return_depth (bool): Whether to use depth gt.
                default: False.
            sweep_idxes (list): List of sweep idxes to be used.
                default: list().
            key_idxes (list): List of key idxes to be used.
                default: list().
        """
        super().__init__()
        self.infos = mmcv.load(info_path)
        self.is_train = is_train
        self.ida_aug_conf = ida_aug_conf
        self.data_root = data_root
        self.result_root = result_root
        if not os.path.exists(result_root):
            os.makedirs(result_root)
            
        self.classes = classes
        self.use_cbgs = use_cbgs
        if self.use_cbgs:
            self.cat2id = {name: i for i, name in enumerate(self.classes)}
            self.sample_indices = self._get_sample_indices()
        self.num_sweeps = num_sweeps
        self.img_mean = np.array(img_conf['img_mean'], np.float32)
        self.img_std = np.array(img_conf['img_std'], np.float32)
        self.to_rgb = img_conf['to_rgb']
        self.return_depth = return_depth
        assert sum([sweep_idx >= 0 for sweep_idx in sweep_idxes]) \
            == len(sweep_idxes), 'All `sweep_idxes` must greater \
                than or equal to 0.'

        self.sweeps_idx = sweep_idxes
        assert sum([key_idx < 0 for key_idx in key_idxes]) == len(key_idxes),\
            'All `key_idxes` must less than 0.'
        self.key_idxes = [0] + key_idxes

        self.ratio_range = [1.0, 0.20]
        self.roll_range = [0.0, 2.00]
        self.pitch_range = [0.0, 0.67]
        if is_train:
            self._set_group_flag()
        from nuscenes.eval.detection.config import config_factory
        self.eval_version = "detection_cvpr_2019"
        self.eval_detection_configs = config_factory("detection_cvpr_2019")
        self.evaluator = RoadSideEvaluator(class_names=self.classes,
                                           current_classes=["Car", "Pedestrian", "Cyclist"],
                                           data_root=data_root,
                                           gt_label_path= kitti_root,
                                           output_dir=self.result_root)
    def set_epoch(self, epoch):
        self.epoch = epoch

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0. In 3D datasets, they are all the same, thus are all
        zeros.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        
    def _get_sample_indices(self):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations after class sampling.
        """
        class_sample_idxs = {cat_id: [] for cat_id in self.cat2id.values()}
        for idx, info in enumerate(self.infos):
            gt_names = set(
                [ann_info['category_name'] for ann_info in info['ann_infos']])
            for gt_name in gt_names:
                gt_name = map_name_from_general_to_detection[gt_name]
                if gt_name not in self.classes:
                    continue
                class_sample_idxs[self.cat2id[gt_name]].append(idx)
        duplicated_samples = sum(
            [len(v) for _, v in class_sample_idxs.items()])
        class_distribution = {
            k: len(v) / duplicated_samples
            for k, v in class_sample_idxs.items()
        }

        sample_indices = []
        frac = 1.0 / len(self.classes)
        ratios = [frac / v for v in class_distribution.values()]
        for cls_inds, ratio in zip(list(class_sample_idxs.values()), ratios):
            sample_indices += np.random.choice(cls_inds,
                                               int(len(cls_inds) *
                                                   ratio)).tolist()
        return sample_indices

    def degree2rad(self, degree):
        return degree * np.pi / 180

    def get_M(self, R, K, R_r, K_r):
        R_inv = np.linalg.inv(R)
        K_inv = np.linalg.inv(K)
        M = np.matmul(K_r, R_r)
        M = np.matmul(M, R_inv)
        M = np.matmul(M, K_inv)
        return M

    def sample_intrin_extrin_augmentation(self, intrin_mat, sweepego2sweepsensor):
        intrin_mat, sweepego2sweepsensor = intrin_mat.numpy(), sweepego2sweepsensor.numpy()
        # rectify intrin_mat
        ratio = np.random.normal(self.ratio_range[0], self.ratio_range[1])
        intrin_mat_rectify = intrin_mat.copy()
        intrin_mat_rectify[:2,:2] = intrin_mat[:2,:2] * ratio
        
        # rectify sweepego2sweepsensor by roll
        roll = np.random.normal(self.roll_range[0], self.roll_range[1])
        roll_rad = self.degree2rad(roll)
        rectify_roll = np.array([[math.cos(roll_rad), -math.sin(roll_rad), 0, 0], 
                                 [math.sin(roll_rad), math.cos(roll_rad), 0, 0], 
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1]])
        sweepego2sweepsensor_rectify_roll = np.matmul(rectify_roll, sweepego2sweepsensor)
        
        # rectify sweepego2sweepsensor by pitch
        pitch = np.random.normal(self.pitch_range[0], self.pitch_range[1])
        pitch_rad = self.degree2rad(pitch)
        rectify_pitch = np.array([[1, 0, 0, 0],
                                  [0,math.cos(pitch_rad), -math.sin(pitch_rad), 0], 
                                  [0,math.sin(pitch_rad), math.cos(pitch_rad), 0],
                                  [0, 0, 0, 1]])
        sweepego2sweepsensor_rectify_pitch = np.matmul(rectify_pitch, sweepego2sweepsensor_rectify_roll)
        M = self.get_M(sweepego2sweepsensor_rectify_roll[:3,:3], intrin_mat_rectify[:3,:3], sweepego2sweepsensor_rectify_pitch[:3,:3], intrin_mat_rectify[:3,:3])
        center = intrin_mat_rectify[:2, 2]  # w, h
        center_ref = np.array([center[0], center[1], 1.0])
        center_ref = np.matmul(M, center_ref.T)[:2]
        transform_pitch = int(center_ref[1] - center[1])

        intrin_mat_rectify, sweepego2sweepsensor_rectify = torch.Tensor(intrin_mat_rectify), torch.Tensor(sweepego2sweepsensor_rectify_pitch)
        return intrin_mat_rectify, sweepego2sweepsensor_rectify, ratio, roll, transform_pitch

    def sample_ida_augmentation(self):
        """Generate ida augmentation values based on ida_config."""
        H, W = self.ida_aug_conf['H'], self.ida_aug_conf['W']
        fH, fW = self.ida_aug_conf['final_dim']
        resize = max(fH / H, fW / W)
        resize_dims = (int(W * resize), int(H * resize))
        newW, newH = resize_dims
        crop_h = int(
            (1 - np.mean(self.ida_aug_conf['bot_pct_lim'])) * newH) - fH
        crop_w = int(max(0, newW - fW) / 2)
        crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        flip = False
        rotate_ida = 0
        return resize, resize_dims, crop, flip, rotate_ida

    def sample_bda_augmentation(self):
        """Generate bda augmentation values based on bda_config."""
        rotate_bda = 0
        scale_bda = 1.0
        flip_dx = False
        flip_dy = False
        return rotate_bda, scale_bda, flip_dx, flip_dy

    def load_lidar2camera_mat(self, file, root_path):
        file_path = file.replace('velodyne', 'calib/virtuallidar_to_camera')
        file_path = file_path.replace('.pcd', '.json')
        with open(os.path.join(root_path, file_path), 'r') as json_file:
            data = json.load(json_file)
        rotation_matrix = np.array(data["rotation"])
        translation = np.array(data["translation"]) 

        lidar2cam = np.eye(4)
        lidar2cam[:3, :3] = rotation_matrix
        lidar2cam[:3, 3] = translation.flatten()
        return lidar2cam
        
        
        
    def load_pcd(self, file, convert=True):
        numpy_pcd_type_mappings = [(np.dtype('float32'), ('F', 4)),
                            (np.dtype('float64'), ('F', 8)),
                            (np.dtype('uint8'), ('U', 1)),
                            (np.dtype('uint16'), ('U', 2)),
                            (np.dtype('uint32'), ('U', 4)),
                            (np.dtype('uint64'), ('U', 8)),
                            (np.dtype('int16'), ('I', 2)),
                            (np.dtype('int32'), ('I', 4)),
                            (np.dtype('int64'), ('I', 8))]
        pcd_type_to_numpy_type = dict((q, p) for (p, q) in numpy_pcd_type_mappings)

        meta = dict()
        with open(file, "rb") as f:
            while True:
                line = str(f.readline().strip(), "utf-8")
                if line.startswith('# .PCD v0.7'):
                    continue
                
                if line.startswith("VERSION"):
                    meta["version"] = line[8:]
                elif line.startswith("FIELDS"):
                    meta["fields"] = line[7:].split()
                elif line.startswith("SIZE"):
                    meta["size"] = list(map(int, line[5:].split()))
                elif line.startswith("TYPE"):
                    meta["type"] = line[5:].split()
                elif line.startswith("COUNT"):
                    meta["count"] = list(map(int, line[6:].split()))
                elif line.startswith("WIDTH"):
                    meta["width"] = int(line[6:])
                elif line.startswith("HEIGHT"):
                    meta["height"] = int(line[7:])
                elif line.startswith("VIEWPOINT"):
                    meta["viewpoint"] = list(map(float, line[10:].split()))
                elif line.startswith("POINTS"):
                    meta["points"] = int(line[7:])
                elif line.startswith("DATA"):
                    meta["data_type"] = line[5:]
                    break
                else:
                    raise KeyError(f"Unknow line: {line}")

            dtype = np.dtype(list(zip(meta["fields"], [pcd_type_to_numpy_type[(t, s)] for t, s in zip(meta["type"], meta["size"])])))
            if meta["data_type"] == "ascii":
                data  = np.loadtxt(f, dtype=dtype, delimiter=' ')
            elif meta["data_type"] == "binary":
                rowstep = meta['points'] * dtype.itemsize
                buf = f.read(rowstep)
                data = np.frombuffer(buf, dtype=dtype)
            elif meta["data_type"] == "binary_compressed":
                fmt = 'II'
                compressed_size, uncompressed_size = struct.unpack(fmt, f.read(struct.calcsize(fmt)))
                compressed_data = f.read(compressed_size)
                buf = lzf.decompress(compressed_data, uncompressed_size)
                if len(buf) != uncompressed_size:
                    raise IOError('Error decompressing data')
                
                data = np.zeros(meta['width'], dtype=dtype)
                ix = 0
                for dti in range(len(dtype)):
                    dt = dtype[dti]
                    bytes = dt.itemsize * meta['width']
                    column = np.frombuffer(buf[ix:(ix+bytes)], dt)
                    data[dtype.names[dti]] = column
                    ix += bytes

            if convert:
                output = np.zeros((len(data) , len(dtype)), dtype=np.float32)
                for ic in range(len(dtype)):
                    output[:, ic] = data[dtype.names[ic]]
                data   = output
        return data, meta
    
    def get_image(self, cam_infos, cams):
        """Given data and cam_names, return image data needed.

        Args:
            sweeps_data (list): Raw data used to generate the data we needed.
            cams (list): Camera names.

        Returns:
            Tensor: Image data after processing.
            Tensor: Transformation matrix from camera to ego.
            Tensor: Intrinsic matrix.
            Tensor: Transformation matrix for ida.
            Tensor: Transformation matrix from key
                frame camera to sweep frame camera.
            Tensor: timestamps.
            dict: meta infos needed for evaluation.
        """
        assert len(cam_infos) > 0
        sweep_imgs = list()
        sweep_sensor2ego_mats = list()
        sweep_intrin_mats = list()
        sweep_ida_mats = list()
        sweep_sensor2sensor_mats = list()
        sweep_sensor2virtual_mats = list()
        sweep_timestamps = list()
        sweep_reference_heights = list()
        gt_depth = list()
        denorms  = list()
        for cam in cams:
            imgs = list()
            sensor2ego_mats = list()
            intrin_mats = list()
            ida_mats = list()
            sensor2sensor_mats = list()
            sensor2virtual_mats=list()
            reference_heights = list()
            timestamps = list()
            key_info = cam_infos[0]
            resize, resize_dims, crop, flip, \
                rotate_ida = self.sample_ida_augmentation(
                    )
            for sweep_idx, cam_info in enumerate(cam_infos):
                img = Image.open(
                    os.path.join(self.data_root, cam_info[cam]['filename']))
                if "rotation_matrix" in cam_info[cam]['calibrated_sensor'].keys():
                    sweepsensor2sweepego_rot = torch.Tensor(cam_info[cam]['calibrated_sensor']['rotation_matrix'])
                else:
                    w, x, y, z = cam_info[cam]['calibrated_sensor']['rotation']
                    # sweep sensor to sweep ego
                    sweepsensor2sweepego_rot = torch.Tensor(
                        Quaternion(w, x, y, z).rotation_matrix)
                sweepsensor2sweepego_tran = torch.Tensor(
                    cam_info[cam]['calibrated_sensor']['translation'])
                sweepsensor2sweepego = sweepsensor2sweepego_rot.new_zeros(
                    (4, 4))
                sweepsensor2sweepego[3, 3] = 1
                sweepsensor2sweepego[:3, :3] = sweepsensor2sweepego_rot
                sweepsensor2sweepego[:3, -1] = sweepsensor2sweepego_tran
                
                sweepego2sweepsensor = sweepsensor2sweepego.inverse()
                denorm = get_denorm(sweepego2sweepsensor.numpy())
                # sweep ego to global
                w, x, y, z = cam_info[cam]['ego_pose']['rotation']
                sweepego2global_rot = torch.Tensor(
                    Quaternion(w, x, y, z).rotation_matrix)
                sweepego2global_tran = torch.Tensor(
                    cam_info[cam]['ego_pose']['translation'])
                sweepego2global = sweepego2global_rot.new_zeros((4, 4))
                sweepego2global[3, 3] = 1
                sweepego2global[:3, :3] = sweepego2global_rot
                sweepego2global[:3, -1] = sweepego2global_tran
                
                intrin_mat = torch.zeros((4, 4))
                intrin_mat[3, 3] = 1
                intrin_mat[:3, :3] = torch.Tensor(
                    cam_info[cam]['calibrated_sensor']['camera_intrinsic'])
                sweepego2sweepsensor = sweepsensor2sweepego.inverse()
            
                # if self.is_train and random.random() < 0.5:
                #     intrin_mat, sweepego2sweepsensor, ratio, roll, transform_pitch = self.sample_intrin_extrin_augmentation(intrin_mat, sweepego2sweepsensor)
                #     img = img_intrin_extrin_transform(img, ratio, roll, transform_pitch, intrin_mat.numpy())
                
                denorm = get_denorm(sweepego2sweepsensor.numpy())
                sweepsensor2sweepego = sweepego2sweepsensor.inverse()

                # global sensor to cur ego
                w, x, y, z = key_info[cam]['ego_pose']['rotation']
                keyego2global_rot = torch.Tensor(
                    Quaternion(w, x, y, z).rotation_matrix)
                keyego2global_tran = torch.Tensor(
                    key_info[cam]['ego_pose']['translation'])
                keyego2global = keyego2global_rot.new_zeros((4, 4))
                keyego2global[3, 3] = 1
                keyego2global[:3, :3] = keyego2global_rot
                keyego2global[:3, -1] = keyego2global_tran
                global2keyego = keyego2global.inverse()

                # cur ego to sensor
                if "rotation_matrix" in key_info[cam]['calibrated_sensor'].keys():
                    keysensor2keyego_rot = torch.Tensor(key_info[cam]['calibrated_sensor']['rotation_matrix'])
                else:
                    w, x, y, z = key_info[cam]['calibrated_sensor']['rotation']
                    keysensor2keyego_rot = torch.Tensor(
                        Quaternion(w, x, y, z).rotation_matrix)
                keysensor2keyego_tran = torch.Tensor(
                    key_info[cam]['calibrated_sensor']['translation'])
                keysensor2keyego = keysensor2keyego_rot.new_zeros((4, 4))
                keysensor2keyego[3, 3] = 1
                keysensor2keyego[:3, :3] = keysensor2keyego_rot
                keysensor2keyego[:3, -1] = keysensor2keyego_tran
                keyego2keysensor = keysensor2keyego.inverse()
                keysensor2sweepsensor = (
                    keyego2keysensor @ global2keyego @ sweepego2global
                    @ sweepsensor2sweepego).inverse()
                sweepsensor2keyego = global2keyego @ sweepego2global @\
                    sweepsensor2sweepego
                sensor2virtual = torch.Tensor(get_sensor2virtual(denorm))
                sensor2ego_mats.append(sweepsensor2keyego)
                sensor2sensor_mats.append(keysensor2sweepsensor)
                sensor2virtual_mats.append(sensor2virtual)

                if self.return_depth and sweep_idx == 0:
                    file_name = os.path.split(cam_info[cam]['filename'])[-1]
                    point_depth = np.fromfile(os.path.join(
                        self.data_root, 'depth_gt', f'{file_name}.bin'),
                                              dtype=np.float32,
                                              count=-1).reshape(-1, 3)
                    point_depth_augmented = depth_transform(
                        point_depth, resize, self.ida_aug_conf['final_dim'],
                        crop, flip, rotate_ida)
                    gt_depth.append(point_depth_augmented)
                img, ida_mat = img_transform(
                    img,
                    resize=resize,
                    resize_dims=resize_dims,
                    crop=crop,
                    flip=flip,
                    rotate=rotate_ida,
                )
                ida_mats.append(ida_mat)
                img = mmcv.imnormalize(np.array(img), self.img_mean,
                                       self.img_std, self.to_rgb)
                img = torch.from_numpy(img).permute(2, 0, 1)
                imgs.append(img)
                intrin_mats.append(intrin_mat)
                timestamps.append(cam_info[cam]['timestamp'])
                reference_heights.append(get_reference_height(denorm))
                
            denorms.append(torch.from_numpy(denorm.astype(np.float32)))
            sweep_imgs.append(torch.stack(imgs))
            sweep_sensor2ego_mats.append(torch.stack(sensor2ego_mats))
            sweep_intrin_mats.append(torch.stack(intrin_mats))
            sweep_ida_mats.append(torch.stack(ida_mats))
            sweep_sensor2sensor_mats.append(torch.stack(sensor2sensor_mats))
            sweep_sensor2virtual_mats.append(torch.stack(sensor2virtual_mats))
            sweep_timestamps.append(torch.tensor(timestamps))
            sweep_reference_heights.append(torch.tensor(reference_heights))
            
        # Get mean pose of all cams.
        ego2global_rotation = np.mean(
            [key_info[cam]['ego_pose']['rotation'] for cam in cams], 0)
        ego2global_translation = np.mean(
            [key_info[cam]['ego_pose']['translation'] for cam in cams], 0)
        img_metas = dict(
            box_type_3d=LiDARInstance3DBoxes,
            ego2global_translation=ego2global_translation,
            ego2global_rotation=ego2global_rotation,
        )

        ret_list = [
            torch.stack(sweep_imgs).permute(1, 0, 2, 3, 4),
            torch.stack(sweep_sensor2ego_mats).permute(1, 0, 2, 3),
            torch.stack(sweep_intrin_mats).permute(1, 0, 2, 3),
            torch.stack(sweep_ida_mats).permute(1, 0, 2, 3),
            torch.stack(sweep_sensor2sensor_mats).permute(1, 0, 2, 3),
            torch.stack(sweep_sensor2virtual_mats).permute(1, 0, 2, 3),
            torch.stack(sweep_timestamps).permute(1, 0),
            torch.stack(sweep_reference_heights).permute(1, 0),
            torch.stack(denorms),
            img_metas,
        ]
        if self.return_depth:
            ret_list.append(torch.stack(gt_depth))
        return ret_list

    def get_gt(self, info, cams):
        """Generate gt labels from info.

        Args:
            info(dict): Infos needed to generate gt labels.
            cams(list): Camera names.

        Returns:
            Tensor: GT bboxes.
            Tensor: GT labels.
        """
        ego2global_rotation = np.mean(
            [info['cam_infos'][cam]['ego_pose']['rotation'] for cam in cams],
            0)
        ego2global_translation = np.mean([
            info['cam_infos'][cam]['ego_pose']['translation'] for cam in cams
        ], 0)
        trans = -np.array(ego2global_translation)
        rot = Quaternion(ego2global_rotation).inverse
        gt_boxes = list()
        gt_labels = list()
        for ann_info in info['ann_infos']:
            # Use ego coordinate.
            if (map_name_from_general_to_detection[ann_info['category_name']]
                    not in self.classes
                    or ann_info['num_lidar_pts'] + ann_info['num_radar_pts'] <=
                    0):
                continue
            box = Box(
                ann_info['translation'],
                ann_info['size'],
                Quaternion(ann_info['rotation']),
                velocity=ann_info['velocity'],
            )
            box.translate(trans)
            box.rotate(rot)
            box_xyz = np.array(box.center)
            box_dxdydz = np.array(box.wlh)[[1, 0, 2]]
            box_yaw = np.array([box.orientation.yaw_pitch_roll[0]])
            box_velo = np.array(box.velocity[:2])
            gt_box = np.concatenate([box_xyz, box_dxdydz, box_yaw, box_velo])
            gt_boxes.append(gt_box)
            gt_labels.append(
                self.classes.index(map_name_from_general_to_detection[
                    ann_info['category_name']]))
        return torch.Tensor(gt_boxes), torch.tensor(gt_labels)

    def choose_cams(self):
        """Choose cameras randomly.

        Returns:
            list: Cameras to be used.
        """
        if self.is_train and self.ida_aug_conf['Ncams'] < len(
                self.ida_aug_conf['cams']):
            cams = np.random.choice(self.ida_aug_conf['cams'],
                                    self.ida_aug_conf['Ncams'],
                                    replace=False)
        else:
            cams = self.ida_aug_conf['cams']
        return cams

    def __getitem__(self, idx):
        if self.use_cbgs:
            idx = self.sample_indices[idx]
        cam_infos = list()
        # TODO: Check if it still works when number of cameras is reduced.
        cams = self.choose_cams()
        for key_idx in self.key_idxes:
            cur_idx = key_idx + idx
            # Handle scenarios when current idx doesn't have previous key
            # frame or previous key frame is from another scene.
            if cur_idx < 0:
                cur_idx = idx
            elif self.infos[cur_idx]['scene_token'] != self.infos[idx][
                    'scene_token']:
                cur_idx = idx
            info = self.infos[cur_idx]
            cam_infos.append(info['cam_infos'])
            for sweep_idx in self.sweeps_idx:
                if len(info['sweeps']) == 0:
                    cam_infos.append(info['cam_infos'])
                else:
                    # Handle scenarios when current sweep doesn't have all
                    # cam keys.
                    for i in range(min(len(info['sweeps']) - 1, sweep_idx), -1,
                                   -1):
                        if sum([cam in info['sweeps'][i]
                                for cam in cams]) == len(cams):
                            cam_infos.append(info['sweeps'][i])
                            break
        image_data_list = self.get_image(cam_infos, cams)
        
        lidar_data_list, _ = self.load_pcd(os.path.join(self.data_root,self.infos[idx]['lidar_infos']['LIDAR_TOP']['filename']))
        lidar2camera = self.load_lidar2camera_mat(self.infos[idx]['lidar_infos']['LIDAR_TOP']['filename'], self.data_root)
        
        ret_list = list()
        (
            sweep_imgs,
            sweep_sensor2ego_mats,
            sweep_intrins,
            sweep_ida_mats,
            sweep_sensor2sensor_mats,
            sweep_sensor2virtual_mats,
            sweep_timestamps,
            sweep_reference_heights,
            denorms,
            img_metas,
        ) = image_data_list[:10]
        img_metas['token'] = self.infos[idx]['sample_token']
        if self.is_train:
            gt_boxes, gt_labels = self.get_gt(self.infos[idx], cams)
        # Temporary solution for test.
        else:
            gt_boxes  = sweep_imgs.new_zeros(0, 7)
            gt_labels = sweep_imgs.new_zeros(0, )
        
        rotate_bda, scale_bda, flip_dx, flip_dy = self.sample_bda_augmentation(
        )

        bda_mat = sweep_imgs.new_zeros(4, 4)
        bda_mat[3, 3] = 1
        gt_boxes, bda_rot = bev_transform(gt_boxes, rotate_bda, scale_bda,
                                          flip_dx, flip_dy)
        bda_mat[:3, :3] = bda_rot
        ret_list = [
            sweep_imgs,
            sweep_sensor2ego_mats,
            sweep_intrins,
            sweep_ida_mats,
            sweep_sensor2sensor_mats,
            sweep_sensor2virtual_mats,
            bda_mat,
            sweep_timestamps,
            sweep_reference_heights,
            denorms,
            img_metas,
            gt_boxes,
            gt_labels,
            lidar_data_list,
            lidar2camera
        ]
        if self.return_depth:
            ret_list.append(image_data_list[9])
        return ret_list

    def __str__(self):
        return f"""NuscData: {len(self)} samples. Split: \
            {"train" if self.is_train else "val"}.
                    Augmentation Conf: {self.ida_aug_conf}"""

    def __len__(self):
        if self.use_cbgs:
            return len(self.sample_indices)
        else:
            return len(self.infos)

    def evaluate_map(self, results):
        thresholds = torch.tensor([0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65])

        num_classes = len(self.map_classes)
        num_thresholds = len(thresholds)

        tp = torch.zeros(num_classes, num_thresholds)
        fp = torch.zeros(num_classes, num_thresholds)
        fn = torch.zeros(num_classes, num_thresholds)

        for result in results:
            pred = result["masks_bev"]
            label = result["gt_masks_bev"]

            pred = pred.detach().reshape(num_classes, -1)
            label = label.detach().bool().reshape(num_classes, -1)

            pred = pred[:, :, None] >= thresholds
            label = label[:, :, None]

            tp += (pred & label).sum(dim=1)
            fp += (pred & ~label).sum(dim=1)
            fn += (~pred & label).sum(dim=1)

        ious = tp / (tp + fp + fn + 1e-7)

        metrics = {}
        for index, name in enumerate(self.map_classes):
            metrics[f"map/{name}/iou@max"] = ious[index].max().item()
            for threshold, iou in zip(thresholds, ious[index]):
                metrics[f"map/{name}/iou@{threshold.item():.2f}"] = iou.item()
        metrics["map/mean/iou@max"] = ious.max(dim=1).values.mean().item()
        print(metrics)
        return metrics

    def evaluate(
        self,
        results,
        metric="bbox",
        logger=None,
        **kwargs,
    ):
        results_cal = []
        all_pred_results = list()
        all_img_metas = list()
        
        for i in range(len(results)):
            det_info = []
            det_info.append(results[i]['boxes_3d'].tensor.detach().cpu().numpy())
            det_info.append(results[i]['scores_3d'].detach().cpu().numpy())
            det_info.append(results[i]['labels_3d'].detach().cpu().numpy())
            all_img_metas.append(results[i]['metas'])
            all_pred_results.append(det_info)
        mAP_3d_moderate, result= self.evaluator.evaluate(all_pred_results, all_img_metas, out_dir= self.result_root)
        if logger is not None:
            logger.info(f"result:\n{result}")
        return {'eval': 'finish'}

def collate_fn(data, is_return_depth=False):
    imgs_batch = list()
    sensor2ego_mats_batch = list()
    intrin_mats_batch = list()
    ida_mats_batch = list()
    sensor2sensor_mats_batch = list()
    sensor2virtual_mats_batch = list()
    bda_mat_batch = list()
    timestamps_batch = list()
    reference_heights_batch = list()
    gt_boxes_batch = list()
    gt_labels_batch = list()
    img_metas_batch = list()
    depth_labels_batch = list()
    denorms_batch = list()
    lidar_data_list_batch = list()
    lidar2camera_batch = list()
    lidar2image_batch = list()
    for iter_data in data:
        (
            sweep_imgs,
            sweep_sensor2ego_mats,
            sweep_intrins,
            sweep_ida_mats,
            sweep_sensor2sensor_mats,
            sweep_sensor2virtual_mats,
            bda_mat,
            sweep_timestamps,
            sweep_reference_heights,
            denorms,
            img_metas,
            gt_boxes,
            gt_labels,
            lidar_data_list,
            lidar2camera   
        ) = iter_data[:15]
        
        if is_return_depth:
            gt_depth = iter_data[12]
            depth_labels_batch.append(gt_depth)
            
        denorms_batch.append(denorms)
        imgs_batch.append(sweep_imgs)
        sensor2ego_mats_batch.append(sweep_sensor2ego_mats)
        intrin_mats_batch.append(sweep_intrins)
        ida_mats_batch.append(sweep_ida_mats)
        sensor2sensor_mats_batch.append(sweep_sensor2sensor_mats)
        sensor2virtual_mats_batch.append(sweep_sensor2virtual_mats)
        bda_mat_batch.append(bda_mat)
        timestamps_batch.append(sweep_timestamps)
        reference_heights_batch.append(sweep_reference_heights)
        img_metas_batch.append(img_metas)
        gt_boxes = LiDARInstance3DBoxes(gt_boxes, box_dim=gt_boxes.shape[-1], origin=(0.5, 0.5, 0.5))
        gt_boxes_batch.append(gt_boxes)
        gt_labels_batch.append(gt_labels)
        lidar_data_list_batch.append(torch.tensor(lidar_data_list, dtype=torch.float32))
        lidar2camera_batch.append(torch.unsqueeze(torch.tensor(lidar2camera,dtype=torch.float32), dim =0))
        lidar2image = sweep_ida_mats[0,0] @ sweep_intrins[0,0] @ torch.tensor(lidar2camera,dtype=torch.float32)
        lidar2image_batch.append(torch.unsqueeze(lidar2image, dim =0))
            
    identity_matrix_batch = torch.eye(4).repeat(len(imgs_batch), 1, 1)
    dict_info = dict()
    dict_info['img'] = torch.cat(imgs_batch, dim=0)
    dict_info['points'] = lidar_data_list_batch
    dict_info['gt_bboxes_3d'] = gt_boxes_batch
    dict_info['gt_labels_3d'] = gt_labels_batch
    dict_info['gt_masks_bev'] = None
    dict_info['camera_intrinsics'] = torch.cat(intrin_mats_batch, dim=0)
    dict_info['camera2ego'] = torch.cat(sensor2ego_mats_batch, dim=0)
    dict_info['lidar2ego'] = identity_matrix_batch  
    dict_info['lidar2camera'] = torch.stack(lidar2camera_batch)
    dict_info['camera2lidar'] = torch.cat(sensor2ego_mats_batch, dim=0)
    dict_info['lidar2image'] = torch.stack(lidar2image_batch)
    dict_info['img_aug_matrix'] = torch.cat(ida_mats_batch, dim=0) 
    dict_info['lidar_aug_matrix'] = identity_matrix_batch  
    dict_info['denorms'] = torch.stack(denorms_batch)
    dict_info['sensor2virtual'] = torch.cat(sensor2virtual_mats_batch, dim =0)
    dict_info['reference_heights'] = torch.cat(reference_heights_batch, dim =0)
    dict_info['metas'] = img_metas_batch
    return dict_info