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
import json
import math

import numpy as np
from tqdm import tqdm

from evaluators.utils import *
from evaluators.kitti_utils import *
from evaluators.kitti_utils import kitti_common as kitti
from evaluators.kitti_utils.eval import kitti_eval

from scripts.gen_info_rope3d import *
from scipy.spatial.transform import Rotation as R

category_map_dair = {"car": "Car", "van": "Car", "truck": "Car", "bus": "Car", "pedestrian": "Pedestrian", "bicycle": "Cyclist", "trailer": "Cyclist", "motorcycle": "Cyclist"}
category_map_rope3d = {"car": "Car", "van": "Car", "truck": "Bus", "bus": "Bus", "pedestrian": "Pedestrian", "bicycle": "Cyclist", "trailer": "Cyclist", "motorcycle": "Cyclist"}

def get_lidar_3d_8points(obj_size, yaw_lidar, center_lidar):
    center_lidar = [float(center_lidar[0]), float(center_lidar[1]), float(center_lidar[2])]
    lidar_r = np.matrix([[math.cos(yaw_lidar), -math.sin(yaw_lidar), 0], [math.sin(yaw_lidar), math.cos(yaw_lidar), 0], [0, 0, 1]])
    l, w, h = obj_size
    center_lidar[2] = center_lidar[2] - h / 2
    corners_3d_lidar = np.matrix(
        [
            [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2],
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
            [0, 0, 0, 0, h, h, h, h],
        ]
    )
    corners_3d_lidar = lidar_r * corners_3d_lidar + np.matrix(center_lidar).T
    return corners_3d_lidar.T

def read_label_bboxes(label_path, Tr_cam2lidar):
    # _, _, Tr_cam2lidar, _ = get_cam2lidar(denorm_file)
    fieldnames = ['type', 'truncated', 'occluded', 'alpha', 'xmin', 'ymin', 'xmax', 'ymax', 'dh', 'dw',
                  'dl', 'lx', 'ly', 'lz', 'ry']
    boxes = []
    with open(label_path, 'r') as csv_file:
        reader = csv.DictReader(csv_file, delimiter=' ', fieldnames=fieldnames)
        for line, row in enumerate(reader):
            alpha = float(row["alpha"])
            pos = np.array((float(row['lx']), float(row['ly']), float(row['lz'])), dtype=np.float32)
            dim = [float(row['dl']), float(row['dw']), float(row['dh'])]
            ry = float(row["ry"])
            if alpha > np.pi:
                alpha -= 2 * np.pi
                ry = alpha2roty(alpha, pos)
            alpha = clip2pi(alpha)
            ry = clip2pi(ry)
            yaw_lidar =  0.5 * np.pi - ry
            if sum(dim) == 0:
                continue
            loc_cam = np.array([float(row['lx']), float(row['ly']), float(row['lz']), 1.0]).reshape(4, 1)
            loc_lidar = np.matmul(Tr_cam2lidar, loc_cam).squeeze(-1)[:3]
            loc_lidar[2] += 0.5 * float(row['dh'])
            center_lidar, obj_size = loc_lidar, dim
            box = get_lidar_3d_8points(obj_size, yaw_lidar, center_lidar)
            boxes.append(box)
    return boxes

def kitti_evaluation(pred_label_path, gt_label_path, current_classes=["Car", "Pedestrian", "Cyclist"], metric_path="metric"):
    pred_annos, image_ids = kitti.get_label_annos(pred_label_path, return_ids=True)
    gt_annos = kitti.get_label_annos(gt_label_path, image_ids=image_ids)
    print(len(pred_annos), len(gt_annos))
    result, ret_dict = kitti_eval(gt_annos, pred_annos, current_classes=current_classes, metric="R40")
    mAP_3d_moderate = ret_dict["KITTI/Car_3D_moderate_strict"]
    os.makedirs(os.path.join(metric_path, "R40"), exist_ok=True)
    with open(os.path.join(metric_path, "R40", 'epoch_result_{}.txt'.format(round(mAP_3d_moderate, 2))), "w") as f:
        f.write(result)
    print(result)
    return mAP_3d_moderate, result

def write_kitti_in_txt(pred_lines, path_txt):
    wf = open(path_txt, "w")
    for line in pred_lines:
        line_string = " ".join(line) + "\n"
        wf.write(line_string)
    wf.close()

def get_velo2cam(src_denorm_file):
    _, _, Tr_cam2lidar, _ = get_cam2lidar(src_denorm_file)
    Tr_velo_to_cam = np.linalg.inv(Tr_cam2lidar) 
    r_velo2cam, t_velo2cam = Tr_velo_to_cam[:3, :3], Tr_velo_to_cam[:3, 3]
    t_velo2cam = t_velo2cam.reshape(3, 1)
    return Tr_velo_to_cam, r_velo2cam, t_velo2cam

def convert_point(point, matrix):
    pos =  matrix @ point
    return pos[0], pos[1], pos[2]

def normalize_angle(angle):
    alpha_tan = np.tan(angle)
    alpha_arctan = np.arctan(alpha_tan)
    if np.cos(angle) < 0:
        alpha_arctan = alpha_arctan + math.pi
    return alpha_arctan

def get_camera_3d_8points(obj_size, yaw_lidar, center_lidar, center_in_cam, r_velo2cam, t_velo2cam):
    liadr_r = np.matrix([[math.cos(yaw_lidar), -math.sin(yaw_lidar), 0], [math.sin(yaw_lidar), math.cos(yaw_lidar), 0], [0, 0, 1]])
    l, w, h = obj_size
    corners_3d_lidar = np.matrix(
        [
            [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2],
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
            [0, 0, 0, 0, h, h, h, h],
        ]
    )
    corners_3d_lidar = liadr_r * corners_3d_lidar + np.matrix(center_lidar).T
    corners_3d_cam = r_velo2cam * corners_3d_lidar + t_velo2cam
    
    x0, z0 = corners_3d_cam[0, 0], corners_3d_cam[2, 0]
    x3, z3 = corners_3d_cam[0, 3], corners_3d_cam[2, 3]
    dx, dz = x0 - x3, z0 - z3
    yaw = math.atan2(-dz, dx)
    alpha = yaw - math.atan2(center_in_cam[0], center_in_cam[2])
    if alpha > math.pi:
        alpha = alpha - 2.0 * math.pi
    if alpha <= (-1 * math.pi):
        alpha = alpha + 2.0 * math.pi
    alpha_arctan = normalize_angle(alpha)
    return alpha_arctan, yaw

def pcd_vis(boxes, save_file="demo.jpg", label_path=None, Tr_velo_to_cam=None):    
    range_list = [(-60, 60), (0, 100), (-2., -2.), 0.1]
    points_filter = PointCloudFilter(side_range=range_list[0], fwd_range=range_list[1], res=range_list[-1])
    bev_image = points_filter.get_meshgrid()
    bev_image = cv2.merge([bev_image, bev_image, bev_image])
    for n in range(len(boxes)):
        corner_points = boxes[n]
        x_img, y_img = points_filter.pcl2xy_plane(corner_points[:, 0], corner_points[:, 1])
        x_img = x_img[:, 0]
        y_img = y_img[:, 0]
        for i in np.arange(4):
            cv2.line(bev_image, (int(x_img[0]), int(y_img[0])), (int(x_img[1]), int(y_img[1])), (255,0,0), 2)
            cv2.line(bev_image, (int(x_img[0]), int(y_img[0])), (int(x_img[3]), int(y_img[3])), (255,0,0), 2)
            cv2.line(bev_image, (int(x_img[1]), int(y_img[1])), (int(x_img[2]), int(y_img[2])), (255,0,0), 2)
            cv2.line(bev_image, (int(x_img[2]), int(y_img[2])), (int(x_img[3]), int(y_img[3])), (255,0,0), 2)
    if label_path is not None:
        denorm_file = label_path.replace("label_2", "denorm")
        Tr_cam2lidar = np.linalg.inv(Tr_velo_to_cam)
        boxes = read_label_bboxes(label_path, Tr_cam2lidar)
        for n in range(len(boxes)):
            corner_points = boxes[n]
            x_img, y_img = points_filter.pcl2xy_plane(corner_points[:, 0], corner_points[:, 1])
            x_img = x_img[:, 0]
            y_img = y_img[:, 0]
            for i in np.arange(4):
                cv2.line(bev_image, (int(x_img[0]), int(y_img[0])), (int(x_img[1]), int(y_img[1])), (0,0,255), 2)
                cv2.line(bev_image, (int(x_img[0]), int(y_img[0])), (int(x_img[3]), int(y_img[3])), (0,0,255), 2)
                cv2.line(bev_image, (int(x_img[1]), int(y_img[1])), (int(x_img[2]), int(y_img[2])), (0,0,255), 2)
                cv2.line(bev_image, (int(x_img[2]), int(y_img[2])), (int(x_img[3]), int(y_img[3])), (0,0,255), 2)
    cv2.imwrite(save_file, bev_image)
    
def bbbox2bbox(box3d, Tr_velo_to_cam, camera_intrinsic, img_size=[1920, 1080]):
    corners_3d = np.array(box3d)
    corners_3d_extend = np.concatenate(
        [corners_3d, np.ones((corners_3d.shape[0], 1), dtype=np.float32)], axis=1) 
    corners_3d_extend = np.matmul(Tr_velo_to_cam, corners_3d_extend.transpose(1, 0))
        
    corners_2d = np.matmul(camera_intrinsic, corners_3d_extend)
    corners_2d = corners_2d[:2] / corners_2d[2]
    box2d = np.array([min(corners_2d[0]), min(corners_2d[1]),
                      max(corners_2d[0]), max(corners_2d[1])])
    
    # [xmin, ymin, xmax, ymax]
    box2d[0] = max(box2d[0], 0.0)
    box2d[1] = max(box2d[1], 0.0)
    box2d[2] = min(box2d[2], img_size[0])
    box2d[3] = min(box2d[3], img_size[1])
    return box2d

def read_json(path):
    with open(path, "r") as f:
        my_json = json.load(f)
        return my_json
        
def get_lidar2cam(calib_path):
    my_json = read_json(calib_path)
    if "Tr_velo_to_cam" in my_json.keys():
        velo2cam = np.array(my_json["Tr_velo_to_cam"]).reshape(3, 4)
        r_velo2cam = velo2cam[:, :3]
        t_velo2cam = velo2cam[:, 3].reshape(3, 1)
    else:
        r_velo2cam = np.array(my_json["rotation"])
        t_velo2cam = np.array(my_json["translation"])
    Tr_velo_to_cam = np.eye(4)
    Tr_velo_to_cam[:3,:3] = r_velo2cam
    Tr_velo_to_cam[:3,3] = t_velo2cam.flatten()
    return Tr_velo_to_cam, r_velo2cam, t_velo2cam

def get_cam_calib_intrinsic(calib_path):
    my_json = read_json(calib_path)
    cam_K = my_json["cam_K"]
    calib = np.array(cam_K).reshape([3, 3], order="C")
    return calib
    
def result2kitti(results_file, results_path, dair_root, gt_label_path, demo=False):
    with open(results_file,'r',encoding='utf8')as fp:
        results = json.load(fp)["results"]
    for sample_token in tqdm(results.keys()):
        sample_id = int(sample_token.split("/")[1].split(".")[0])
        camera_intrinsic_file = os.path.join(dair_root, "calib/camera_intrinsic", "{:06d}".format(sample_id) + ".json")
        virtuallidar_to_camera_file = os.path.join(dair_root, "calib/virtuallidar_to_camera", "{:06d}".format(sample_id) + ".json")
        camera_intrinsic = get_cam_calib_intrinsic(camera_intrinsic_file)
        Tr_velo_to_cam, r_velo2cam, t_velo2cam = get_lidar2cam(virtuallidar_to_camera_file)
        camera_intrinsic = np.concatenate([camera_intrinsic, np.zeros((camera_intrinsic.shape[0], 1))], axis=1)
        preds = results[sample_token]
        pred_lines = []
        bboxes = []
        for pred in preds:
            loc = pred["translation"]
            dim = pred["size"]
            yaw_lidar = pred["box_yaw"]
            detection_score = pred["detection_score"]
            class_name = pred["detection_name"]
            
            w, l, h = dim[0], dim[1], dim[2]
            x, y, z = loc[0], loc[1], loc[2]            
            bottom_center = [x, y, z]
            obj_size = [l, w, h]
            bottom_center_in_cam = r_velo2cam * np.matrix(bottom_center).T + t_velo2cam
            alpha, yaw = get_camera_3d_8points(
                obj_size, yaw_lidar, bottom_center, bottom_center_in_cam, r_velo2cam, t_velo2cam
            )
            yaw  = 0.5 * np.pi - yaw_lidar

            cam_x, cam_y, cam_z = convert_point(np.array([x, y, z, 1]).T, Tr_velo_to_cam)

            box = get_lidar_3d_8points([w, l, h], yaw_lidar, [x, y, z + h/2])
            box2d = bbbox2bbox(box, Tr_velo_to_cam, camera_intrinsic)
            if detection_score > 0.45 and class_name in category_map_dair.keys():
                i1 = category_map_dair[class_name]
                i2 = str(0)
                i3 = str(0)
                i4 = str(round(alpha, 4))
                i5, i6, i7, i8 = (
                    str(round(box2d[0], 4)),
                    str(round(box2d[1], 4)),
                    str(round(box2d[2], 4)),
                    str(round(box2d[3], 4)),
                )
                i9, i11, i10 = str(round(h, 4)), str(round(w, 4)), str(round(l, 4))
                i12, i13, i14 = str(round(cam_x, 4)), str(round(cam_y, 4)), str(round(cam_z, 4))
                i15 = str(round(yaw, 4))
                line = [i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, str(round(detection_score, 4))]
                pred_lines.append(line)
                bboxes.append(box)
        os.makedirs(os.path.join(results_path, "data"), exist_ok=True)
        write_kitti_in_txt(pred_lines, os.path.join(results_path, "data", "{:06d}".format(sample_id) + ".txt"))       
        if demo:
            os.makedirs(os.path.join(results_path, "demo"), exist_ok=True)
            label_path = os.path.join(gt_label_path, "{:06d}".format(sample_id) + ".txt")
            demo_file = os.path.join(results_path, "demo", "{:06d}".format(sample_id) + ".jpg")
            pcd_vis(bboxes, demo_file, label_path, Tr_velo_to_cam)
    return os.path.join(results_path, "data")

def result2kitti_rope3d(results_file, results_path, dair_root, gt_label_path, demo=False):
    with open(results_file,'r',encoding='utf8')as fp:
        results = json.load(fp)["results"]
    with open("data/rope3d-kitti/map_token2id.json") as fp:
        token2sample = json.load(fp)

    for sample_token in tqdm(results.keys()):
        sample_id = int(token2sample[sample_token])
        src_denorm_file = os.path.join(dair_root, "training/denorm", sample_token + ".txt")
        src_calib_file = os.path.join(dair_root, "training/calib", sample_token + ".txt")
        if not os.path.exists(src_denorm_file):
            src_denorm_file = os.path.join(dair_root, "validation/denorm", sample_token + ".txt")
            src_calib_file = os.path.join(dair_root, "validation/calib", sample_token + ".txt")
  
        Tr_velo_to_cam, r_velo2cam, t_velo2cam = get_velo2cam(src_denorm_file)
        camera_intrinsic = load_calib(src_calib_file)
        camera_intrinsic = np.concatenate([camera_intrinsic, np.zeros((camera_intrinsic.shape[0], 1))], axis=1)
        preds = results[sample_token]
        pred_lines = []
        bboxes = []
        for pred in preds:
            loc = pred["translation"]
            dim = pred["size"]
            yaw_lidar = pred["box_yaw"]
            detection_score = pred["detection_score"]
            class_name = pred["detection_name"]
            
            w, l, h = dim[0], dim[1], dim[2]
            x, y, z = loc[0], loc[1], loc[2]            
            bottom_center = [x, y, z]
            obj_size = [l, w, h]
            bottom_center_in_cam = r_velo2cam * np.matrix(bottom_center).T + t_velo2cam
            alpha, yaw = get_camera_3d_8points(
                obj_size, yaw_lidar, bottom_center, bottom_center_in_cam, r_velo2cam, t_velo2cam
            )
            yaw  = 0.5 * np.pi - yaw_lidar

            cam_x, cam_y, cam_z = convert_point(np.array([x, y, z, 1]).T, Tr_velo_to_cam)
            box = get_lidar_3d_8points([w, l, h], yaw_lidar, [x, y, z + h/2])
            box2d = bbbox2bbox(box, Tr_velo_to_cam, camera_intrinsic)
            if detection_score > 0.45 and class_name in category_map_rope3d.keys():
                i1 = category_map_rope3d[class_name]
                i2 = str(0)
                i3 = str(0)
                i4 = str(round(alpha, 4))
                i5, i6, i7, i8 = (
                    str(round(box2d[0], 4)),
                    str(round(box2d[1], 4)),
                    str(round(box2d[2], 4)),
                    str(round(box2d[3], 4)),
                )
                i9, i11, i10 = str(round(h, 4)), str(round(w, 4)), str(round(l, 4))
                i12, i13, i14 = str(round(cam_x, 4)), str(round(cam_y, 4)), str(round(cam_z, 4))
                i15 = str(round(yaw, 4))
                line = [i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, str(round(detection_score, 4))]
                pred_lines.append(line)
                bboxes.append(box)
        os.makedirs(os.path.join(results_path, "data"), exist_ok=True)
        write_kitti_in_txt(pred_lines, os.path.join(results_path, "data", "{:06d}".format(sample_id) + ".txt"))       
        if demo:
            os.makedirs(os.path.join(results_path, "demo"), exist_ok=True)
            label_path = os.path.join(gt_label_path, "{:06d}".format(sample_id) + ".txt")
            demo_file = os.path.join(results_path, "demo", "{:06d}".format(sample_id) + ".jpg")
            pcd_vis(bboxes, demo_file, label_path)
    return os.path.join(results_path, "data")