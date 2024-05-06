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

import json
import os
import shutil
import glob
import argparse
 

def load_json(path):
    with open(path) as file:
        data = json.load(file)
    return data


def make_merge_dir(data_path): 
    os.makedirs(data_path, exist_ok=True)   
    os.makedirs(os.path.join(data_path, 'calib'), exist_ok=True)
    os.makedirs(os.path.join(data_path, 'calib/camera_intrinsic'), exist_ok=True)
    os.makedirs(os.path.join(data_path, 'calib/virtuallidar_to_camera'), exist_ok=True)
    os.makedirs(os.path.join(data_path, 'image'), exist_ok=True)
    os.makedirs(os.path.join(data_path, 'label'), exist_ok=True)
    os.makedirs(os.path.join(data_path, 'label/camera'), exist_ok=True)
    os.makedirs(os.path.join(data_path, 'label/virtuallidar'), exist_ok=True)    
    os.makedirs(os.path.join(data_path, 'velodyne'), exist_ok=True)   


def merge_v2x_v2xseq(json_path, data_folder_path, merge_folder_path, data_type, start_idx, train_dataset, val_dataset):
    json_data = load_json(json_path)
    for item in json_data:
        if data_type == 'v2x_seq':
            frame_id = item.get('frame_id')
            sequence_id = item.get('sequence_id')
        else:
            frame_id = item.get('image_path').split('/')[1].split('.')[0]
            
        new_frame_id = str(int(frame_id) + start_idx).zfill(6)

        new_image_path = item.get('image_path').replace(frame_id, new_frame_id)
        new_pointcloud_path = item.get('pointcloud_path').replace(frame_id, new_frame_id)
        new_calib_camera_intrinsic_path = item.get('calib_camera_intrinsic_path').replace(frame_id, new_frame_id)                
        new_calib_virtuallidar_to_camera_path = item.get('calib_virtuallidar_to_camera_path').replace(frame_id, new_frame_id)            
        new_label_camera_std_path = item.get('label_camera_std_path').replace(frame_id, new_frame_id)                            
        new_label_lidar_std_path = item.get('label_lidar_std_path').replace(frame_id, new_frame_id)    
                                                                                                                       
        shutil.copy(os.path.join(data_folder_path, item.get('image_path')), os.path.join(merge_folder_path, new_image_path))
        shutil.copy(os.path.join(data_folder_path, item.get('pointcloud_path')), os.path.join(merge_folder_path, new_pointcloud_path))
        shutil.copy(os.path.join(data_folder_path, item.get('calib_camera_intrinsic_path')), os.path.join(merge_folder_path, new_calib_camera_intrinsic_path))
        shutil.copy(os.path.join(data_folder_path, item.get('calib_virtuallidar_to_camera_path')), os.path.join(merge_folder_path, new_calib_virtuallidar_to_camera_path))
        shutil.copy(os.path.join(data_folder_path, item.get('label_camera_std_path')), os.path.join(merge_folder_path, new_label_camera_std_path))
        shutil.copy(os.path.join(data_folder_path, item.get('label_lidar_std_path')), os.path.join(merge_folder_path, new_label_lidar_std_path))                                            
        
        if data_type == 'v2x_seq':
            if sequence_id == '0000' or sequence_id == '0054' or sequence_id == "0066" or  sequence_id == "0089":
                val_dataset.append(new_frame_id)
            else:
                train_dataset.append(new_frame_id)    
        else:
            train_dataset.append(new_frame_id)
            
            
        if data_type == 'v2x_seq':
            item['image_path'] = new_image_path
            item['pointcloud_path'] = new_pointcloud_path
            item['calib_camera_intrinsic_path'] = new_calib_camera_intrinsic_path
            item['calib_virtuallidar_to_camera_path'] = new_calib_virtuallidar_to_camera_path
            item['label_camera_std_path'] = new_label_camera_std_path
            item['label_lidar_std_path'] = new_label_lidar_std_path
            item['calib_virtuallidar_to_world_path'] = item.get('calib_virtuallidar_to_world_path').replace(frame_id, new_frame_id)  
            item['frame_id'] = new_frame_id
    
    return json_data
   

# Merge the roadside data from dair-v2x-i and dair-v2x-seq, and select some of the data as test data. 
# Training data:
#              ├── DAIR-V2X-I
#              ├── V2X-Seq (excluding sequences "0000", "0054", "0066", and "0089")
# Testing data:
#              ├── V2X-Seq (sequences "0000", "0054", "0066", and "0089")

# Directory structure
# ├── data
# │   ├── dair-v2x-i
# │   │   ├── velodyne
# │   │   ├── image
# │   │   ├── calib
# │   │   ├── label
# |   |   └── data_info.json
# │   ├── dair-v2x-seq # only use the infrastructure side file !!!!!!!!!! 
# │   │   ├── velodyne
# │   │   ├── image
# │   │   ├── calib
# │   │   ├── label
# |   |   └── data_info.json
# |   |   └── single-infrastructure-split-data.json
# │   ├── dair-v2x-merge  
# │   │   ├── velodyne
# │   │   ├── image
# │   │   ├── calib
# │   │   ├── label
# |   |   └── data_info.json            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data Merge Script')
    parser.add_argument('--merge_folder', type=str, default='/dataset/dair-v2x-merge', help='dair-v2x-merge')
    parser.add_argument('--v2x_seq_folder', type=str, default='/dataset/dair-v2x-seq', help='dair-v2x-seq folder')
    parser.add_argument('--v2x_i_folder', type=str, default='/dataset/dair-v2x-i', help='dair-v2x-i folder')
    args = parser.parse_args()
    
    make_merge_dir(args.merge_folder)
    v2x_seq_json_path = os.path.join(args.v2x_seq_folder,'data_info.json')
    v2x_i_json_path = os.path.join(args.v2x_i_folder,'data_info.json')
    
    train_dataset = []
    val_dataset =[]
    
    v2x_i_json = merge_v2x_v2xseq(v2x_i_json_path, args.v2x_i_folder, args.merge_folder, 'v2x_i', 0, train_dataset, val_dataset)
    v2x_seq_json = merge_v2x_v2xseq(v2x_seq_json_path, args.v2x_seq_folder, args.merge_folder, 'v2x_seq', 20000, train_dataset, val_dataset)

    v2x_merge_json = v2x_i_json + v2x_seq_json
    v2x_merge_json_json_data = json.dumps(v2x_merge_json)
    with open(os.path.join(args.merge_folder, 'data_info.json'), "w") as file:
        file.write(v2x_merge_json_json_data)


    train_val_data= {"train": train_dataset, "val":val_dataset}
    json_data = json.dumps(train_val_data, indent=4)

    print(len(train_dataset), len(val_dataset),len(train_dataset)+len(val_dataset))
    with open(os.path.join(args.merge_folder, 'single-infrastructure-split-data.json'), "w") as file:
        file.write(json_data)
