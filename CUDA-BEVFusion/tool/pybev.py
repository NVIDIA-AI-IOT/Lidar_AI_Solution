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
import cv2
import numpy as np
import tensor
import libpybev

model = os.environ["DEBUG_MODEL"]
precision = os.environ["DEBUG_PRECISION"]
data  = os.environ["DEBUG_DATA"]

image_names = [
    "0-FRONT.jpg",
    "1-FRONT_RIGHT.jpg",
    "2-FRONT_LEFT.jpg",
    "3-BACK.jpg",
    "4-BACK_LEFT.jpg",
    "5-BACK_RIGHT.jpg"
]

images = []
for file in image_names:
    if(file.endswith(".jpg")):
        image = cv2.imread(f"{data}/{file}")
        image = image[..., ::-1]
        images.append(image)

images = np.stack(images, axis=0)[None]

camera_intrinsics = tensor.load(f"{data}/camera_intrinsics.tensor")
camera2lidar = tensor.load(f"{data}/camera2lidar.tensor")
lidar2image = tensor.load(f"{data}/lidar2image.tensor")
img_aug_matrix = tensor.load(f"{data}/img_aug_matrix.tensor")
points = tensor.load(f"{data}/points.tensor")

core = libpybev.load_bevfusion(
    f"model/{model}/build/camera.backbone.plan",
    f"model/{model}/build/camera.vtransform.plan",
    f"model/{model}/lidar.backbone.xyz.onnx",
    f"model/{model}/build/fuser.plan",
    f"model/{model}/build/head.bbox.plan",
    precision
)

if core is None:
    print("Failed to create core")
    exit(0)

core.print()

core.update(
    camera2lidar,
    camera_intrinsics,
    lidar2image,
    img_aug_matrix
)

# while True:
boxes = core.forward(images, points, with_normalization=True, with_dlpack=False)

np.set_printoptions(3, suppress=True, linewidth=300)
print(boxes[:10])