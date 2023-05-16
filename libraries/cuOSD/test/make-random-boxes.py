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

import random
import numpy as np


def max_iou_of(query, library):

    l, t, r, b = np.array(library).T
    ql, qt, qr, qb = query

    cl = np.maximum(l, ql)
    ct = np.maximum(t, qt)
    cr = np.minimum(r, qr)
    cb = np.minimum(b, qb)

    cw = np.maximum(0, cr - cl + 1)
    ch = np.maximum(0, cb - ct + 1)
    carea = cw * ch
    iou = carea / ((r-l+1) * (b-t+1) + (qr-ql+1) * (qb-qt+1) - carea)
    return iou.max()


random.seed(31)

with open("std-random-boxes.txt", "w") as f:

    image_width  = 1280
    image_height = 720
    # image_width  = 1920
    # image_height = 1080
    thickness    = 1
    font_size    = 5
    f.write(f"{image_width},{image_height}\n\n")
    f.write("# type, left, top, right, bottom, thickness, name, confidence, font_size\n")

    box_count  = 50
    no_overlap = True
    box_width  = 50, 100
    box_height = 50, 100
    namelist   = [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush' ]

    boxes = []
    while len(boxes) < box_count:
        width  = int(random.random() * (box_width[1]  - box_width[0])  + box_width[0])
        height = int(random.random() * (box_height[1] - box_height[0]) + box_height[0])
        left = int(random.random() * (image_width  - width - font_size * 2)) + font_size * 2
        top  = int(random.random() * (image_height - height - font_size * 2)) + font_size * 2
        box = np.array([left, top, left + width, top + height])

        if len(boxes) > 0 and no_overlap and max_iou_of(box, boxes) > 0.0:
            continue

        name   = namelist[random.randint(0, len(namelist) - 1)]
        confidence = random.random()
        f.write(f"detbox,{left},{top},{left+width},{top+height},{thickness},{name},{confidence:.2f},{font_size}\n")
        boxes.append(box)