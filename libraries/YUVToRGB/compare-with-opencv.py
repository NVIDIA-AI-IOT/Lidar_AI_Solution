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

import numpy as np
import itertools
import cv2
import os
import re

data  = open("workspace/data/nv12_3840x2160.yuv", "rb").read()
array = np.frombuffer(data, np.uint8, len(data)).reshape(2160 * 3 // 2, 3840)
nv12_base_image = cv2.cvtColor(array, cv2.COLOR_YUV2BGR_NV12)

data  = open("workspace/data/yuyv_3840x2160_yuyv422.yuv", "rb").read()
array = np.frombuffer(data, np.uint8, len(data)).reshape(2160, 3840, 2)
yuyv_base_image = cv2.cvtColor(array, cv2.COLOR_YUV2BGR_YUYV)

def get_cv_image(iwidth, iheight, owidth, oheight, interp, format):

    base = nv12_base_image
    if format == "YUYV":
        base = yuyv_base_image

    interpolations = {"nearest": cv2.INTER_NEAREST, "bilinear": cv2.INTER_LINEAR}
    return cv2.resize(base[:iheight, :iwidth], (owidth, oheight), interpolation=interpolations[interp])

def load_tensor(file):

    data = open(file, "rb").read()
    magic_number, w, h, c, b, layout, dtype_idx = np.frombuffer(data, np.uint32, 7, 0)

    assert magic_number == 0xAABBCCEF, "Invalid tensor file"

    dtype_list = [None,np.uint8,np.float32,np.float16]
    dtype      = dtype_list[dtype_idx]
    array      = np.frombuffer(data, dtype, w * h * c * b, 28)
    
    if layout in [3, 4, 5, 6, 7, 8, 9, 10]:
        array = array.reshape(b, h, w, c)
    elif layout in [1, 2]:
        array = array.reshape(b, c, h, w)
    return array, layout

def save_tensor_to_image(tensor, layout, file):

    if layout in [3, 4]:
        image = tensor.astype(np.uint8)
    elif layout in [5, 6, 7, 8, 9, 10]:
        image = tensor[..., :3].astype(np.uint8)
    else:
        image = tensor[:3].transpose(1, 2, 0).astype(np.uint8)

    if layout in [1, 3, 5, 7, 9]:
        image = image[..., ::-1]

    cv2.imwrite(file, image)
    return image

def fformat(fmt, vars):
    for name in re.findall("{([\S]*?)}", fmt):
        fmt = fmt.replace("{" + name + "}", str(vars[name]))
    return fmt

input_sizes  = [[3840, 2160], [2160, 1440]]
output_sizes = [[1920, 1280], [1080, 720]]
input_batch    = 1
formats = "BL,PL,YUYV".split(",")
interps = "nearest,bilinear".split(",")
dtypes  = "uint8,float32,float16".split(",")
layouts = "NCHW16_BGR,NCHW16_RGB,NCHW32_BGR,NCHW32_RGB,NCHW4_BGR,NCHW4_RGB,NCHW_BGR,NCHW_RGB,NHWC_BGR,NHWC_RGB".split(",")

for (input_width, input_height), (output_width, output_height), format, interp, dtype, layout in itertools.product(input_sizes, output_sizes, formats, interps, dtypes, layouts):

    file = fformat("workspace/{format}-{interp}-{dtype}-{layout}-{input_width}x{input_height}x{input_batch}-{output_width}x{output_height}.binary", locals())
    cmd  = fformat("./yuvtorgb --input={input_width}x{input_height}x{input_batch}/{format} \
            --output={output_width}x{output_height}/{dtype}/{layout} --interp={interp} \
            --save={file}", locals())

    with os.popen(cmd) as p:
        # perf = float(re.findall("performance: ([0-9.]+) us", p.readlines()[0])[0])
        pass

    image, int_layout = load_tensor(file)
    for i in range(image.shape[0]):
        prefix = fformat("{format}-{interp}-{dtype}-{layout}-{input_width}x{input_height}x{input_batch}-{output_width}x{output_height}", locals())
        output_image = save_tensor_to_image(image[i], int_layout, fformat("workspace/{prefix}.png", locals()))

        oheight, owidth = output_image.shape[:2]
        cv_base = get_cv_image(input_width, input_height, owidth, oheight, interp, format)
        diff    = (np.abs(cv_base.astype(int) - output_image.astype(int))).astype(np.uint8)

        diff_sum  = np.sum(diff)
        diff_mean = np.mean(diff)
        diff_max  = np.max(diff)
        diff_std  = np.std(diff)
        print(fformat("{interp}: {input_width}x{input_height}x{input_batch}/{format} to {output_width}x{output_height}/{dtype}/{layout}, diff sum = {diff_sum}, avg = {diff_mean}, max = {diff_max}, std = {diff_std}", locals()))

        if np.sum(diff) != 0:
            diff = (diff * 255).astype(np.uint8)
            cv2.imwrite(fformat("workspace/{prefix}-diff.png", locals()), diff)
            cv2.imwrite(fformat("workspace/{prefix}-cv.png", locals()), cv_base)