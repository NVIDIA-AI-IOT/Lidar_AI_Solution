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
import os
import re

input_sizes  = [[3840, 2160], [2160, 1440]]
output_sizes = [[1920, 1280], [1080, 720]]
input_batch  = 1
formats = "BL,PL,YUYV".split(",")
interps = "bilinear,nearest".split(",")
dtypes  = "uint8,float32,float16".split(",")
layouts = "NCHW16_BGR,NCHW16_RGB,NCHW_BGR,NCHW_RGB,NHWC_BGR,NHWC_RGB".split(",")

for (input_width, input_height), (output_width, output_height), format, interp, dtype, layout in itertools.product(input_sizes, output_sizes, formats, interps, dtypes, layouts):

    file = f"workspace/{format}-{interp}-{dtype}-{layout}-{input_width}x{input_height}x{input_batch}-{output_width}x{output_height}.binary"
    cmd = f"./yuvtorgb --input={input_width}x{input_height}x{input_batch}/{format} \
            --output={output_width}x{output_height}/{dtype}/{layout} --interp={interp} \
            --perf"

    with os.popen(cmd) as p:
        perf = float(re.findall("performance: ([0-9.]+) us", p.readlines()[0])[0])

    print(f"{interp}: {input_width}x{input_height}x{input_batch}/{format} to {output_width}x{output_height}/{dtype}/{layout} perf: {perf:.2f} us")