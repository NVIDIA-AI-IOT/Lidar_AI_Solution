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
import numpy as np

class Backend:
    PangoCairo  = 1
    StbTrueType = 2

class ClockFormat:
    YYMMDD_HHMMSS = 1
    YYMMDD = 2
    HHMMSS = 3

class ImageFormat:
    RGB  = 1
    RGBA = 2
    BlockLinearNV12 = 3
    PitchLinearNV12 = 4

class Context:
    def __init__(self):
        raise NotImplementedError("This object is implemented must be in the binary library")

    def set_backend(self, text_backend:Backend): ...
    def measure_text(self, utf8_text, font_size, font)->Tuple[int, int]: ...
    def text(self, utf8_text, font_size, font, x, y, border_color, bg_color): ...
    def clock(self, format, time, font_size, font, x, y, border_color, bg_color): ...
    def line(self, x0, y0, x1, y1, thickness, color, interpolation): ...
    def arrow(self, x0, y0, x1, y1, arrow_size, thickness, color, interpolation): ...
    def point(self, cx, cy, radius, color): ...
    def circle(self, cx, cy, radius, thickness, border_color, bg_color): ...
    def rectangle(self, left, top, right, bottom, thickness, border_color, bg_color): ...
    def boxblur(self, left, top, right, bottom, kernel_size): ...
    def rotationbox(self, cx, cy, width, height, yaw, thickness, border_color, interpolation, bg_color): ...
    def segmentmask(self, left, top, right, bottom, thickness, d_seg, seg_width, seg_height, seg_threshold, border_color, seg_color): ...
    def polyline(self, h_pts, d_pts, n_pts, thickness, is_closed, border_color, interpolation, fill_color): ...
    def rgba_source(self, d_src, cx, cy, w, h): ...
    def nv12_source(self, d_src0, d_src1, cx, cy, w, h, mask_color, block_linear): ...
    def apply(self, data0, data1, width, stride, height, format, stream, gpu_memory, launch): ...
    def launch(self, data0, data1, width, stride, height, format, stream, gpu_memory): ...

def exit_cleanup(): ...    

from .pycuosd import *

def apply_to_numpy(self, image:np.ndarray, stream=0):
    width, height = image.shape[1], image.shape[0]
    self.apply(image.ctypes.data, 0, width, width * 3, height, ImageFormat.RGB, stream, False, True)

Context.__call__ = apply_to_numpy

import atexit
atexit.register(exit_cleanup)