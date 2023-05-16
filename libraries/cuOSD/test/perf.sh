#!/bin/bash
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

./cuosd perf --seed=31 --input=1280x720/PL --font=workspace/my.nvfont --text=100 --save=workspace/text.png
./cuosd perf --seed=31 --input=1280x720/PL --line=100 --save=workspace/line.png
./cuosd perf --seed=31 --input=1280x720/PL --circle=100 --save=workspace/circle.png
./cuosd perf --seed=31 --input=1280x720/PL --rectangle=100 --save=workspace/rectangle.png
./cuosd perf --seed=31 --input=1280x720/PL --arrow=100 --save=workspace/arrow.png
./cuosd perf --seed=31 --input=1280x720/PL --point=100 --save=workspace/point.png
./cuosd perf --seed=31 --input=1280x720/PL --font=workspace/my.nvfont --clock=100 --save=workspace/clock.png
./cuosd perf --seed=31 --input=1280x720/PL --font=workspace/my.nvfont --text=10 --line=10 --circle=10 --rectangle=10 --arrow=10 --point=10 --clock=10 --save=workspace/mix.png

./cuosd perf --seed=31 --input=1280x720/BL --font=workspace/my.nvfont --text=100 --save=workspace/text.png
./cuosd perf --seed=31 --input=1280x720/BL --line=100 --save=workspace/line.png
./cuosd perf --seed=31 --input=1280x720/BL --circle=100 --save=workspace/circle.png
./cuosd perf --seed=31 --input=1280x720/BL --rectangle=100 --save=workspace/rectangle.png
./cuosd perf --seed=31 --input=1280x720/BL --arrow=100 --save=workspace/arrow.png
./cuosd perf --seed=31 --input=1280x720/BL --point=100 --save=workspace/point.png
./cuosd perf --seed=31 --input=1280x720/BL --font=workspace/my.nvfont --clock=100 --save=workspace/clock.png
./cuosd perf --seed=31 --input=1280x720/BL --font=workspace/my.nvfont --text=10 --line=10 --circle=10 --rectangle=10 --arrow=10 --point=10 --clock=10 --save=workspace/mix.png


./cuosd perf --seed=31 --input=1280x720/RGBA --font=workspace/my.nvfont --text=100 --save=workspace/text.png
./cuosd perf --seed=31 --input=1280x720/RGBA --line=100 --save=workspace/line.png
./cuosd perf --seed=31 --input=1280x720/RGBA --circle=100 --save=workspace/circle.png
./cuosd perf --seed=31 --input=1280x720/RGBA --rectangle=100 --save=workspace/rectangle.png
./cuosd perf --seed=31 --input=1280x720/RGBA --arrow=100 --save=workspace/arrow.png
./cuosd perf --seed=31 --input=1280x720/RGBA --point=100 --save=workspace/point.png
./cuosd perf --seed=31 --input=1280x720/RGBA --font=workspace/my.nvfont --clock=100 --save=workspace/clock.png
./cuosd perf --seed=31 --input=1280x720/RGBA --font=workspace/my.nvfont --text=10 --line=10 --circle=10 --rectangle=10 --arrow=10 --point=10 --clock=10 --save=workspace/mix.png


./cuosd perf --seed=31 --input=640x640/RGBA --font=workspace/my.nvfont --text=100 --save=workspace/text.png
./cuosd perf --seed=31 --input=640x640/RGBA --line=100 --save=workspace/line.png
./cuosd perf --seed=31 --input=640x640/RGBA --circle=100 --save=workspace/circle.png
./cuosd perf --seed=31 --input=640x640/RGBA --rectangle=100 --save=workspace/rectangle.png
./cuosd perf --seed=31 --input=640x640/RGBA --arrow=100 --save=workspace/arrow.png
./cuosd perf --seed=31 --input=640x640/RGBA --point=100 --save=workspace/point.png
./cuosd perf --seed=31 --input=640x640/RGBA --font=workspace/my.nvfont --clock=100 --save=workspace/clock.png
./cuosd perf --seed=31 --input=640x640/RGBA --font=workspace/my.nvfont --text=10 --line=10 --circle=10 --rectangle=10 --arrow=10 --point=10 --clock=10 --save=workspace/mix.png


./cuosd perf --seed=31 --input=1920x1080/PL --font=workspace/my.nvfont --text=100 --save=workspace/text.png
./cuosd perf --seed=31 --input=1920x1080/PL --line=100 --save=workspace/line.png
./cuosd perf --seed=31 --input=1920x1080/PL --circle=100 --save=workspace/circle.png
./cuosd perf --seed=31 --input=1920x1080/PL --rectangle=100 --save=workspace/rectangle.png
./cuosd perf --seed=31 --input=1920x1080/PL --arrow=100 --save=workspace/arrow.png
./cuosd perf --seed=31 --input=1920x1080/PL --point=100 --save=workspace/point.png
./cuosd perf --seed=31 --input=1920x1080/PL --font=workspace/my.nvfont --clock=100 --save=workspace/clock.png
./cuosd perf --seed=31 --input=1920x1080/PL --font=workspace/my.nvfont --text=10 --line=10 --circle=10 --rectangle=10 --arrow=10 --point=10 --clock=10 --save=workspace/mix.png