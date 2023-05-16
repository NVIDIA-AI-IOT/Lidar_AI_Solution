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

pycode="
import ctypes

dll = ctypes.cdll.LoadLibrary('libcuda.so')
major = ctypes.c_int(0)
minor = ctypes.c_int(0)
dll.cuInit(ctypes.c_int(0))
device = 0
ret = dll.cuDeviceComputeCapability(ctypes.pointer(major), ctypes.pointer(minor), ctypes.c_int(device))
ret = int(ret)
if ret != 0:
    exit(ret)

name = ctypes.create_string_buffer(100)
ret = dll.cuDeviceGetName(ctypes.pointer(name), 100, device)

name = str(name.value, encoding='utf-8')
major = major.value
minor = minor.value

sm = f'{major}{minor}'
print(sm, end='')
"

cudasm=`python3 -c "$pycode"`