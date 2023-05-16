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

trt_version=8503

if [ ! -f "model/rpn_centerhead_sim.plan.${trt_version}" ]; then
    echo Building the model: model/rpn_centerhead_sim.plan.${trt_version}, this will take 2 minutes. Wait a moment 🤗🤗🤗~.
    trtexec --onnx=model/rpn_centerhead_sim.onnx \
        --saveEngine=model/rpn_centerhead_sim.plan.${trt_version} \
        --workspace=4096 --fp16 --outputIOFormats=fp16:chw \
        --inputIOFormats=fp16:chw --verbose --dumpLayerInfo \
        --dumpProfile --separateProfileRun \
        --profilingVerbosity=detailed > model/rpn_centerhead_sim.${trt_version}.log 2>&1

    rm -rf model/rpn_centerhead_sim.plan
    dir=`pwd`
    ln -s ${dir}/model/rpn_centerhead_sim.plan.${trt_version} model/rpn_centerhead_sim.plan
else
    echo Model model/rpn_centerhead_sim.plan.${trt_version} already build 🙋🙋🙋.
fi