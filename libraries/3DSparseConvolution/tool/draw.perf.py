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

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

perfint8 = pd.read_csv("workspace/perf-int8.log")
perffp16 = pd.read_csv("workspace/perf-float16.log")
# perffp16 = pd.read_csv("workspace/perf-float16-bevfusion-xyz.log")
# perffp16 = pd.read_csv("workspace/perf-float16-bevfusion-zyx.log")
# perffp16 = pd.read_csv("workspace/perf-float16-centerpoint-zyx.log")

plt.figure(figsize=(21, 8))
plt.suptitle("Summary of SCNModel Inference on Orin", fontsize=30)
column = "scn_time"
plt.subplot(1, 3, 1)
plt.title("SCN Model")
plt.xlabel("Inference time(unit: ms)")
plt.ylabel("Count")
plt.grid(alpha=0.5)
scn_time_fp16 = np.array(perffp16[column])
plt.hist(scn_time_fp16, 50, alpha=0.5, label="fp16")
fp16_min, fp16_max, fp16_mean = scn_time_fp16.min(), scn_time_fp16.max(), scn_time_fp16.mean()

scn_time_int8 = np.array(perfint8[column])
int8_min, int8_max, int8_mean = scn_time_int8.min(), scn_time_int8.max(), scn_time_int8.mean()

plt.hist(scn_time_int8, 50, alpha=0.5, label="int8")
plt.legend([f"fp16 min={fp16_min:.1f},max={fp16_max:.1f},mean={fp16_mean:.1f}", f"int8 min={int8_min:.1f},max={int8_max:.1f},mean={int8_mean:.1f}"], loc='upper right')

column = "voxelization_time"
plt.subplot(1, 3, 2)
plt.title("Voxelization")
plt.xlabel("Inference time(unit: ms)")
plt.ylabel("Count")
plt.grid(alpha=0.5)
scn_time_fp16 = np.array(perffp16[column])
plt.hist(scn_time_fp16, 50, alpha=0.5, label="fp16")
fp16_min, fp16_max, fp16_mean = scn_time_fp16.min(), scn_time_fp16.max(), scn_time_fp16.mean()
plt.legend([f"min={fp16_min:.2f},max={fp16_max:.2f},mean={fp16_mean:.2f}"], loc='upper right')


column = "num_valid"
plt.subplot(1, 3, 3)
plt.title("Number of Valid Point")
plt.xlabel("Number of Valid Point")
plt.ylabel("Count")
plt.grid(alpha=0.5)
scn_time_fp16 = np.array(perffp16[column])
plt.hist(scn_time_fp16, 50, alpha=0.5, label="fp16")
fp16_min, fp16_max, fp16_mean = scn_time_fp16.min(), scn_time_fp16.max(), scn_time_fp16.mean()

plt.legend([f"min={fp16_min},max={fp16_max},mean={int(fp16_mean)}"], loc='upper right')
plt.savefig("workspace/perf.png")
# print(perfint8.loc["scn_time"].hist(bins=50))