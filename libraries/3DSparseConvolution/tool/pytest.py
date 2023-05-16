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

import pyscn
import numpy as np
import tensor

workdir  = "workspace/centerpoint"
inference_type = "fp16"  # fp16 or int8
features = tensor.load(f"{workdir}/in_features.torch.fp16.tensor")
indices  = tensor.load(f"{workdir}/in_indices_zyx.torch.int32.tensor")

pyscn.set_verbose(True)
model = pyscn.SCNModel(f"{workdir}/centerpoint.scn.PTQ.onnx", inference_type)
features, indices = model.forward(features, indices, [41, 1440, 1440], 0)

tensor.save(features, f"{workdir}/out_dense.py.fp16.tensor")
print(f"[PASSED 🤗].\nTo verify result:\n  python tool/compare.py {workdir}/out_dense.py.fp16.tensor {workdir}/out_dense.torch.fp16.tensor --detail")
del model