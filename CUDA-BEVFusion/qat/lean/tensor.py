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

def save(tensor, file):
    
    if not isinstance(tensor, np.ndarray):
        import torch
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.detach().cpu().data.numpy()
        else:
            tensor = np.array(tensor)

    dtype_map = {"float32" : 3, "float16" : 2, "int32" : 1, "int64" : 4, "uint64": 5, "uint32": 6, "int8": 7, "uint8": 8}
    if str(tensor.dtype) not in dtype_map:
        raise RuntimeError(f"Unsupport dtype {tensor.dtype}")

    magic_number = 0x33ff1101
    with open(file, "wb") as f:
        head = np.array([magic_number, tensor.ndim, dtype_map[str(tensor.dtype)]], dtype=np.int32).tobytes()
        f.write(head)

        dims = np.array(tensor.shape, dtype=np.int32).tobytes()
        f.write(dims)
        
        data = tensor.tobytes()
        f.write(data)


def load(file, return_torch=False):

    dtype_for_integer_mapping = {3: np.float32, 2: np.float16, 1: np.int32, 4: np.int64, 5: np.uint64, 6: np.uint32, 7: np.int8, 8: np.uint8}
    dtype_size_mapping        = {np.float32 : 4, np.float16 : 2, np.int32 : 4, np.int64 : 8, np.uint64 : 8, np.uint32 : 4, np.int8 : 1, np.uint8 : 1}

    with open(file, "rb") as f:
        magic_number, ndim, dtype_integer = np.frombuffer(f.read(12), dtype=np.int32)
        if dtype_integer not in dtype_for_integer_mapping:
            raise RuntimeError(f"Can not find match dtype for index {dtype_integer}")

        dtype            = dtype_for_integer_mapping[dtype_integer]
        magic_number_std = 0x33ff1101
        assert magic_number == magic_number_std, f"this file is not tensor file"
        dims   = np.frombuffer(f.read(ndim * 4), dtype=np.int32)
        volumn = np.cumprod(dims)[-1]
        data   = np.frombuffer(f.read(volumn * dtype_size_mapping[dtype]), dtype=dtype).reshape(*dims)

        if return_torch:
            import torch
            return torch.from_numpy(data)
        return data
