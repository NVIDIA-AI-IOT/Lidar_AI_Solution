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

import tensor
import numpy as np
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Compare result between CPP and PyTorch")
    parser.add_argument("cpp_tensor", help="cpp tensor file or dense tensor[ /data/cpp.features.tensor:cpp.indices.tensor ]")
    parser.add_argument("pytorch_tensor", help="pytorch tensor file or dense tensor[ /data/torch.features.tensor:torch.indices.tensor ]")
    parser.add_argument("--detail", action="store_true", help="Print detail information")
    parser.add_argument("--sort", action="store_true", help="Sort value")
    parser.add_argument("--to-dense", action="store_true", help="To dense")
    parser.add_argument("--dense-grid", type=str, help="Parse cpp_tensor/pytorch_tensor as dense, like: 41x1440x1440")
    args = parser.parse_args()
    return args

def scatter_nd(indices, updates, shape):
    ret = np.zeros(shape, dtype=updates.dtype)
    ndim = indices.shape[-1]
    output_shape = list(indices.shape[:-1]) + shape[indices.shape[-1]:]
    flatted_indices = indices.reshape(-1, ndim)
    slices = [flatted_indices[:, i] for i in range(ndim)]
    slices += [Ellipsis]
    ret[slices] = updates.reshape(*output_shape)
    return ret

def tensor_to_dense(features, indices, spatial_shape, channels_first: bool = True):
    output_shape = [1] + list(
        spatial_shape) + [features.shape[1]]
    res = scatter_nd(
        indices.astype(np.int32), features,
        output_shape)
    if not channels_first:
        return res
    ndim = len(spatial_shape)
    trans_params = list(range(0, ndim + 1))
    trans_params.insert(1, ndim + 1)
    return res.transpose(*trans_params)

def split_dense_path(path : str):
    # /data/cpp.features.tensor:cpp.indices.tensor
    p = path.rfind("/")
    p = max(p + 1, 0)
    m = path[p:].split(":")
    if len(m) != 2:
        raise RuntimeError("Parse dense path failed.")
    
    directory = path[:p]
    return os.path.join(directory, m[0]), os.path.join(directory, m[1])

def load_tensor(file, dense_grid, detail=False, to_dense=False):
    
    if dense_grid is None:
        return tensor.load(file)

    fpath, ipath = split_dense_path(file)
    features = tensor.load(fpath)
    indices  = tensor.load(ipath)
    if detail:
        print("================ Dense Tensor Information =================")
        print(f"Load dense tensor[grid = {dense_grid}]: {file}")
        print(f"  Features[{features.shape}, {features.dtype}]:{fpath}")
        print(f"  Indices[{indices.shape}, {indices.dtype}]:{ipath}")
        print("======================================================")

    if to_dense:
        return tensor_to_dense(features, indices, dense_grid)
    
    gz, gy, gx = dense_grid
    zbit = int(np.ceil(np.log2(gz)))
    ybit = int(np.ceil(np.log2(gy)))
    xbit = int(np.ceil(np.log2(gx)))
    ib, iz, iy, ix = indices.astype(np.uint64).T
    idx = (ib << (zbit + ybit + xbit)) | (iz << (ybit + xbit)) | (iy << xbit) | ix
    relocation = np.argsort(idx)
    return np.ascontiguousarray(features[relocation])

def compare_and_print(cpp_file, torch_file, detail=False, dense_grid=None, to_dense=False, do_sort=False):

    cpp_tensor   = load_tensor(cpp_file, dense_grid, detail, to_dense)
    torch_tensor = load_tensor(torch_file, dense_grid, detail, to_dense)

    cpp_shape    = ' x '.join(map(str, cpp_tensor.shape))
    torch_shape  = ' x '.join(map(str, torch_tensor.shape))

    if detail:
        print("================ Compare Information =================")
        print(f" CPP     Tensor: {cpp_shape}, {cpp_tensor.dtype} : {cpp_file}")
        print(f" PyTorch Tensor: {torch_shape}, {torch_tensor.dtype} : {torch_file}")

    if np.cumprod(cpp_tensor.shape)[-1] != np.cumprod(torch_tensor.shape)[-1]:
        raise RuntimeError(f"Invalid compare with mismatched shape, {cpp_shape} < ----- > {torch_shape}")

    cpp_tensor   = cpp_tensor.reshape(-1).astype(np.float32)
    torch_tensor = torch_tensor.reshape(-1).astype(np.float32)

    if do_sort:
        cpp_tensor   = np.sort(cpp_tensor)
        torch_tensor = np.sort(torch_tensor)

    diff        = np.abs(cpp_tensor - torch_tensor)
    absdiff_max = diff.max().item()
    print(f"\033[31m[absdiff]: max:{absdiff_max}, sum:{diff.sum().item():.6f}, std:{diff.std().item():.6f}, mean:{diff.mean().item():.6f}\033[0m")
    if not detail:
        return

    print(f"CPP:   absmax:{np.abs(cpp_tensor).max().item():.6f}, min:{cpp_tensor.min().item():.6f}, std:{cpp_tensor.std().item():.6f}, mean:{cpp_tensor.mean().item():.6f}")
    print(f"Torch: absmax:{np.abs(torch_tensor).max().item():.6f}, min:{torch_tensor.min().item():.6f}, std:{torch_tensor.std().item():.6f}, mean:{torch_tensor.mean().item():.6f}")
    
    absdiff_p75 = absdiff_max * 0.75
    absdiff_p50 = absdiff_max * 0.50
    absdiff_p25 = absdiff_max * 0.25
    numel       = cpp_tensor.shape[0]
    num_p75     = np.sum(diff > absdiff_p75)
    num_p50     = np.sum(diff > absdiff_p50)
    num_p25     = np.sum(diff > absdiff_p25)
    num_p00     = np.sum(diff > 0)
    num_eq00    = np.sum(diff == 0)
    print(f"[absdiff > m75% --- {absdiff_p75:.6f}]: {num_p75 / numel * 100:.3f} %, {num_p75}")
    print(f"[absdiff > m50% --- {absdiff_p50:.6f}]: {num_p50 / numel * 100:.3f} %, {num_p50}")
    print(f"[absdiff > m25% --- {absdiff_p25:.6f}]: {num_p25 / numel * 100:.3f} %, {num_p25}")
    print(f"[absdiff > 0]: {num_p00 / numel * 100:.3f} %, {num_p00}")
    print(f"[absdiff = 0]: {num_eq00 / numel * 100:.3f} %, {num_eq00}")

    cpp_norm   = np.linalg.norm(cpp_tensor)
    torch_norm = np.linalg.norm(torch_tensor)
    sim        = (np.matmul(cpp_tensor, torch_tensor) / (cpp_norm * torch_norm))
    print(f"[cosine]: {sim * 100:.3f} %")
    print("======================================================")
    
    # np.testing.assert_almost_equal(cpp_tensor, torch_tensor, decimal=3)

if __name__ == "__main__":
    
    args = parse_args()
    grids = None
    if args.dense_grid is not None:
        grids = args.dense_grid.split("x")
        assert len(grids) == 3, "Invalid grid argument."
        grids = list(map(int, grids))

    compare_and_print(args.cpp_tensor, args.pytorch_tensor, args.detail, grids, args.to_dense, args.sort)