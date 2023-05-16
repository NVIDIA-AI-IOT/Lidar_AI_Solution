/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef __LAUNCH_CUH__
#define __LAUNCH_CUH__

#include "check.hpp"

namespace nv {

#define LINEAR_LAUNCH_THREADS 512
#define cuda_linear_index (blockDim.x * blockIdx.x + threadIdx.x)
#define cuda_2d_x (blockDim.x * blockIdx.x + threadIdx.x)
#define cuda_2d_y (blockDim.y * blockIdx.y + threadIdx.y)
#define divup(a, b) ((static_cast<int>(a) + static_cast<int>(b) - 1) / static_cast<int>(b))

#ifdef CUDA_DEBUG
#define cuda_linear_launch(kernel, stream, num, ...)                                   \
  do {                                                                                 \
    size_t __num__ = (size_t)(num);                                                    \
    size_t __blocks__ = (__num__ + LINEAR_LAUNCH_THREADS - 1) / LINEAR_LAUNCH_THREADS; \
    kernel<<<__blocks__, LINEAR_LAUNCH_THREADS, 0, stream>>>(__num__, __VA_ARGS__);    \
    nv::check_runtime(cudaPeekAtLastError(), #kernel, __LINE__, __FILE__);             \
    nv::check_runtime(cudaStreamSynchronize(stream), #kernel, __LINE__, __FILE__);     \
  } while (false)

#define cuda_2d_launch(kernel, stream, nx, ny, ...)                                \
  do {                                                                             \
    dim3 __threads__(32, 32);                                                      \
    dim3 __blocks__(divup(nx, 32), divup(ny, 32));                                 \
    kernel<<<__blocks__, __threads__, 0, stream>>>(nx, ny, __VA_ARGS__);           \
    nv::check_runtime(cudaPeekAtLastError(), #kernel, __LINE__, __FILE__);         \
    nv::check_runtime(cudaStreamSynchronize(stream), #kernel, __LINE__, __FILE__); \
  } while (false)
#else  // CUDA_DEBUG
#define cuda_linear_launch(kernel, stream, num, ...)                                \
  do {                                                                              \
    size_t __num__ = (size_t)(num);                                                 \
    size_t __blocks__ = divup(__num__, LINEAR_LAUNCH_THREADS);                      \
    kernel<<<__blocks__, LINEAR_LAUNCH_THREADS, 0, stream>>>(__num__, __VA_ARGS__); \
    nv::check_runtime(cudaPeekAtLastError(), #kernel, __LINE__, __FILE__);          \
  } while (false)

#define cuda_2d_launch(kernel, stream, nx, ny, nz, ...)                      \
  do {                                                                       \
    dim3 __threads__(32, 32);                                                \
    dim3 __blocks__(divup(nx, 32), divup(ny, 32), nz);                       \
    kernel<<<__blocks__, __threads__, 0, stream>>>(nx, ny, nz, __VA_ARGS__); \
    nv::check_runtime(cudaPeekAtLastError(), #kernel, __LINE__, __FILE__);   \
  } while (false)
#endif  // CUDA_DEBUG
};      // namespace nv

#endif  // __LAUNCH_CUH__