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

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cub/device/device_scan.cuh>
#include <limits>
#include <stdexcept>
#include <string>

#include "lidar-preprocess.hpp"

namespace bevfusion {
namespace lidar {
namespace {
void check(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(std::string("Lidar preprocessing: ") + cudaGetErrorString(error));
}

__device__ float read_float(const unsigned char* point, uint32_t offset, bool bigendian) {
  const unsigned char* p = point + offset;
  const uint32_t bits = bigendian ? (uint32_t(p[0]) << 24 | uint32_t(p[1]) << 16 | uint32_t(p[2]) << 8 | p[3])
                                  : (uint32_t(p[3]) << 24 | uint32_t(p[2]) << 16 | uint32_t(p[1]) << 8 | p[0]);
  return __uint_as_float(bits);
}

__device__ const unsigned char* point_at(const unsigned char* data, int index, PointCloudLayout layout) {
  return data + size_t(index / layout.width) * layout.row_step + size_t(index % layout.width) * layout.point_step;
}

__global__ void flag_points(const unsigned char* data, PointCloudLayout layout, int count, int* flags) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  const auto* point = point_at(data, index, layout);
  bool valid = true;
  for (int j = 0; j < 4; ++j) valid &= isfinite(read_float(point, layout.offsets[j], layout.is_bigendian));
  flags[index] = valid ? 1 : 0;
}

__global__ void pack_points(const unsigned char* data, PointCloudLayout layout, int count, const int* flags,
                            const int* offsets, int capacity, __half* output) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count || !flags[index] || offsets[index] >= capacity) return;
  const auto* point = point_at(data, index, layout);
  __half* dst = output + size_t(offsets[index]) * 5;
  for (int j = 0; j < 4; ++j) dst[j] = __float2half_rn(read_float(point, layout.offsets[j], layout.is_bigendian));
  dst[4] = __float2half_rn(0.0f);
}

__global__ void count_points(const int* flags, const int* offsets, int count, int capacity, int* output) {
  *output = min(capacity, offsets[count - 1] + flags[count - 1]);
}

class PreprocessorImpl : public Preprocessor {
 public:
  ~PreprocessorImpl() override {
    cudaFree(flags_);
    cudaFree(offsets_);
    cudaFree(output_);
    cudaFree(count_);
    cudaFree(workspace_);
    if (host_count_) cudaFreeHost(host_count_);
  }

  void init(int input, int output) {
    if (input <= 0 || output <= 0) throw std::invalid_argument("Lidar capacities must be positive");
    cudaFuncAttributes attributes{};
    check(cudaFuncGetAttributes(&attributes, flag_points));
    input_capacity_ = input;
    output_capacity_ = output;
    check(cudaMalloc(reinterpret_cast<void**>(&flags_), size_t(input) * sizeof(int)));
    check(cudaMalloc(reinterpret_cast<void**>(&offsets_), size_t(input) * sizeof(int)));
    check(cudaMalloc(reinterpret_cast<void**>(&output_), size_t(output) * 5 * sizeof(__half)));
    check(cudaMalloc(reinterpret_cast<void**>(&count_), sizeof(int)));
    check(cudaMallocHost(reinterpret_cast<void**>(&host_count_), sizeof(int)));
    check(cub::DeviceScan::ExclusiveSum(nullptr, workspace_bytes_, flags_, offsets_, input));
    check(cudaMalloc(&workspace_, workspace_bytes_));
  }

  PreparedLidar forward(const void* data, const PointCloudLayout& layout, void* stream) override {
    const uint64_t count64 = uint64_t(layout.width) * layout.height;
    if (count64 > uint64_t(input_capacity_)) throw std::invalid_argument("Lidar input exceeds configured capacity");
    if (uint64_t(layout.width) * layout.point_step > layout.row_step) {
      throw std::invalid_argument("Lidar row stride is too small");
    }
    for (auto offset : layout.offsets) {
      if (uint64_t(offset) + sizeof(float) > layout.point_step && count64) {
        throw std::invalid_argument("Lidar field exceeds point stride");
      }
    }
    if (!count64) return {};
    if (!data) throw std::invalid_argument("Null lidar device buffer");
    const int count = static_cast<int>(count64);
    const auto cuda_stream = static_cast<cudaStream_t>(stream);
    const auto* bytes = static_cast<const unsigned char*>(data);
    const int blocks = (count - 1) / 256 + 1;
    flag_points<<<blocks, 256, 0, cuda_stream>>>(bytes, layout, count, flags_);
    check(cudaGetLastError());
    check(cub::DeviceScan::ExclusiveSum(workspace_, workspace_bytes_, flags_, offsets_, count, cuda_stream));
    pack_points<<<blocks, 256, 0, cuda_stream>>>(bytes, layout, count, flags_, offsets_, output_capacity_, output_);
    check(cudaGetLastError());
    count_points<<<1, 1, 0, cuda_stream>>>(flags_, offsets_, count, output_capacity_, count_);
    check(cudaGetLastError());
    check(cudaMemcpyAsync(host_count_, count_, sizeof(int), cudaMemcpyDeviceToHost, cuda_stream));
    check(cudaStreamSynchronize(cuda_stream));
    return {reinterpret_cast<const nvtype::half*>(output_), *host_count_};
  }

 private:
  int input_capacity_ = 0;
  int output_capacity_ = 0;
  int* flags_ = nullptr;
  int* offsets_ = nullptr;
  __half* output_ = nullptr;
  int* count_ = nullptr;
  int* host_count_ = nullptr;
  void* workspace_ = nullptr;
  size_t workspace_bytes_ = 0;
};
}  // namespace

std::shared_ptr<Preprocessor> create_preprocessor(int max_input_points, int max_output_points) {
  auto instance = std::make_shared<PreprocessorImpl>();
  instance->init(max_input_points, max_output_points);
  return instance;
}
}  // namespace lidar
}  // namespace bevfusion
