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
#include <thrust/sort.h>

#include "camera-geometry.hpp"
#include "common/check.hpp"
#include "common/launch.cuh"
#include "common/tensor.hpp"

namespace bevfusion {
namespace camera {

struct GeometryParameterExtra : public GeometryParameter {
  unsigned int D;
  nvtype::Float3 dx;
  nvtype::Float3 bx;
  nvtype::Int3 nx;
};

static __forceinline__ __device__ float dot(const float4& T, const float3& p) { return T.x * p.x + T.y * p.y + T.z * p.z; }

static __forceinline__ __device__ float project(const float4& T, const float3& p) {
  return T.x * p.x + T.y * p.y + T.z * p.z + T.w;
}

static __forceinline__ __device__ float3 inverse_project(const float4* T, const float3& p) {
  float3 r;
  r.x = p.x - T[0].w;
  r.y = p.y - T[1].w;
  r.z = p.z - T[2].w;
  return make_float3(dot(T[0], r), dot(T[1], r), dot(T[2], r));
}

static __global__ void arange_kernel(unsigned int num, int32_t* p) {
  int idx = cuda_linear_index;
  if (idx < num) {
    p[idx] = idx;
  }
}

static __global__ void interval_starts_kernel(unsigned int num, unsigned int remain, unsigned int total, const int32_t* ranks,
                                              const int32_t* indices, int32_t* interval_starts, int32_t* interval_starts_size) {
  int idx = cuda_linear_index;
  if (idx >= num) return;

  unsigned int i = remain + 1 + idx;
  if (ranks[i] != ranks[i - 1]) {
    unsigned int offset = atomicAdd(interval_starts_size, 1);
    interval_starts[offset] = idx + 1;
  }
}

static __global__ void collect_starts_kernel(unsigned int num, unsigned int remain, unsigned int numel_geometry,
                                             const int32_t* indices, const int32_t* interval_starts, const int32_t* geometry,
                                             int3* intervals) {
  int i = cuda_linear_index;
  if (i >= num) return;

  int3 val;
  val.x = interval_starts[i] + remain;

  // https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution/pull/250
  val.y = i < num - 1 ? interval_starts[i + 1] + remain : numel_geometry - 1;
  val.z = geometry[indices[interval_starts[i] + remain]];
  intervals[i] = val;
}

static void __host__ matrix_inverse_4x4(const float* m, float* inv) {
  double det = m[0] * (m[5] * m[10] - m[9] * m[6]) - m[1] * (m[4] * m[10] - m[6] * m[8]) + m[2] * (m[4] * m[9] - m[5] * m[8]);
  double invdet = 1.0 / det;
  inv[0] = (m[5] * m[10] - m[9] * m[6]) * invdet;
  inv[1] = (m[2] * m[9] - m[1] * m[10]) * invdet;
  inv[2] = (m[1] * m[6] - m[2] * m[5]) * invdet;
  inv[3] = m[3];
  inv[4] = (m[6] * m[8] - m[4] * m[10]) * invdet;
  inv[5] = (m[0] * m[10] - m[2] * m[8]) * invdet;
  inv[6] = (m[4] * m[2] - m[0] * m[6]) * invdet;
  inv[7] = m[7];
  inv[8] = (m[4] * m[9] - m[8] * m[5]) * invdet;
  inv[9] = (m[8] * m[1] - m[0] * m[9]) * invdet;
  inv[10] = (m[0] * m[5] - m[4] * m[1]) * invdet;
  inv[11] = m[11];
  inv[12] = m[12];
  inv[13] = m[13];
  inv[14] = m[14];
  inv[15] = m[15];
}

static __global__ void create_frustum_kernel(unsigned int feat_width, unsigned int feat_height, unsigned int D,
                                             unsigned int image_width, unsigned int image_height, float w_interval,
                                             float h_interval, nvtype::Float3 dbound, float3* frustum) {
  int ix = cuda_2d_x;
  int iy = cuda_2d_y;
  int id = blockIdx.z;
  if (ix >= feat_width || iy >= feat_height) return;

  unsigned int offset = (id * feat_height + iy) * feat_width + ix;
  frustum[offset] = make_float3(ix * w_interval, iy * h_interval, dbound.x + id * dbound.z);
}

static __global__ void compute_geometry_kernel(unsigned int numel_frustum, const float3* frustum, const float4* camera2lidar,
                                               const float4* camera_intrins_inv, const float4* img_aug_matrix_inv,
                                               nvtype::Float3 bx, nvtype::Float3 dx, nvtype::Int3 nx, unsigned int* keep_count,
                                               int* ranks, nvtype::Int3 geometry_dim, unsigned int num_camera,
                                               int* geometry_out) {
  int tid = cuda_linear_index;
  if (tid >= numel_frustum) return;

  float3 point = frustum[tid];
  // float3 point     = make_float3(point_half.x, point_half.y, point_half.z);
  for (int icamerea = 0; icamerea < num_camera; ++icamerea) {
    float3 projed = inverse_project(img_aug_matrix_inv, point);
    projed.x *= projed.z;
    projed.y *= projed.z;
    projed = make_float3(dot(camera_intrins_inv[4 * icamerea + 0], projed), dot(camera_intrins_inv[4 * icamerea + 1], projed),
                         dot(camera_intrins_inv[4 * icamerea + 2], projed));
    projed = make_float3(project(camera2lidar[4 * icamerea + 0], projed), project(camera2lidar[4 * icamerea + 1], projed),
                         project(camera2lidar[4 * icamerea + 2], projed));

    int _pid = icamerea * numel_frustum + tid;
    int3 coords;
    coords.x = int((projed.x - (bx.x - dx.x / 2.0)) / dx.x);
    coords.y = int((projed.y - (bx.y - dx.y / 2.0)) / dx.y);
    coords.z = int((projed.z - (bx.z - dx.z / 2.0)) / dx.z);
    geometry_out[_pid] = (coords.z * geometry_dim.z * geometry_dim.y + coords.x) * geometry_dim.x + coords.y;

    bool kept = coords.x >= 0 && coords.y >= 0 && coords.z >= 0 && coords.x < nx.x && coords.y < nx.y && coords.z < nx.z;
    if (!kept) {
      ranks[_pid] = 0;
    } else {
      atomicAdd(keep_count, 1);
      ranks[_pid] = (coords.x * nx.y + coords.y) * nx.z + coords.z;
    }
  }
}

class GeometryImplement : public Geometry {
 public:
  virtual ~GeometryImplement() {
    if (counter_host_) checkRuntime(cudaFreeHost(counter_host_));
    if (keep_count_) checkRuntime(cudaFree(keep_count_));
    if (frustum_) checkRuntime(cudaFree(frustum_));
    if (geometry_) checkRuntime(cudaFree(geometry_));
    if (ranks_) checkRuntime(cudaFree(ranks_));
    if (indices_) checkRuntime(cudaFree(indices_));
    if (interval_starts_) checkRuntime(cudaFree(interval_starts_));
    if (interval_starts_size_) checkRuntime(cudaFree(interval_starts_size_));
    if (intervals_) checkRuntime(cudaFree(intervals_));
    if (camera2lidar_) checkRuntime(cudaFree(camera2lidar_));
    if (camera_intrinsics_inverse_) checkRuntime(cudaFree(camera_intrinsics_inverse_));
    if (img_aug_matrix_inverse_) checkRuntime(cudaFree(img_aug_matrix_inverse_));
    if (camera_intrinsics_inverse_host_) checkRuntime(cudaFreeHost(camera_intrinsics_inverse_host_));
    if (img_aug_matrix_inverse_host_) checkRuntime(cudaFreeHost(img_aug_matrix_inverse_host_));
  }

  bool init(GeometryParameter param) {
    static_cast<GeometryParameter&>(param_) = param;
    param_.D = (unsigned int)std::round((param_.dbound.y - param_.dbound.x) / param_.dbound.z);
    param_.bx = nvtype::Float3(param_.xbound.x + param_.xbound.z / 2.0f, param_.ybound.x + param_.ybound.z / 2.0f,
                               param_.zbound.x + param_.zbound.z / 2.0f);

    param_.dx = nvtype::Float3(param_.xbound.z, param_.ybound.z, param_.zbound.z);
    param_.nx = nvtype::Int3(static_cast<int>(std::round((param_.xbound.y - param_.xbound.x) / param_.xbound.z)),
                             static_cast<int>(std::round((param_.ybound.y - param_.ybound.x) / param_.ybound.z)),
                             static_cast<int>(std::round((param_.zbound.y - param_.zbound.x) / param_.zbound.z)));

    cudaStream_t stream = nullptr;
    float w_interval = (param_.image_width - 1.0f) / (param_.feat_width - 1.0f);
    float h_interval = (param_.image_height - 1.0f) / (param_.feat_height - 1.0f);
    numel_frustum_ = param_.feat_width * param_.feat_height * param_.D;
    numel_geometry_ = numel_frustum_ * param_.num_camera;

    checkRuntime(cudaMallocHost(&counter_host_, sizeof(int32_t)));
    checkRuntime(cudaMalloc(&keep_count_, sizeof(int32_t)));
    checkRuntime(cudaMalloc(&frustum_, numel_frustum_ * sizeof(float3)));
    checkRuntime(cudaMalloc(&geometry_, numel_geometry_ * sizeof(int32_t)));
    checkRuntime(cudaMalloc(&ranks_, numel_geometry_ * sizeof(int32_t)));
    checkRuntime(cudaMalloc(&indices_, numel_geometry_ * sizeof(int32_t)));
    checkRuntime(cudaMalloc(&interval_starts_, numel_geometry_ * sizeof(int32_t)));
    checkRuntime(cudaMalloc(&interval_starts_size_, sizeof(int32_t)));
    checkRuntime(cudaMalloc(&intervals_, numel_geometry_ * sizeof(int3)));

    bytes_of_matrix_ = param_.num_camera * 4 * 4 * sizeof(float);
    checkRuntime(cudaMalloc(&camera2lidar_, bytes_of_matrix_));
    checkRuntime(cudaMalloc(&camera_intrinsics_inverse_, bytes_of_matrix_));
    checkRuntime(cudaMalloc(&img_aug_matrix_inverse_, bytes_of_matrix_));
    checkRuntime(cudaMallocHost(&camera_intrinsics_inverse_host_, bytes_of_matrix_));
    checkRuntime(cudaMallocHost(&img_aug_matrix_inverse_host_, bytes_of_matrix_));
    cuda_2d_launch(create_frustum_kernel, stream, param_.feat_width, param_.feat_height, param_.D, param_.image_width,
                   param_.image_height, w_interval, h_interval, param_.dbound, frustum_);
    return true;
  }

  // You can call this function if you need to update the matrix
  // All matrix pointers must be on the host
  virtual void update(const float* camera2lidar, const float* camera_intrinsics, const float* img_aug_matrix,
                      void* stream = nullptr) override {
    Asserts(frustum_ != nullptr,
            "If the excess memory has been freed, then the update call will not be logical for the "
            "program.");

    cudaStream_t _stream = static_cast<cudaStream_t>(stream);
    for (unsigned int icamera = 0; icamera < param_.num_camera; ++icamera) {
      unsigned int offset = icamera * 4 * 4;
      matrix_inverse_4x4(camera_intrinsics + offset, camera_intrinsics_inverse_host_ + offset);
      matrix_inverse_4x4(img_aug_matrix + offset, img_aug_matrix_inverse_host_ + offset);
    }

    // For users, please ensure that the pointer lifecycle is available for asynchronous copying.
    checkRuntime(cudaMemcpyAsync(camera2lidar_, camera2lidar, bytes_of_matrix_, cudaMemcpyHostToDevice, _stream));
    checkRuntime(cudaMemcpyAsync(camera_intrinsics_inverse_, camera_intrinsics_inverse_host_, bytes_of_matrix_,
                                 cudaMemcpyHostToDevice, _stream));
    checkRuntime(cudaMemcpyAsync(img_aug_matrix_inverse_, img_aug_matrix_inverse_host_, bytes_of_matrix_, cudaMemcpyHostToDevice,
                                 _stream));
    checkRuntime(cudaMemsetAsync(keep_count_, 0, sizeof(unsigned int), _stream));

    cuda_linear_launch(compute_geometry_kernel, _stream, numel_frustum_, frustum_, reinterpret_cast<const float4*>(camera2lidar_),
                       reinterpret_cast<const float4*>(camera_intrinsics_inverse_),
                       reinterpret_cast<const float4*>(img_aug_matrix_inverse_), param_.bx, param_.dx, param_.nx, keep_count_,
                       ranks_, param_.geometry_dim, param_.num_camera, geometry_);
    checkRuntime(cudaMemcpyAsync(counter_host_, keep_count_, sizeof(unsigned int), cudaMemcpyDeviceToHost, _stream));
    cuda_linear_launch(arange_kernel, _stream, numel_geometry_, indices_);
    thrust::stable_sort_by_key(thrust::cuda::par.on(_stream), ranks_, ranks_ + numel_geometry_, indices_, thrust::less<int>());
    checkRuntime(cudaStreamSynchronize(_stream));

    unsigned int remain_ranks = numel_geometry_ - *counter_host_;
    unsigned int threads = *counter_host_ - 1;
    checkRuntime(cudaMemsetAsync(interval_starts_size_, 0, sizeof(int32_t), _stream));

    // set interval_starts_[0] to 0
    checkRuntime(cudaMemsetAsync(interval_starts_, 0, sizeof(int32_t), _stream));
    cuda_linear_launch(interval_starts_kernel, _stream, threads, remain_ranks, numel_geometry_, ranks_, indices_,
                       interval_starts_ + 1, interval_starts_size_);
    checkRuntime(cudaMemcpyAsync(counter_host_, interval_starts_size_, sizeof(unsigned int), cudaMemcpyDeviceToHost, _stream));
    checkRuntime(cudaStreamSynchronize(_stream));

    // interval_starts_[0] = 0,  and counter += 1
    n_intervals_ = *counter_host_ + 1;

    thrust::stable_sort(thrust::cuda::par.on(_stream), interval_starts_, interval_starts_ + n_intervals_, thrust::less<int>());
    cuda_linear_launch(collect_starts_kernel, _stream, n_intervals_, remain_ranks, numel_geometry_, indices_, interval_starts_,
                       geometry_, intervals_);
  }

  virtual void free_excess_memory() override {
    if (counter_host_) {
      checkRuntime(cudaFreeHost(counter_host_));
      counter_host_ = nullptr;
    }
    if (keep_count_) {
      checkRuntime(cudaFree(keep_count_));
      keep_count_ = nullptr;
    }
    if (frustum_) {
      checkRuntime(cudaFree(frustum_));
      frustum_ = nullptr;
    }
    if (geometry_) {
      checkRuntime(cudaFree(geometry_));
      geometry_ = nullptr;
    }
    if (ranks_) {
      checkRuntime(cudaFree(ranks_));
      ranks_ = nullptr;
    }
    if (interval_starts_) {
      checkRuntime(cudaFree(interval_starts_));
      interval_starts_ = nullptr;
    }
    if (interval_starts_size_) {
      checkRuntime(cudaFree(interval_starts_size_));
      interval_starts_size_ = nullptr;
    }
    if (camera2lidar_) {
      checkRuntime(cudaFree(camera2lidar_));
      camera2lidar_ = nullptr;
    }
    if (camera_intrinsics_inverse_) {
      checkRuntime(cudaFree(camera_intrinsics_inverse_));
      camera_intrinsics_inverse_ = nullptr;
    }
    if (img_aug_matrix_inverse_) {
      checkRuntime(cudaFree(img_aug_matrix_inverse_));
      img_aug_matrix_inverse_ = nullptr;
    }
    if (camera_intrinsics_inverse_host_) {
      checkRuntime(cudaFreeHost(camera_intrinsics_inverse_host_));
      camera_intrinsics_inverse_host_ = nullptr;
    }
    if (img_aug_matrix_inverse_host_) {
      checkRuntime(cudaFreeHost(img_aug_matrix_inverse_host_));
      img_aug_matrix_inverse_host_ = nullptr;
    }
  }

  virtual unsigned int num_intervals() override { return n_intervals_; }

  virtual unsigned int num_indices() override { return numel_geometry_; }

  virtual nvtype::Int3* intervals() override { return reinterpret_cast<nvtype::Int3*>(intervals_); }

  virtual unsigned int* indices() override { return reinterpret_cast<unsigned int*>(indices_); }

 private:
  size_t bytes_of_matrix_ = 0;
  float* camera2lidar_ = nullptr;
  float* camera_intrinsics_inverse_ = nullptr;
  float* img_aug_matrix_inverse_ = nullptr;
  float* camera_intrinsics_inverse_host_ = nullptr;
  float* img_aug_matrix_inverse_host_ = nullptr;

  float3* frustum_ = nullptr;
  unsigned int numel_frustum_ = 0;

  unsigned int n_intervals_ = 0;
  unsigned int numel_geometry_ = 0;
  int32_t* geometry_ = nullptr;
  int32_t* ranks_ = nullptr;
  int32_t* indices_ = nullptr;
  int3* intervals_ = nullptr;
  int32_t* interval_starts_ = nullptr;
  int32_t* interval_starts_size_ = nullptr;
  unsigned int* keep_count_ = nullptr;
  unsigned int* counter_host_ = nullptr;
  GeometryParameterExtra param_;
};

std::shared_ptr<Geometry> create_geometry(GeometryParameter param) {
  std::shared_ptr<GeometryImplement> instance(new GeometryImplement());
  if (!instance->init(param)) {
    instance.reset();
  }
  return instance;
}

};  // namespace camera
};  // namespace bevfusion