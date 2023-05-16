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
 
#include <iostream>

#include "voxelization.cuh"

#define checkCudaErrors(op)                                                                  \
  {                                                                                          \
    auto status = ((op));                                                                    \
    if (status != 0) {                                                                       \
      std::cout << "Cuda failure: " << cudaGetErrorString(status) << " in file " << __FILE__ \
                << ":" << __LINE__ << " error status: " << status << std::endl;              \
      abort();                                                                               \
    }                                                                                        \
  }

const unsigned int MAX_POINTS_NUM = 300000;
const int THREADS_FOR_VOXEL = 256;

__device__ inline uint64_t hash(uint64_t k) {
  k ^= k >> 16;
  k *= 0x85ebca6b;
  k ^= k >> 13;
  k *= 0xc2b2ae35;
  k ^= k >> 16;
  return k;
}

__device__ inline void insertHashTable(const uint32_t key, uint32_t *value,
                                       const uint32_t hash_size, uint32_t *hash_table) {
  uint64_t hash_value = hash(key);
  uint32_t slot = hash_value % (hash_size / 2) /*key, value*/;
  uint32_t empty_key = UINT32_MAX;
  while (true) {
    uint32_t pre_key = atomicCAS(hash_table + slot, empty_key, key);
    if (pre_key == empty_key) {
      hash_table[slot + hash_size / 2 /*offset*/] = atomicAdd(value, 1);
      break;
    } else if (pre_key == key) {
      break;
    }
    slot = (slot + 1) % (hash_size / 2);
  }
}

__device__ inline uint32_t lookupHashTable(const uint32_t key, const uint32_t hash_size,
                                           const uint32_t *hash_table) {
  uint64_t hash_value = hash(key);
  uint32_t slot = hash_value % (hash_size / 2) /*key, value*/;
  uint32_t empty_key = UINT32_MAX;
  int cnt = 0;
  while (true /* need to be adjusted according to data*/) {
    cnt++;
    if (hash_table[slot] == key) {
      return hash_table[slot + hash_size / 2];
    } else if (hash_table[slot] == empty_key) {
      return empty_key;
    } else {
      slot = (slot + 1) % (hash_size / 2);
    }
  }
  return empty_key;
}

__global__ void buildHashKernel(const half *points, size_t points_size, float min_x_range,
                                float max_x_range, float min_y_range, float max_y_range,
                                float min_z_range, float max_z_range, float voxel_x_size,
                                float voxel_y_size, float voxel_z_size, int grid_z_size,
                                int grid_y_size, int grid_x_size, int feature_num,
                                unsigned int *hash_table, unsigned int *real_voxel_num) {
  int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (point_idx >= points_size) {
    return;
  }

  float px = points[feature_num * point_idx];
  float py = points[feature_num * point_idx + 1];
  float pz = points[feature_num * point_idx + 2];

  int voxel_idx = floorf((px - min_x_range) / voxel_x_size);
  if (voxel_idx < 0 || voxel_idx >= grid_x_size) return;

  int voxel_idy = floorf((py - min_y_range) / voxel_y_size);
  if (voxel_idy < 0 || voxel_idy >= grid_y_size) return;

  int voxel_idz = floorf((pz - min_z_range) / voxel_z_size);
  if (voxel_idz < 0 || voxel_idz >= grid_z_size) return;
  unsigned int voxel_offset =
      voxel_idz * grid_y_size * grid_x_size + voxel_idy * grid_x_size + voxel_idx;
  insertHashTable(voxel_offset, real_voxel_num, points_size * 2 * 2, hash_table);
}

__global__ void voxelizationKernel(const half *points, size_t points_size, float min_x_range,
                                   float max_x_range, float min_y_range, float max_y_range,
                                   float min_z_range, float max_z_range, float voxel_x_size,
                                   float voxel_y_size, float voxel_z_size, int grid_z_size,
                                   int grid_y_size, int grid_x_size, int feature_num,
                                   int max_voxels, int max_points_per_voxel,
                                   unsigned int *hash_table, unsigned int *num_points_per_voxel,
                                   float *voxels_temp, unsigned int *voxel_indices,
                                   unsigned int *real_voxel_num, IndiceOrder order) {
  int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (point_idx >= points_size) {
    return;
  }

  float px = points[feature_num * point_idx];
  float py = points[feature_num * point_idx + 1];
  float pz = points[feature_num * point_idx + 2];

  if (px < min_x_range || px >= max_x_range || py < min_y_range || py >= max_y_range ||
      pz < min_z_range || pz >= max_z_range) {
    return;
  }

  int voxel_idx = floorf((px - min_x_range) / voxel_x_size);
  int voxel_idy = floorf((py - min_y_range) / voxel_y_size);
  int voxel_idz = floorf((pz - min_z_range) / voxel_z_size);
  if ((voxel_idx < 0 || voxel_idx >= grid_x_size)) {
    return;
  }
  if ((voxel_idy < 0 || voxel_idy >= grid_y_size)) {
    return;
  }
  if ((voxel_idz < 0 || voxel_idz >= grid_z_size)) {
    return;
  }

  unsigned int voxel_offset =
      voxel_idz * grid_y_size * grid_x_size + voxel_idy * grid_x_size + voxel_idx;

  // scatter to voxels
  unsigned int voxel_id = lookupHashTable(voxel_offset, points_size * 2 * 2, hash_table);
  if (voxel_id >= max_voxels) {
    return;
  }

  unsigned int current_num = atomicAdd(num_points_per_voxel + voxel_id, 1);
  if (current_num < max_points_per_voxel) {
    unsigned int dst_offset =
        voxel_id * (feature_num * max_points_per_voxel) + current_num * feature_num;
    unsigned int src_offset = point_idx * feature_num;
    for (int feature_idx = 0; feature_idx < feature_num; ++feature_idx) {
      voxels_temp[dst_offset + feature_idx] = points[src_offset + feature_idx];
    }

    // now only deal with batch_size = 1
    // since not sure what the input format will be if batch size > 1
    // uint4 idx = {0, voxel_idz, voxel_idy, voxel_idx};
    uint4 idx = {0, (unsigned int)voxel_idx, (unsigned int)voxel_idy, (unsigned int)voxel_idz};
    ((uint4 *)voxel_indices)[voxel_id] = idx;
  }
}

__global__ void featureExtractionKernel(float *voxels_temp, unsigned int *num_points_per_voxel,
                                        const unsigned int *real_voxel_num,
                                        int max_points_per_voxel, int feature_num,
                                        half *voxel_features) {
  int voxel_idx = blockIdx.x * blockDim.x + threadIdx.x;
  num_points_per_voxel[voxel_idx] = num_points_per_voxel[voxel_idx] > max_points_per_voxel
                                        ? max_points_per_voxel
                                        : num_points_per_voxel[voxel_idx];
  int valid_points_num = num_points_per_voxel[voxel_idx];
  int offset = voxel_idx * max_points_per_voxel * feature_num;
  for (int feature_idx = 0; feature_idx < feature_num; ++feature_idx) {
    for (int point_idx = 0; point_idx < valid_points_num - 1; ++point_idx) {
      voxels_temp[offset + feature_idx] +=
          voxels_temp[offset + (point_idx + 1) * feature_num + feature_idx];
    }
    voxels_temp[offset + feature_idx] /= valid_points_num;
  }

  // move to be continuous
  for (int feature_idx = 0; feature_idx < feature_num; ++feature_idx) {
    int dst_offset = voxel_idx * feature_num;
    int src_offset = voxel_idx * feature_num * max_points_per_voxel;
    voxel_features[dst_offset + feature_idx] = __float2half(voxels_temp[src_offset + feature_idx]);
  }
}

cudaError_t featureExtractionLaunch(float *voxels_temp, unsigned int *num_points_per_voxel,
                                    const unsigned int *real_voxel_num, int max_points_per_voxel,
                                    int feature_num, half *voxel_features, cudaStream_t stream) {
  int threadNum = THREADS_FOR_VOXEL;
  dim3 blocks((*real_voxel_num + threadNum - 1) / threadNum);
  dim3 threads(threadNum);
  featureExtractionKernel<<<blocks, threads, 0, stream>>>(voxels_temp, num_points_per_voxel,
                                                          real_voxel_num, max_points_per_voxel,
                                                          feature_num, voxel_features);
  cudaError_t err = cudaGetLastError();
  return err;
}

cudaError_t voxelizationLaunch(const half *points, size_t points_size, float min_x_range,
                               float max_x_range, float min_y_range, float max_y_range,
                               float min_z_range, float max_z_range, float voxel_x_size,
                               float voxel_y_size, float voxel_z_size, int grid_z_size,
                               int grid_y_size, int grid_x_size, int feature_num, int max_voxels,
                               int max_points_per_voxel, unsigned int *hash_table,
                               unsigned int *num_points_per_voxel, float *voxel_features,
                               unsigned int *voxel_indices, unsigned int *real_voxel_num,
                               IndiceOrder order, cudaStream_t stream) {
  int threadNum = THREADS_FOR_VOXEL;
  dim3 blocks((points_size + threadNum - 1) / threadNum);
  dim3 threads(threadNum);
  buildHashKernel<<<blocks, threads, 0, stream>>>(
      points, points_size, min_x_range, max_x_range, min_y_range, max_y_range, min_z_range,
      max_z_range, voxel_x_size, voxel_y_size, voxel_z_size, grid_z_size, grid_y_size, grid_x_size,
      feature_num, hash_table, real_voxel_num);

  voxelizationKernel<<<blocks, threads, 0, stream>>>(
      points, points_size, min_x_range, max_x_range, min_y_range, max_y_range, min_z_range,
      max_z_range, voxel_x_size, voxel_y_size, voxel_z_size, grid_z_size, grid_y_size, grid_x_size,
      feature_num, max_voxels, max_points_per_voxel, hash_table, num_points_per_voxel,
      voxel_features, voxel_indices, real_voxel_num, order);
  cudaError_t err = cudaGetLastError();
  return err;
}

class Params {
 public:
  const unsigned int task_num_stride[6] = {
      0, 1, 3, 5, 6, 8,
  };
  static const unsigned int num_classes = 10;
  const char *class_name[num_classes] = {
      "car",        "truck",   "construction_vehicle", "bus",         "trailer", "barrier",
      "motorcycle", "bicycle", "pedestrian",           "traffic_cone"};

  const float out_size_factor = 8;
  const float voxel_size[2] = {
      0.075,
      0.075,
  };
  const float pc_range[2] = {
      -54,
      -54,
  };
  const float score_threshold = 0.0;
  const float post_center_range[6] = {
      -61.2, -61.2, -10.0, 61.2, 61.2, 10.0,
  };

  const float min_x_range = -54;
  const float max_x_range = 54;
  const float min_y_range = -54;
  const float max_y_range = 54;
  const float min_z_range = -5.0;
  const float max_z_range = 3.0;
  // the size of a pillar
  const float pillar_x_size = 0.075;
  const float pillar_y_size = 0.075;
  const float pillar_z_size = 0.2;
  const int max_points_per_voxel = 10;

  const unsigned int max_voxels = 160000;
  const unsigned int feature_num = 5;

  Params(){};

  int getGridXSize() { return (int)std::round((max_x_range - min_x_range) / pillar_x_size); }
  int getGridYSize() { return (int)std::round((max_y_range - min_y_range) / pillar_y_size); }
  int getGridZSize() { return (int)std::round((max_z_range - min_z_range) / pillar_z_size); }
};

static Params params_;

Voxelization::Voxelization() { alloc_resource(); }

Voxelization::~Voxelization() {
  checkCudaErrors(cudaFree(hash_table_));
  checkCudaErrors(cudaFree(voxels_temp_));

  checkCudaErrors(cudaFree(d_voxel_features_));
  checkCudaErrors(cudaFree(d_voxel_num_));
  checkCudaErrors(cudaFree(d_voxel_indices_));

  checkCudaErrors(cudaFree(d_real_num_voxels_));
  checkCudaErrors(cudaFreeHost(h_real_num_voxels_));
}

unsigned int Voxelization::getOutput(half **d_voxel_features, unsigned int **d_voxel_indices,
                                     std::vector<int> &sparse_shape) {
  *d_voxel_features = d_voxel_features_;
  *d_voxel_indices = d_voxel_indices_;

  sparse_shape.clear();
  // sparse_shape.push_back(params_.getGridZSize() + 1);
  // sparse_shape.push_back(params_.getGridYSize());
  // sparse_shape.push_back(params_.getGridXSize());
  sparse_shape.push_back(params_.getGridXSize());
  sparse_shape.push_back(params_.getGridYSize());
  sparse_shape.push_back(params_.getGridZSize() + 1);
  return *h_real_num_voxels_;
}

int Voxelization::alloc_resource() {
  hash_table_size_ = MAX_POINTS_NUM * 2 * 2 * sizeof(unsigned int);

  voxels_temp_size_ =
      params_.max_voxels * params_.max_points_per_voxel * params_.feature_num * sizeof(float);
  voxel_features_size_ =
      params_.max_voxels * params_.max_points_per_voxel * params_.feature_num * sizeof(half);

  checkCudaErrors(cudaMallocManaged((void **)&hash_table_, hash_table_size_));
  checkCudaErrors(cudaMallocManaged((void **)&voxels_temp_, voxels_temp_size_));

  voxel_num_size_ = params_.max_voxels * sizeof(unsigned int);
  voxel_idxs_size_ = params_.max_voxels * 4 * sizeof(unsigned int);

  checkCudaErrors(cudaMallocManaged((void **)&d_voxel_features_, voxel_features_size_));
  checkCudaErrors(cudaMallocManaged((void **)&d_voxel_num_, voxel_num_size_));
  checkCudaErrors(cudaMallocManaged((void **)&d_voxel_indices_, voxel_idxs_size_));
  checkCudaErrors(cudaMalloc((void **)&d_real_num_voxels_, sizeof(unsigned int)));
  checkCudaErrors(cudaMallocHost((void **)&h_real_num_voxels_, sizeof(unsigned int)));

  checkCudaErrors(cudaMemset(d_voxel_num_, 0, voxel_num_size_));
  checkCudaErrors(cudaMemset(d_voxel_features_, 0, voxel_features_size_));
  checkCudaErrors(cudaMemset(d_voxel_indices_, 0, voxel_idxs_size_));
  checkCudaErrors(cudaMemset(d_real_num_voxels_, 0, sizeof(unsigned int)));

  return 0;
}

int Voxelization::generateVoxels(const half *points, size_t points_size, IndiceOrder order,
                                 cudaStream_t stream_) {
  // flash memory for every run
  checkCudaErrors(cudaMemsetAsync(hash_table_, 0xff, hash_table_size_, stream_));
  checkCudaErrors(cudaMemsetAsync(voxels_temp_, 0xff, voxels_temp_size_, stream_));

  checkCudaErrors(cudaMemsetAsync(d_voxel_num_, 0, voxel_num_size_, stream_));
  checkCudaErrors(cudaMemsetAsync(d_real_num_voxels_, 0, sizeof(unsigned int), stream_));

  checkCudaErrors(cudaStreamSynchronize(stream_));

  checkCudaErrors(voxelizationLaunch(
      points, points_size, params_.min_x_range, params_.max_x_range, params_.min_y_range,
      params_.max_y_range, params_.min_z_range, params_.max_z_range, params_.pillar_x_size,
      params_.pillar_y_size, params_.pillar_z_size, params_.getGridZSize(), params_.getGridYSize(),
      params_.getGridXSize(), params_.feature_num, params_.max_voxels, params_.max_points_per_voxel,
      hash_table_, d_voxel_num_,
      /*d_voxel_features_*/ voxels_temp_, d_voxel_indices_, d_real_num_voxels_, order, stream_));
  checkCudaErrors(cudaMemcpyAsync(h_real_num_voxels_, d_real_num_voxels_, sizeof(int),
                                  cudaMemcpyDeviceToHost, stream_));
  checkCudaErrors(cudaStreamSynchronize(stream_));

  checkCudaErrors(featureExtractionLaunch(voxels_temp_, d_voxel_num_, h_real_num_voxels_,
                                          params_.max_points_per_voxel, params_.feature_num,
                                          d_voxel_features_, stream_));

  checkCudaErrors(cudaStreamSynchronize(stream_));
  return 0;
}