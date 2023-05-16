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
 
#include "preprocess.h"
#include <assert.h>
#include <iostream>

PreProcessCuda::PreProcessCuda()
{}

PreProcessCuda::~PreProcessCuda()
{
    checkCudaErrors(cudaFree(hash_table_));
    checkCudaErrors(cudaFree(voxels_temp_));

    checkCudaErrors(cudaFree(d_voxel_features_));
    checkCudaErrors(cudaFree(d_voxel_num_));
    checkCudaErrors(cudaFree(d_voxel_indices_));

    checkCudaErrors(cudaFree(d_real_num_voxels_));
    checkCudaErrors(cudaFreeHost(h_real_num_voxels_));
}

unsigned int PreProcessCuda::getOutput(half** d_voxel_features, unsigned int** d_voxel_indices, std::vector<int>& sparse_shape){
    *d_voxel_features = d_voxel_features_;
    *d_voxel_indices = d_voxel_indices_;

    sparse_shape.clear();
    sparse_shape.push_back(params_.getGridZSize() + 1);
    sparse_shape.push_back(params_.getGridYSize());
    sparse_shape.push_back(params_.getGridXSize());

    return *h_real_num_voxels_;
}

int PreProcessCuda::alloc_resource(){
    hash_table_size_ = MAX_POINTS_NUM * 2 * 2 * sizeof(unsigned int);

    voxels_temp_size_ = params_.max_voxels * params_.max_points_per_voxel * params_.feature_num * sizeof(float);
    voxel_features_size_ = params_.max_voxels * params_.max_points_per_voxel * params_.feature_num * sizeof(half);

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

int PreProcessCuda::generateVoxels(const float *points, size_t points_size, cudaStream_t stream)
{
    // flash memory for every run 
    checkCudaErrors(cudaMemsetAsync(hash_table_, 0xff, hash_table_size_, stream));
    checkCudaErrors(cudaMemsetAsync(voxels_temp_, 0xff, voxels_temp_size_, stream));

    checkCudaErrors(cudaMemsetAsync(d_voxel_num_, 0, voxel_num_size_, stream));
    checkCudaErrors(cudaMemsetAsync(d_real_num_voxels_, 0, sizeof(unsigned int), stream));
    checkCudaErrors(cudaStreamSynchronize(stream));

    checkCudaErrors(voxelizationLaunch(points, points_size,
          params_.min_x_range, params_.max_x_range,
          params_.min_y_range, params_.max_y_range,
          params_.min_z_range, params_.max_z_range,
          params_.pillar_x_size, params_.pillar_y_size, params_.pillar_z_size,
          params_.getGridYSize(), params_.getGridXSize(), params_.feature_num, params_.max_voxels,
          params_.max_points_per_voxel, hash_table_,
    d_voxel_num_, /*d_voxel_features_*/voxels_temp_, d_voxel_indices_,
    d_real_num_voxels_, stream));
    checkCudaErrors(cudaMemcpyAsync(h_real_num_voxels_, d_real_num_voxels_, sizeof(int), cudaMemcpyDeviceToHost, stream));
    checkCudaErrors(cudaStreamSynchronize(stream));

    checkCudaErrors(featureExtractionLaunch(voxels_temp_, d_voxel_num_,
          *h_real_num_voxels_, params_.max_points_per_voxel, params_.feature_num,
    d_voxel_features_, stream));

    checkCudaErrors(cudaStreamSynchronize(stream));
    return 0;
}