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

#ifndef __LIDAR_VOXELIZATION_HPP__
#define __LIDAR_VOXELIZATION_HPP__

#include <memory>
#include <vector>

#include "common/dtype.hpp"

namespace bevfusion {
namespace lidar {

enum class CoordinateOrder : int {
  NoneOrder = 0,
  XYZ = 1,  // BEVFusion
  ZYX = 2   // CenterPoint
};

struct VoxelizationParameter {
  nvtype::Float3 min_range;
  nvtype::Float3 max_range;
  nvtype::Float3 voxel_size;
  nvtype::Int3 grid_size;
  int num_feature;
  int max_voxels;
  int max_points_per_voxel;
  int max_points;

  static nvtype::Int3 compute_grid_size(const nvtype::Float3& max_range, const nvtype::Float3& min_range,
                                        const nvtype::Float3& voxel_size);
};

class Voxelization {
 public:
  virtual ~Voxelization() = default;
  // points and voxels must be of half-float device pointer
  virtual void forward(const nvtype::half* points, int num_points, void* stream = nullptr,
                       CoordinateOrder output_order = CoordinateOrder::XYZ) = 0;

  virtual unsigned int num_voxels() = 0;
  virtual unsigned int voxel_dim() = 0;
  virtual unsigned int indices_dim() = 0;
  virtual std::vector<int> grid_size() = 0;
  virtual const void* indices() = 0;
  virtual const void* features() = 0;
  virtual CoordinateOrder order() = 0;
};

std::shared_ptr<Voxelization> create_voxelization(VoxelizationParameter parameter);

};  // namespace lidar
};  // namespace bevfusion

#endif  // __LIDAR_VOXELIZATION_HPP__