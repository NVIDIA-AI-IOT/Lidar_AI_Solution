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

#ifndef __LIDAR_SCN_HPP__
#define __LIDAR_SCN_HPP__

#include <vector>

#include "lidar-voxelization.hpp"

namespace bevfusion {
namespace lidar {

// use model accuracy during SCN model inference.
enum class Precision : int { NonePrecision = 0, Float16 = 1, Int8 = 2 };

struct SCNParameter {
  VoxelizationParameter voxelization;
  std::string model;
  CoordinateOrder order = CoordinateOrder::XYZ;
  Precision precision = Precision::Float16;
};

class SCN {
 public:
  // points and voxels must be of half-float device pointer
  virtual const nvtype::half* forward(const nvtype::half* points, unsigned int num_points, void* stream = nullptr) = 0;
  virtual std::vector<int64_t> shape() = 0;
};

std::shared_ptr<SCN> create_scn(const SCNParameter& param);

};  // namespace lidar
};  // namespace bevfusion

#endif  // __LIDAR_SCN_HPP__