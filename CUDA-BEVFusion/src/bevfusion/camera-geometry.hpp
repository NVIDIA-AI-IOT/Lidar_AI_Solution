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

#ifndef __CAMERA_GEOMETRY_HPP__
#define __CAMERA_GEOMETRY_HPP__

#include <memory>

#include "common/dtype.hpp"

namespace bevfusion {
namespace camera {

struct GeometryParameter {
  nvtype::Float3 xbound;
  nvtype::Float3 ybound;
  nvtype::Float3 zbound;
  nvtype::Float3 dbound;
  nvtype::Int3 geometry_dim;  // w(x 360), h(y 360), c(z 80)
  unsigned int feat_width;
  unsigned int feat_height;
  unsigned int image_width;
  unsigned int image_height;
  unsigned int num_camera;
};

class Geometry {
 public:
  virtual nvtype::Int3* intervals() = 0;
  virtual unsigned int num_intervals() = 0;

  virtual unsigned int* indices() = 0;
  virtual unsigned int num_indices() = 0;

  // You can call this function if you need to update the matrix
  // All matrix pointers must be on the host
  // img_aug_matrix is num_camera x 4 x 4 matrix on host
  // lidar2image    is num_camera x 4 x 4 matrix on host
  virtual void update(const float* camera2lidar, const float* camera_intrinsics, const float* img_aug_matrix,
                      void* stream = nullptr) = 0;

  // Consider releasing excess memory if you don't need to update the matrix
  // After freeing the memory, the update function call will raise a logical exception.
  virtual void free_excess_memory() = 0;
};

std::shared_ptr<Geometry> create_geometry(GeometryParameter param);

};  // namespace camera
};  // namespace bevfusion

#endif  // __CAMERA_GEOMETRY_HPP__