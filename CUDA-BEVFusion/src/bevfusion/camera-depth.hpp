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

#ifndef __CAMERA_DEPTH_HPP__
#define __CAMERA_DEPTH_HPP__

#include <memory>

#include "common/dtype.hpp"

namespace bevfusion {
namespace camera {

class Depth {
 public:
  // points must be of half-float type
  // return value will is a num_camera x image_height x image_width Tensor
  virtual nvtype::half* forward(const nvtype::half* points, int num_points, int points_dim, void* stream = nullptr) = 0;

  // You can call this function if you need to update the matrix
  // All matrix pointers must be on the host
  // img_aug_matrix is num_camera x 4 x 4 matrix on host
  // lidar2image    is num_camera x 4 x 4 matrix on host
  virtual void update(const float* img_aug_matrix, const float* lidar2image, void* stream = nullptr) = 0;
};

std::shared_ptr<Depth> create_depth(unsigned int image_width, unsigned int image_height, unsigned int num_camera);

};  // namespace camera
};  // namespace bevfusion

#endif  // __CAMERA_DEPTH_HPP__