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

#ifndef __CAMERA_BEVPOOL_HPP__
#define __CAMERA_BEVPOOL_HPP__

#include <memory>
#include <string>
#include <vector>

#include "common/dtype.hpp"

namespace bevfusion {
namespace camera {

class BEVPool {
 public:
  virtual ~BEVPool() = default;
  virtual nvtype::half* forward(const nvtype::half* camera_feature, const nvtype::half* depth_weights,
                                const unsigned int* indices, const nvtype::Int3* intervals, unsigned int num_intervals,
                                void* stream = nullptr) = 0;
  virtual std::vector<int> shape() = 0;
};

std::shared_ptr<BEVPool> create_bevpool(const std::vector<int>& camera_shape, unsigned int bev_width, unsigned int bev_height);

};  // namespace camera
};  // namespace bevfusion

#endif  // __CAMERA_BEVPOOL_HPP__