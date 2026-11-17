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

#pragma once

#include <cstdint>
#include <memory>

#include "common/dtype.hpp"

namespace bevfusion {
namespace lidar {

// Raw device bytes. Each field is a scalar FLOAT32; offsets need not be aligned.
struct PointCloudLayout {
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t point_step = 0;
  uint32_t row_step = 0;
  uint32_t offsets[4] = {};  // x, y, z, intensity
  bool is_bigendian = false;
};

struct PreparedLidar {
  const nvtype::half* points = nullptr;
  int count = 0;
};

class Preprocessor {
 public:
  virtual ~Preprocessor() = default;
  // Stable filtering and truncation, followed by FP16 [x,y,z,intensity,0] packing.
  // Synchronizes stream to return the scalar count. The point array stays on the GPU.
  // The returned buffer is owned by this object and valid until the next call.
  virtual PreparedLidar forward(const void* device_points, const PointCloudLayout& layout, void* stream) = 0;
};

// Allocates all workspace up front. Throws on failure so callers can select a CPU backend.
std::shared_ptr<Preprocessor> create_preprocessor(int max_input_points, int max_output_points);

}  // namespace lidar
}  // namespace bevfusion
