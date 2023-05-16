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

#ifndef __HEAD_TRANSBBOX_HPP__
#define __HEAD_TRANSBBOX_HPP__

#include <memory>
#include <string>
#include <vector>

#include "common/dtype.hpp"

namespace bevfusion {
namespace head {
namespace transbbox {

struct TransBBoxParameter {
  std::string model;
  float out_size_factor = 8;
  nvtype::Float2 voxel_size{0.075, 0.075};
  nvtype::Float2 pc_range{-54.0f, -54.0f};
  nvtype::Float3 post_center_range_start{-61.2, -61.2, -10.0};
  nvtype::Float3 post_center_range_end{61.2, 61.2, 10.0};
  float confidence_threshold = 0.0f;
  bool sorted_bboxes = true;
};

struct Position {
  float x, y, z;
};

struct Size {
  float w, l, h;  // x, y, z
};

struct Velocity {
  float vx, vy;
};

struct BoundingBox {
  Position position;
  Size size;
  Velocity velocity;
  float z_rotation;
  float score;
  int id;
};

class TransBBox {
 public:
  virtual std::vector<BoundingBox> forward(const nvtype::half* transfusion_feature, float confidence_threshold, void* stream,
                                           bool sorted_by_conf = false) = 0;
  virtual void print() = 0;
};

std::shared_ptr<TransBBox> create_transbbox(const TransBBoxParameter& param);

};  // namespace transbbox
};  // namespace head
};  // namespace bevfusion

#endif  // __HEAD_TRANSBBOX_HPP__