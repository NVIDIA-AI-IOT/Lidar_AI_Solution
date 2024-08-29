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

#ifndef __BEVFUSION_HPP__
#define __BEVFUSION_HPP__

#include "camera-backbone.hpp"
#include "camera-bevpool.hpp"
#include "camera-depth.hpp"
#include "camera-geometry.hpp"
#include "camera-normalization.hpp"
#include "camera-vtransform.hpp"
#include "head-transbbox.hpp"
#include "lidar-scn.hpp"
#include "transfusion.hpp"

namespace bevfusion {

struct CoreParameter {
  std::string camera_model;
  std::string camera_vtransform;
  camera::GeometryParameter geometry;
  camera::NormalizationParameter normalize;
  lidar::SCNParameter lidar_scn;
  std::string transfusion;
  head::transbbox::TransBBoxParameter transbbox;
};

class Core {
 public:
  virtual ~Core() = default;
  virtual std::vector<head::transbbox::BoundingBox> forward(const unsigned char **camera_images, const nvtype::half *lidar_points,
                                                            int num_points, void *stream) = 0;

  virtual std::vector<head::transbbox::BoundingBox> forward_no_normalize(const nvtype::half *camera_normed_images_device,
                                                                         const nvtype::half *lidar_points, int num_points,
                                                                         void *stream) = 0;

  virtual void print() = 0;
  virtual void set_timer(bool enable) = 0;

  virtual void update(const float *camera2lidar, const float *camera_intrinsics, const float *lidar2image,
                      const float *img_aug_matrix, void *stream = nullptr) = 0;

  // Consider releasing excess memory if you don't need to update the matrix
  // After freeing the memory, the update function call will raise a logical exception.
  virtual void free_excess_memory() = 0;
};

std::shared_ptr<Core> create_core(const CoreParameter &param);

};  // namespace bevfusion

#endif  // __BEVFUSION_HPP__