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

#ifndef __CAMERA_NORMALIZATION_HPP__
#define __CAMERA_NORMALIZATION_HPP__

#include <initializer_list>
#include <memory>

#include "common/dtype.hpp"

namespace bevfusion {
namespace camera {

enum class NormType : int { Nothing = 0, MeanStd = 1, AlphaBeta = 2 };
enum class ChannelType : int { Nothing = 0, Invert = 1 };
enum class Interpolation : int { Nearest = 0, Bilinear = 1 };

struct NormMethod {
  float mean[3];
  float std[3];
  float alpha, beta;
  NormType type = NormType::Nothing;
  ChannelType channel_type = ChannelType::Nothing;

  // out = (x * alpha - mean) / std + beta
  static NormMethod mean_std(const float mean[3], const float std[3], float alpha = 1 / 255.0f, float beta = 0.0f,
                             ChannelType channel_type = ChannelType::Nothing);

  // out = x * alpha + beta
  static NormMethod alpha_beta(float alpha, float beta = 0, ChannelType channel_type = ChannelType::Nothing);

  // None
  static NormMethod None();
};

struct NormalizationParameter {
  int image_width;
  int image_height;
  int num_camera;
  int output_width;
  int output_height;
  float resize_lim;
  Interpolation interpolation;
  NormMethod method;
};

class Normalization {
 public:
  virtual ~Normalization() = default;
  // Here you should provide num_camera 3-channel images with the same dimensions as the width and
  // height provided at create time. The images pointer should be the host address. The pipeline is
  // as follows:
  // Step1: load pixel for interpolation.
  // Step2: execute ChannelType transformation, RGB->BGR etc.
  // Step3: execute Normalize transformation, MeanStd or AlphaBeta.
  // Step4: store as a planar memory.
  virtual nvtype::half* forward(const unsigned char** images, void* stream) = 0;
};

std::shared_ptr<Normalization> create_normalization(const NormalizationParameter& param);

};  // namespace camera
};  // namespace bevfusion

#endif  // __CAMERA_NORMALIZATION_HPP__