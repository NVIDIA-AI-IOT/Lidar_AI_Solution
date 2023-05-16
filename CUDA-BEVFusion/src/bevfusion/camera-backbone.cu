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

#include <cuda_fp16.h>

#include <numeric>

#include "camera-backbone.hpp"
#include "common/check.hpp"
#include "common/launch.cuh"
#include "common/tensorrt.hpp"

namespace bevfusion {
namespace camera {

class BackboneImplement : public Backbone {
 public:
  virtual ~BackboneImplement() {
    if (feature_) checkRuntime(cudaFree(feature_));
    if (depth_weights_) checkRuntime(cudaFree(depth_weights_));
  }

  bool init(const std::string& model) {
    engine_ = TensorRT::load(model);
    if (engine_ == nullptr) return false;

    depth_dims_ = engine_->static_dims(2);
    feature_dims_ = engine_->static_dims(3);
    int32_t volumn = std::accumulate(depth_dims_.begin(), depth_dims_.end(), 1, std::multiplies<int32_t>());
    checkRuntime(cudaMalloc(&depth_weights_, volumn * sizeof(nvtype::half)));

    volumn = std::accumulate(feature_dims_.begin(), feature_dims_.end(), 1, std::multiplies<int32_t>());
    checkRuntime(cudaMalloc(&feature_, volumn * sizeof(nvtype::half)));

    // N C D H W
    camera_shape_ = {feature_dims_[0], feature_dims_[3], depth_dims_[1], feature_dims_[1], feature_dims_[2]};
    return true;
  }

  virtual void print() override { engine_->print("Camerea Backbone"); }

  virtual void forward(const nvtype::half* images, const nvtype::half* depth, void* stream = nullptr) override {
    engine_->forward({images, depth, depth_weights_, feature_}, static_cast<cudaStream_t>(stream));
  }

  virtual nvtype::half* depth() override { return depth_weights_; }
  virtual nvtype::half* feature() override { return feature_; }
  virtual std::vector<int> depth_shape() override { return depth_dims_; }
  virtual std::vector<int> feature_shape() override { return feature_dims_; }
  virtual std::vector<int> camera_shape() override { return camera_shape_; }

 private:
  std::shared_ptr<TensorRT::Engine> engine_;
  nvtype::half* feature_ = nullptr;
  nvtype::half* depth_weights_ = nullptr;
  std::vector<int> feature_dims_, depth_dims_, camera_shape_;
};

std::shared_ptr<Backbone> create_backbone(const std::string& model) {
  std::shared_ptr<BackboneImplement> instance(new BackboneImplement());
  if (!instance->init(model)) {
    instance.reset();
  }
  return instance;
}

};  // namespace camera
};  // namespace bevfusion