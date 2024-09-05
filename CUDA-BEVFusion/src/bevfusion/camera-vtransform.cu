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

#include "camera-vtransform.hpp"
#include "common/check.hpp"
#include "common/launch.cuh"
#include "common/tensorrt.hpp"

namespace bevfusion {
namespace camera {

class VTransformImplement : public VTransform {
 public:
  const char* BindingInput  = "feat_in";
  const char* BindingOutput = "feat_out";

  virtual ~VTransformImplement() {
    if (output_feature_) checkRuntime(cudaFree(output_feature_));
  }

  bool init(const std::string& model) {
    engine_ = TensorRT::load(model);
    if (engine_ == nullptr) return false;

    output_dims_ = engine_->static_dims(BindingOutput);
    int32_t volumn = std::accumulate(output_dims_.begin(), output_dims_.end(), 1, std::multiplies<int32_t>());
    checkRuntime(cudaMalloc(&output_feature_, volumn * sizeof(nvtype::half)));
    return true;
  }

  virtual void print() override { engine_->print("Camerea VTransform"); }

  virtual nvtype::half* forward(const nvtype::half* camera_bev, void* stream = nullptr) override {
    engine_->forward({
      {BindingInput, camera_bev},
      {BindingOutput, output_feature_}
    }, static_cast<cudaStream_t>(stream));
    return output_feature_;
  }

  virtual std::vector<int> feat_shape() override { return output_dims_; }

 private:
  std::shared_ptr<TensorRT::Engine> engine_;
  nvtype::half* output_feature_ = nullptr;
  std::vector<int> output_dims_;
};

std::shared_ptr<VTransform> create_vtransform(const std::string& model) {
  std::shared_ptr<VTransformImplement> instance(new VTransformImplement());
  if (!instance->init(model)) {
    instance.reset();
  }
  return instance;
}

};  // namespace camera
};  // namespace bevfusion