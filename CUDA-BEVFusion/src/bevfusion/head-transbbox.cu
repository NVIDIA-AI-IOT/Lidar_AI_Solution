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

#include <algorithm>
#include <numeric>

#include "common/check.hpp"
#include "common/launch.cuh"
#include "common/tensorrt.hpp"
#include "head-transbbox.hpp"

namespace bevfusion {
namespace head {
namespace transbbox {

#define MAX_DETECTION_BOX_SIZE 1024

typedef struct {
  half x, y, z;
} half3;

static __global__ void decode_kernel(unsigned int num, const half* reg, const half* height, const half* dim, const half* rot,
                                     const half* vel, const half* score, int num_class, TransBBoxParameter param,
                                     float confidence_threshold, BoundingBox* output, unsigned int* output_size,
                                     unsigned int max_output_size) {
  int ibox = cuda_linear_index;
  if (ibox >= num) return;

  int label = 0;
  float confidence = score[0 * num + ibox];

  for (int i = 1; i < num_class; ++i) {
    float local_score = score[i * num + ibox];
    if (local_score > confidence) {
      label = i;
      confidence = local_score;
    }
  }

  if (confidence < confidence_threshold) return;

  auto xs = __half2float(reg[0 * num + ibox]);
  auto ys = __half2float(reg[1 * num + ibox]);
  xs = xs * param.out_size_factor * param.voxel_size.x + param.pc_range.x;
  ys = ys * param.out_size_factor * param.voxel_size.y + param.pc_range.y;

  auto zs = __half2float(height[ibox]);
  if (xs < param.post_center_range_start.x || xs > param.post_center_range_end.x) return;
  if (ys < param.post_center_range_start.y || ys > param.post_center_range_end.y) return;

  float3 dim_;
  dim_.x = exp(__half2float(dim[0 * num + ibox]));
  dim_.y = exp(__half2float(dim[1 * num + ibox]));
  dim_.z = exp(__half2float(dim[2 * num + ibox]));
  zs = zs - dim_.z * 0.5f;

  if (zs < param.post_center_range_start.z || zs > param.post_center_range_end.z) return;

  unsigned int iout = atomicAdd(output_size, 1);
  if (iout >= max_output_size) return;

  auto& obox = output[iout];
  auto vx = __half2float(vel[0 * num + ibox]);
  auto vy = __half2float(vel[1 * num + ibox]);
  auto rs = atan2(__half2float(rot[0 * num + ibox]), __half2float(rot[1 * num + ibox]));

  *(float3*)&obox.position = make_float3(xs, ys, zs);
  *(float3*)&obox.size = dim_;
  obox.velocity.vx = vx;
  obox.velocity.vy = vy;
  obox.z_rotation = rs;
  obox.score = confidence;
  obox.id = label;
}

class TransBBoxImplement : public TransBBox {
 public:
  // "middle", "reg", "height", "dim", "rot", "vel", "score"
  const char* BindingMiddle = "middle";  // input
  const char* BindingReg    = "reg";     // output
  const char* BindingHeight = "height";
  const char* BindingDim    = "dim";
  const char* BindingRot    = "rot";
  const char* BindingVel    = "vel";
  const char* BindingScore  = "score";

  virtual ~TransBBoxImplement() {
    for (size_t i = 0; i < bindings_.size(); ++i) checkRuntime(cudaFree(bindings_[i]));

    if (output_device_size_) checkRuntime(cudaFree(output_device_size_));
    if (output_device_boxes_) checkRuntime(cudaFree(output_device_boxes_));
    if (output_host_size_) checkRuntime(cudaFreeHost(output_host_size_));
    if (output_host_boxes_) checkRuntime(cudaFreeHost(output_host_boxes_));
  }

  virtual bool init(const TransBBoxParameter& param) {
    engine_ = TensorRT::load(param.model);
    if (engine_ == nullptr) return false;

    if (engine_->has_dynamic_dim()) {
      printf("Dynamic shapes are not supported.\n");
      return false;
    }

    param_ = param;
    create_binding_memory();
    checkRuntime(cudaMalloc(&output_device_size_, sizeof(unsigned int)));
    checkRuntime(cudaMalloc(&output_device_boxes_, MAX_DETECTION_BOX_SIZE * sizeof(BoundingBox)));
    checkRuntime(cudaMallocHost(&output_host_size_, sizeof(unsigned int)));
    checkRuntime(cudaMallocHost(&output_host_boxes_, MAX_DETECTION_BOX_SIZE * sizeof(BoundingBox)));
    output_cache_.resize(MAX_DETECTION_BOX_SIZE);
    return true;
  }

  void create_binding_memory() {
    const char* bindings[] = {BindingMiddle, BindingReg, BindingHeight, BindingDim, BindingRot, BindingVel, BindingScore};
    for (size_t i = 0; i < sizeof(bindings) / sizeof(bindings[0]); ++i) {
      if (engine_->is_input(bindings[i])) continue;

      auto shape = engine_->static_dims(bindings[i]);
      Asserts(engine_->dtype(bindings[i]) == TensorRT::DType::HALF, "Invalid binding data type.");

      size_t volumn = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
      half* pdata = nullptr;
      checkRuntime(cudaMalloc(&pdata, volumn * sizeof(half)));

      bindshape_.push_back(shape);
      bindings_.push_back(pdata);
    }
    Assertf(bindings_.size() == 6, "Invalid output num of bindings[%d]", static_cast<int>(bindings_.size()));
  }

  virtual void print() override { engine_->print("BBox"); }

  virtual std::vector<BoundingBox> forward(const nvtype::half* transfusion_feature, float confidence_threshold, void* stream,
                                           bool sorted) override {
    cudaStream_t _stream = static_cast<cudaStream_t>(stream);

    engine_->forward(std::unordered_map<std::string, const void *>{
          {BindingMiddle, transfusion_feature}, 
          {BindingReg, bindings_[0]},
          {BindingHeight, bindings_[1]},
          {BindingDim, bindings_[2]},
          {BindingRot, bindings_[3]},
          {BindingVel, bindings_[4]},
          {BindingScore, bindings_[5]}
        }, _stream);
    checkRuntime(cudaMemsetAsync(output_device_size_, 0, sizeof(unsigned int), _stream));

    cuda_linear_launch(decode_kernel, _stream, bindshape_[0][2], bindings_[0], bindings_[1], bindings_[2], bindings_[3],
                       bindings_[4], bindings_[5], bindshape_[5][1], param_, confidence_threshold, output_device_boxes_,
                       output_device_size_, MAX_DETECTION_BOX_SIZE);

    // int num_outbox = min(MAX_DETECTION_BOX_SIZE, )
    checkRuntime(cudaMemcpyAsync(output_host_boxes_, output_device_boxes_, MAX_DETECTION_BOX_SIZE * sizeof(BoundingBox),
                                 cudaMemcpyDeviceToHost, _stream));
    checkRuntime(cudaMemcpyAsync(output_host_size_, output_device_size_, sizeof(unsigned int), cudaMemcpyDeviceToHost, _stream));
    checkRuntime(cudaStreamSynchronize(_stream));

    unsigned int real_size = min(MAX_DETECTION_BOX_SIZE, *output_host_size_);
    auto output = std::vector<BoundingBox>(output_host_boxes_, output_host_boxes_ + real_size);
    if (sorted) {
      std::sort(output.begin(), output.end(), [](BoundingBox& a, BoundingBox& b) { return a.score > b.score; });
    }
    return output;
  }

 private:
  std::shared_ptr<TensorRT::Engine> engine_;
  std::vector<half*> bindings_;
  std::vector<std::vector<int>> bindshape_;
  TransBBoxParameter param_;
  std::vector<BoundingBox> output_cache_;
  BoundingBox* output_host_boxes_ = nullptr;
  BoundingBox* output_device_boxes_ = nullptr;
  unsigned int* output_device_size_ = nullptr;
  unsigned int* output_host_size_ = nullptr;
};

std::shared_ptr<TransBBox> create_transbbox(const TransBBoxParameter& param) {
  std::shared_ptr<TransBBoxImplement> instance(new TransBBoxImplement());
  if (!instance->init(param)) {
    instance.reset();
  }
  return instance;
}

};  // namespace transbbox
};  // namespace head
};  // namespace bevfusion