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
 
#include <memory>

#include "common.h"
#include "NvInfer.h"
#include "NvOnnxConfig.h"
#include "NvInferRuntime.h"
#include "preprocess.h"
#include "postprocess.h"
#include "spconv/engine.hpp"
#include "tensorrt.hpp"
#include "timer.hpp"

typedef struct float11 { float val[11]; } float11;

class CenterPoint {
  private:
    Params params_;
    bool verbose_;

    std::shared_ptr<PreProcessCuda> pre_;
    std::shared_ptr<spconv::Engine> scn_engine_;
    std::shared_ptr<TensorRT::Engine> trt_;
    std::shared_ptr<PostProcessCuda> post_;

    std::vector<float> timing_pre_;
    std::vector<float> timing_scn_engine_;
    std::vector<float> timing_trt_;
    std::vector<float> timing_post_;

    unsigned int* h_detections_num_;
    float* d_detections_;
    float* d_detections_reshape_;     //add d_detections_reshape_

    half* d_reg_[NUM_TASKS];
    half* d_height_[NUM_TASKS];
    half* d_dim_[NUM_TASKS];
    half* d_rot_[NUM_TASKS];
    half* d_vel_[NUM_TASKS];
    half* d_hm_[NUM_TASKS];

    int reg_n_;
    int reg_c_;
    int reg_h_;
    int reg_w_;
    int height_c_;
    int dim_c_;
    int rot_c_;
    int vel_c_;
    int hm_c_[NUM_TASKS];

    half* d_voxel_features;
    unsigned int* d_voxel_indices;
    std::vector<int> sparse_shape;

    std::vector<float11> detections_;
    unsigned int h_mask_size_;
    uint64_t* h_mask_ = nullptr;
    EventTimer timer_;

  public:
    CenterPoint(std::string modelFile, bool verbose = false);
    ~CenterPoint(void);

    int prepare();
    int doinfer(void* points, unsigned int point_num, cudaStream_t stream);
    std::vector<Bndbox> nms_pred_;
    void perf_report();
};