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
 
#include "centerpoint.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include "NvInfer.h"
#include "NvOnnxConfig.h"
#include "NvInferRuntime.h"
#include "timer.hpp"

#include <algorithm>
#include <math.h>
#include <sys/time.h>
#include <unistd.h>

template<typename T>
double getAverage(std::vector<T> const& v) {
    if (v.empty()) {
        return 0;
    }
    return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

CenterPoint::CenterPoint(std::string modelFile, bool verbose): verbose_(verbose)
{
    trt_ = TensorRT::load(modelFile);
    if(trt_ == nullptr) abort();

    pre_.reset(new PreProcessCuda());
    post_.reset(new PostProcessCuda());

    scn_engine_ = spconv::load_engine_from_onnx("../model/centerpoint.scn.onnx");

    checkCudaErrors(cudaMallocHost((void **)&h_detections_num_, sizeof(unsigned int)));
    checkCudaErrors(cudaMemset(h_detections_num_, 0, sizeof(unsigned int)));

    checkCudaErrors(cudaMalloc((void **)&d_detections_, MAX_DET_NUM * DET_CHANNEL * sizeof(float)));
    checkCudaErrors(cudaMemset(d_detections_, 0, MAX_DET_NUM * DET_CHANNEL * sizeof(float)));

    //add d_detections_reshape_
    checkCudaErrors(cudaMalloc((void **)&d_detections_reshape_, MAX_DET_NUM * DET_CHANNEL * sizeof(float)));
    checkCudaErrors(cudaMemset(d_detections_reshape_, 0, MAX_DET_NUM * DET_CHANNEL * sizeof(float)));

    detections_.resize(MAX_DET_NUM, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});

    for(unsigned int i=0; i < NUM_TASKS; i++) {
        checkCudaErrors(cudaMalloc((void **)&d_reg_[i], trt_->getBindingNumel("reg_" + std::to_string(i)) * sizeof(half)));
        checkCudaErrors(cudaMalloc((void **)&d_height_[i], trt_->getBindingNumel("height_" + std::to_string(i)) * sizeof(half)));
        checkCudaErrors(cudaMalloc((void **)&d_dim_[i], trt_->getBindingNumel("dim_" + std::to_string(i)) * sizeof(half)));
        checkCudaErrors(cudaMalloc((void **)&d_rot_[i], trt_->getBindingNumel("rot_" + std::to_string(i)) * sizeof(half)));
        checkCudaErrors(cudaMalloc((void **)&d_vel_[i], trt_->getBindingNumel("vel_" + std::to_string(i)) * sizeof(half)));
        checkCudaErrors(cudaMalloc((void **)&d_hm_[i], trt_->getBindingNumel("hm_" + std::to_string(i)) * sizeof(half)));

        if(i==0){
            auto d = trt_->getBindingDims("reg_" + std::to_string(i));
            reg_n_ = d[0];
            reg_c_ = d[1];
            reg_h_ = d[2];
            reg_w_ = d[3];

            d = trt_->getBindingDims("height_" + std::to_string(i));
            height_c_ = d[1];
            d = trt_->getBindingDims("dim_" + std::to_string(i));
            dim_c_ = d[1];
            d = trt_->getBindingDims("rot_" + std::to_string(i));
            rot_c_ = d[1];
            d = trt_->getBindingDims("vel_" + std::to_string(i));
            vel_c_ = d[1];
        }
        auto d = trt_->getBindingDims("hm_" + std::to_string(i));
        hm_c_[i] = d[1];
    }
    h_mask_size_ = params_.nms_pre_max_size * DIVUP(params_.nms_pre_max_size, NMS_THREADS_PER_BLOCK) * sizeof(uint64_t);
    checkCudaErrors(cudaMallocHost((void **)&h_mask_, h_mask_size_));
    checkCudaErrors(cudaMemset(h_mask_, 0, h_mask_size_));
    return;
}

CenterPoint::~CenterPoint(void)
{
    pre_.reset();
    trt_.reset();
    post_.reset();
    scn_engine_.reset();

    checkCudaErrors(cudaFreeHost(h_detections_num_));
    checkCudaErrors(cudaFree(d_detections_));
    checkCudaErrors(cudaFree(d_detections_reshape_)); 

    for (unsigned int i=0; i < NUM_TASKS; i++) {
        checkCudaErrors(cudaFree(d_reg_[i]));
        checkCudaErrors(cudaFree(d_height_[i]));
        checkCudaErrors(cudaFree(d_dim_[i]));
        checkCudaErrors(cudaFree(d_rot_[i]));
        checkCudaErrors(cudaFree(d_vel_[i]));
        checkCudaErrors(cudaFree(d_hm_[i]));
    }

    checkCudaErrors(cudaFreeHost(h_mask_));
    return;
}

int CenterPoint::prepare(){
    pre_->alloc_resource();
    return 0;
}

int CenterPoint::doinfer(void* points, unsigned int point_num, cudaStream_t stream)
{
    float elapsedTime = 0.0f;

    timer_.start(stream);
    pre_->generateVoxels((float *)points, point_num, stream);
    timing_pre_.push_back(timer_.stop("Voxelization", verbose_));

    unsigned int valid_num = pre_->getOutput(&d_voxel_features, &d_voxel_indices, sparse_shape);
    if (verbose_) {
        std::cout << "valid_num: " << valid_num <<std::endl;
    }

    timer_.start(stream);
    auto result = scn_engine_->forward(
        {valid_num, 5}, spconv::DType::Float16, d_voxel_features,
        {valid_num, 4}, spconv::DType::Int32,   d_voxel_indices,
        1, sparse_shape, stream
    );
    timing_scn_engine_.push_back(timer_.stop("3D Backbone", verbose_));

    timer_.start(stream);
    trt_->forward({result->features_data(), d_reg_[0], d_height_[0], d_dim_[0], d_rot_[0], d_vel_[0], d_hm_[0],
                                                d_reg_[1], d_height_[1], d_dim_[1], d_rot_[1], d_vel_[1], d_hm_[1],
                                                d_reg_[2], d_height_[2], d_dim_[2], d_rot_[2], d_vel_[2], d_hm_[2],
                                                d_reg_[3], d_height_[3], d_dim_[3], d_rot_[3], d_vel_[3], d_hm_[3],
                                                d_reg_[4], d_height_[4], d_dim_[4], d_rot_[4], d_vel_[4], d_hm_[4],
                                                d_reg_[5], d_height_[5], d_dim_[5], d_rot_[5], d_vel_[5], d_hm_[5]}, stream);
    timing_trt_.push_back(timer_.stop("RPN + Head", verbose_));
    nms_pred_.clear();

    timer_.start(stream);
    for(unsigned int i_task =0; i_task < NUM_TASKS; i_task++) {
        checkCudaErrors(cudaMemset(h_detections_num_, 0, sizeof(unsigned int)));
        checkCudaErrors(cudaMemset(d_detections_, 0, MAX_DET_NUM * DET_CHANNEL * sizeof(float)));
        checkCudaErrors(cudaMemset(d_detections_reshape_, 0, MAX_DET_NUM * DET_CHANNEL * sizeof(float)));

        post_->doPostDecodeCuda(reg_n_, reg_h_, reg_w_, reg_c_, height_c_, dim_c_, rot_c_, vel_c_, hm_c_[i_task],
                                d_reg_[i_task],
                                d_height_[i_task],
                                d_dim_[i_task],
                                d_rot_[i_task],
                                d_vel_[i_task],
                                d_hm_[i_task],
                                h_detections_num_,
                                d_detections_, stream);
        if(*h_detections_num_ == 0) continue;

        checkCudaErrors(cudaMemcpyAsync(detections_.data(), d_detections_, MAX_DET_NUM * DET_CHANNEL * sizeof(float), cudaMemcpyDeviceToHost, stream));
        checkCudaErrors(cudaStreamSynchronize(stream));

        std::sort(detections_.begin(), detections_.end(),
                [](float11 boxes1, float11 boxes2) { return boxes1.val[10] > boxes2.val[10]; });

        checkCudaErrors(cudaMemcpyAsync(d_detections_, detections_.data() , MAX_DET_NUM * DET_CHANNEL * sizeof(float), cudaMemcpyHostToDevice, stream));
        checkCudaErrors(cudaMemsetAsync(h_mask_, 0, h_mask_size_, stream));

        post_->doPermuteCuda(*h_detections_num_, d_detections_, d_detections_reshape_, stream);
        checkCudaErrors(cudaStreamSynchronize(stream));

        post_->doPostNMSCuda(*h_detections_num_, d_detections_reshape_, h_mask_, stream);
        checkCudaErrors(cudaStreamSynchronize(stream));

        int col_blocks = DIVUP(*h_detections_num_, NMS_THREADS_PER_BLOCK);
        std::vector<uint64_t> remv(col_blocks, 0);
        std::vector<bool> keep(*h_detections_num_, false);
        int max_keep_size = 0;
        for (unsigned int i_nms = 0; i_nms < *h_detections_num_; i_nms++) {
            unsigned int nblock = i_nms / NMS_THREADS_PER_BLOCK;
            unsigned int inblock = i_nms % NMS_THREADS_PER_BLOCK;

            if (!(remv[nblock] & (1ULL << inblock))) {
                keep[i_nms] = true;
                if (max_keep_size++ < params_.nms_post_max_size) {
                    nms_pred_.push_back(Bndbox(detections_[i_nms].val[0], detections_[i_nms].val[1], detections_[i_nms].val[2],
                                        detections_[i_nms].val[3], detections_[i_nms].val[4], detections_[i_nms].val[5],
                                        detections_[i_nms].val[6], detections_[i_nms].val[7], detections_[i_nms].val[8],
                                        params_.task_num_stride[i_task] + static_cast<int>(detections_[i_nms].val[9]), detections_[i_nms].val[10]));
                }
                uint64_t* p = h_mask_ + i_nms * col_blocks;
                for (int j_nms = nblock; j_nms < col_blocks; j_nms++) {
                    remv[j_nms] |= p[j_nms];
                }
            }
        }
    }
    timing_post_.push_back(timer_.stop("Decode + NMS", verbose_));
    if (verbose_) {
        std::cout << "Detection NUM: " << nms_pred_.size() << std::endl;
        // for(int loop = 0; loop<nms_pred_.size();loop++){
        //     printf("%d, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f\n", loop, nms_pred_[loop].x, nms_pred_[loop].y,nms_pred_[loop].z,nms_pred_[loop].w,nms_pred_[loop].l,nms_pred_[loop].h,nms_pred_[loop].vx,nms_pred_[loop].vy,nms_pred_[loop].rt,nms_pred_[loop].score);
        // }
    }
    return 0;
}

void CenterPoint::perf_report(){
    float a = getAverage(timing_pre_);
    float b = getAverage(timing_scn_engine_);
    float c = getAverage(timing_trt_);
    float d = getAverage(timing_post_);
    float total = a + b + c + d;
    std::cout << "\nPerf Report: "        << std::endl;
    std::cout << "    Voxelization: "   << a << " ms." <<std::endl;
    std::cout << "    3D Backbone: "    << b << " ms." << std::endl;
    std::cout << "    RPN + Head: "     << c << " ms." << std::endl;
    std::cout << "    Decode + NMS: "   << d << " ms." << std::endl;
    std::cout << "    Total: "          << total << " ms." << std::endl;
}
