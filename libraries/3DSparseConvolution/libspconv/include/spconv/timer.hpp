/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */
 
#ifndef __SPCONV_TIMER_HPP__
#define __SPCONV_TIMER_HPP__

#include <cuda_runtime.h>

#include "check.hpp"

namespace spconv {

class EventTimer {
 public:
  EventTimer() {
    check_cuda_api(cudaEventCreate(&begin_));
    check_cuda_api(cudaEventCreate(&end_));
  }

  virtual ~EventTimer() {
    check_cuda_api(cudaEventDestroy(begin_));
    check_cuda_api(cudaEventDestroy(end_));
  }

  void start(void *stream) {
    stream_ = (cudaStream_t)stream;
    check_cuda_api(cudaEventRecord(begin_, (cudaStream_t)stream));
  }

  float stop(const char *prefix, bool print = true) {
    float times = 0;
    check_cuda_api(cudaEventRecord(end_, stream_));
    check_cuda_api(cudaEventSynchronize(end_));
    check_cuda_api(cudaEventElapsedTime(&times, begin_, end_));
    if (print) printf("[Times %s]: %.3f ms\n", prefix, times);
    return times;
  }

 private:
  cudaStream_t stream_ = nullptr;
  cudaEvent_t begin_ = nullptr, end_ = nullptr;
};

};  // namespace spconv

#endif  // #ifndef __SPCONV_TIMER_HPP__