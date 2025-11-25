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
 
#ifndef __SPCONV_CHECK_HPP__
#define __SPCONV_CHECK_HPP__

#include <assert.h>
#include <cuda_runtime.h>
#include <string>
#include <spconv/engine.hpp>

namespace spconv {

#define check_cuda_api(call) spconv::check_cuda_api_(call, #call, __LINE__, __FILE__)
#define check_cuda_kernel(...)                                                            \
  do {                                                                              \
    __VA_ARGS__;                                                                    \
    spconv::check_cuda_api_(cudaPeekAtLastError(), #__VA_ARGS__, __LINE__, __FILE__); \
  } while (0)

#define spconv_assertf(cond, fmt, ...)                                                                 \
  do {                                                                                          \
    if (!(cond)) {                                                                              \
      spconv::logger_output(__FILE__, __LINE__, spconv::LoggerLevel::Error, "Assert failed 💀. %s in file %s:%d, message: " fmt, #cond, __FILE__, \
              __LINE__, __VA_ARGS__);                                                           \
    }                                                                                           \
  } while (false)

#define spconv_asserts(cond, s)                                                                      \
  do {                                                                                        \
    if (!(cond)) {                                                                            \
      spconv::logger_output(__FILE__, __LINE__, spconv::LoggerLevel::Error, "Assert failed 💀. %s in file %s:%d, message: " s, #cond, __FILE__, \
              __LINE__);                                                                      \
    }                                                                                         \
  } while (false)

#define spconv_assert(cond)                                                                     \
  do {                                                                                   \
    if (!(cond)) {                                                                       \
      spconv::logger_output(__FILE__, __LINE__, spconv::LoggerLevel::Error, "Assert failed 💀. %s in file %s:%d", #cond, __FILE__, __LINE__); \
    }                                                                                    \
  } while (false)

#define spconv_logf(level, fmt, ...)  spconv::logger_output(__FILE__, __LINE__, level, fmt, __VA_ARGS__);
#define spconv_log(level, msg)        spconv::logger_output(__FILE__, __LINE__, level, "%s", msg);

Exported std::string format(const char *fmt, ...);
Exported bool check_cuda_api_(cudaError_t e, const char *call, int line, const char *file);

};  // namespace spconv

#endif  // #ifndef __SPCONV_CHECK_HPP__