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
 
#ifndef __SPCONV_VERSION_HPP__
#define __SPCONV_VERSION_HPP__

#define NVSPCONV_MAJOR 1
#define NVSPCONV_MINOR 3
#define NVSPCONV_REL 2
#define NVSPCONV_STR(v) #v
#define NVSPCONV_VERSION_COMBIN(major, minor, rel) \
  (NVSPCONV_STR(major) "." NVSPCONV_STR(minor) "." NVSPCONV_STR(rel))
#define NVSPCONV_VERSION NVSPCONV_VERSION_COMBIN(NVSPCONV_MAJOR, NVSPCONV_MINOR, NVSPCONV_REL)

#endif  // #ifndef __SPCONV_VERSION_HPP__
