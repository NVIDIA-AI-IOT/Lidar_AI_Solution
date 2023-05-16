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
 
#ifndef YUV_TO_RGB_HPP
#define YUV_TO_RGB_HPP

#include <yuv_to_rgb_kernel.hpp>
#include <cuda_runtime.h>

// YUV YUV PL
struct YUVHostImage{
    uint8_t* data = nullptr;
    unsigned int width  = 0, height = 0;  
    unsigned int y_area = 0;
    unsigned int stride = 0;
    YUVFormat format = YUVFormat::NoneEnum;
};

struct YUVGPUImage{
    void* luma   = 0;     // y
    void* chroma = 0;     // uv
    void* luma_array   = nullptr;        //  nullptr if format == PL
    void* chroma_array = nullptr;        //  nullptr if format == PL
    unsigned int width = 0, height = 0;
    unsigned int stride = 0;
    unsigned int batch = 0;
    YUVFormat format  = YUVFormat::NoneEnum;
};

struct RGBHostImage{
    uint8_t* data = nullptr;
    unsigned int width = 0, height = 0;
};

struct RGBGPUImage{
    void* data  = nullptr;
    int width   = 0, height = 0;
    int batch   = 0;
    int channel = 0;
    int stride  = 0;
    PixelLayout layout = PixelLayout::NoneEnum;
    DataType dtype     = DataType::NoneEnum;
};

const char* pixel_layout_name(PixelLayout layout);
const char* yuvformat_name(YUVFormat format);
const char* interp_name(Interpolation interp);
const char* dtype_name(DataType dtype);
size_t      dtype_sizeof(DataType dtype);

YUVGPUImage* create_yuv_gpu_image(int width, int height, int batch_size, YUVFormat format);
void free_yuv_gpu_image(YUVGPUImage* p);

void copy_yuv_host_to_gpu(const YUVHostImage* yuv, YUVGPUImage* gpu, unsigned int ibatch, unsigned int crop_width, unsigned int crop_height, cudaStream_t stream = nullptr);

RGBGPUImage* create_rgb_gpu_image(int width, int height, int batch, PixelLayout layout, DataType dtype);
void free_rgb_gpu_image(RGBGPUImage* p);
bool save_rgbgpu_to_file(const std::string& file, RGBGPUImage* gpu, cudaStream_t stream=nullptr);

YUVHostImage* read_yuv(const std::string& file, int width, int height, YUVFormat format);
void free_yuv_host_image(YUVHostImage* p);

void batched_convert_yuv_to_rgb(
    YUVGPUImage* input, RGBGPUImage* output, 
    int scaled_width, int scaled_height,
    int output_xoffset, int output_yoffset, FillColor fillcolor,
    float mean0 = 0,  float mean1 = 0,  float mean2 = 0, 
    float scale0 = 1, float scale1 = 1, float scale2 = 1,
    Interpolation interp = Interpolation::Nearest,
    cudaStream_t stream = nullptr
);

#endif // YUV_TO_RGB_HPP