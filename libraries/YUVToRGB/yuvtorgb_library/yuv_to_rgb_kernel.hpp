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
 
#ifndef YUV_TO_RGB_KERNEL_HPP
#define YUV_TO_RGB_KERNEL_HPP

enum class PixelLayout : unsigned int{
    NoneEnum   = 0,
    NCHW_RGB   = 1,
    NCHW_BGR   = 2,
    NHWC_RGB   = 3,
    NHWC_BGR   = 4,

    // for DLA, x = [4, 16, 32]
    // data layout = tensor.view(N, C//x, x, H, W).transpose(N, C//x, H, W, x)
    // shape       = N, C, H, W
    NCHW16_RGB = 5,  // c = (c + 15) / 16 * 16 if c % 16 != 0 else c
    NCHW16_BGR = 6, 
    NCHW32_RGB = 7,  // c = (c + 31) / 32 * 32 if c % 32 != 0 else c
    NCHW32_BGR = 8,
    NCHW4_RGB  = 9,  // c = (c + 3) / 4 * 4 if c % 4 != 0 else c
    NCHW4_BGR  = 10
};

enum class DataType : unsigned int{
    NoneEnum         = 0,
    Uint8            = 1,
    Float32          = 2,
    Float16          = 3,
    Int8             = 4
};

enum class Interpolation : unsigned int{
    NoneEnum = 0,
    Nearest  = 1,
    Bilinear = 2
};

enum class YUVFormat : unsigned int{
    NoneEnum          = 0,
    NV12BlockLinear   = 1,
    NV12PitchLinear   = 2,
    YUV422Packed_YUYV = 3
};

struct FillColor{unsigned char color[3];};

// If yuv_format == NV12BlockLinear, luma must be of type cudaTexture_t, otherwise luma must be ydata of type unsigned char*.
// If yuv_format == NV12BlockLinear, chroma must be of type cudaTexture_t, otherwise chroma must be uvdata of type unsigned char*.
// if out_layout  == NHWC_RGB or NHWC_BGR, out_stride are used, otherwise ignore out_stride
// The pipeline is as follows:
// - Input the RGBA/BGRA layout image pointer
// 1.Resize : use input_width to scaled_width, input_height to scaled_height
// 2.Translation : translate the image using output_xoffset and output_yoffset on the output(size = output_width, output_height)
// 3.Conversion : YUV to RGB/BGR...
void batched_convert_yuv_to_rgb(
    const void* luma, const void* chroma, int input_width, int input_stride, int input_height, int input_batch, YUVFormat yuv_format, 
    int scaled_width, int scaled_height, int output_xoffset, int output_yoffset, FillColor fillcolor, 
    void* out_ptr, int out_width, int out_stride, int out_height, 
    DataType out_dtype, PixelLayout out_layout, Interpolation interp,
    float mean0, float mean1, float mean2, float scale0, float scale1, float scale2,
    void* stream
);

#endif // YUV_TO_RGB_KERNEL_HPP