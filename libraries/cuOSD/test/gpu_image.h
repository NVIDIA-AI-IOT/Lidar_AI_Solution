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
 
#ifndef GPU_IMAGE_H
#define GPU_IMAGE_H

namespace gpu{

    enum class ImageFormat : int {
        None = 0,
        RGB  = 1,
        RGBA = 2,
        BlockLinearNV12 = 3,
        PitchLinearNV12 = 4
    };

    struct Image{
        void* data0    = nullptr;
        void* data1    = nullptr;
        void *reserve0 = nullptr;
        void *reserve1 = nullptr;
        int width      = 0;
        int height     = 0;
        int stride  = 0;
        ImageFormat format = ImageFormat::None;
    };

    struct Segment{
        float* data     = nullptr;
        int width       = 0;
        int height      = 0;
    };

    struct Point
    {
        int x           = 0;
        int y           = 0;
    };

    struct Polyline
    {
        int* h_pts      = nullptr;
        int* d_pts      = nullptr;
        int n_pts       = 0;
    };

    // Get name of enumerate type
    const char* image_format_name(ImageFormat format);

    // Create gpu image using size and format
    Image* create_image(int width, int height, ImageFormat format);

    // Create segment with fixed size 10 x 10
    Segment* create_segment();

    // Create polyline for test
    Polyline* create_polyline();

    // Set image color
    void set_color(Image* image, unsigned char r, unsigned char g, unsigned char b, unsigned char a=255, void* stream=nullptr);

    // mask specific r-g-b pixel to transparent, and apply uniform alpha for none-transparent pixels
    void mask_rgba_alpha(Image* image, unsigned char r, unsigned char g, unsigned char b, unsigned char a, void* _stream);

    // Copy yuv to image
    void copy_yuvnv12_to(Image* image, int dst_x, int dst_y, int dst_w, int dst_h, const char* yuvnv12file, int yuvwidth, int yuvheight, unsigned char yuvalpha=255, void* stream=nullptr);

    // Save image to file, file format is png if rgba, otherwise jpg
    bool save_image(Image* image, const char* file, void* stream=nullptr);

    // Free image pointer
    void free_image(Image* image);

    // Free segment pointer
    void free_segment(Segment* segment);

    // Free polyline pointer
    void free_polyline(Polyline* segment);
};

#endif // GPU_IMAGE_H