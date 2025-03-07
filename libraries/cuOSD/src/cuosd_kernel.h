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
 
#ifndef CUOSD_KERNEL_H
#define CUOSD_KERNEL_H

enum class ImageFormat : int {
    None         = 0,
    RGB          = 1,
    RGBA         = 2,
    BlockLinearNV12 = 3,
    PitchLinearNV12 = 4
};

enum class CommandType : int {
    None         = 0,
    Circle       = 1,
    Rectangle    = 2,
    Text         = 3,
    Segment      = 4,
    PolyFill     = 5,
    RGBASource   = 6,
    NV12Source   = 7,
    BoxBlur      = 8,
    Ellipse      = 9
};

struct TextLocation{
    int image_x, image_y;
    int text_x;
    int text_w, text_h;
};

// cuOSDContextCommand includes basic attributes for color and bounding box coordinate
struct cuOSDContextCommand{
    CommandType type = CommandType::None;
    unsigned char c0, c1, c2, c3;
    int bounding_left   = 0;
    int bounding_top    = 0;
    int bounding_right  = 0;
    int bounding_bottom = 0;
};

// CircleCommand:
// cx, cy: center point coordinate of the circle
// thickness: border width in case > 0, -1 stands for fill mode
struct CircleCommand : cuOSDContextCommand{
    int cx, cy, radius, thickness;

    CircleCommand(int cx, int cy, int radius, int thickness, unsigned char c0, unsigned char c1, unsigned char c2, unsigned char c3);
};

// EllipseCommand:
// cx, cy: center point coordinate of the ellipse
// width, height: axis length of the ellipse
// thickness: border width in case > 0, -1 stands for fill mode
struct EllipseCommand : cuOSDContextCommand{
    int cx, cy, width, height, radius, thickness;
    float afactor, bfactor, cfactor, yaw;
    EllipseCommand(int cx, int cy, int width, int height, float yaw, int thickness, unsigned char c0, unsigned char c1, unsigned char c2, unsigned char c3);
};

// RectangleCommand:
// ax1, ..., dy1: 4 outer corner points coordinate of the rectangle
// ax2, ..., dy2: 4 inner corner points coordinate of the rectangle
//    a1 ------ d1
//    | a2---d2 |
//    | |     | |
//    | b2---c2 |
//    b1 ------ c1
// thickness: border width in case > 0, -1 stands for fill mode
struct RectangleCommand : cuOSDContextCommand{
    int thickness = -1;
    bool interpolation = false;
    float ax1, ay1, bx1, by1, cx1, cy1, dx1, dy1;
    float ax2, ay2, bx2, by2, cx2, cy2, dx2, dy2;

    RectangleCommand();
};

struct BoxBlurCommand : cuOSDContextCommand{
    int kernel_size = 7;

    BoxBlurCommand();
};

// TextCommand:
// text_line_size && ilocation are inner attribute for text memory management
struct TextCommand : cuOSDContextCommand{
    int text_line_size;
    int ilocation;

    TextCommand() = default;
    TextCommand(int text_line_size, int ilocation, unsigned char c0, unsigned char c1, unsigned char c2, unsigned char c3);
};

// SegmentCommand:
// scale_x: seg mask w / outer rect w
// scale_y: seg mask h / outer rect h
struct SegmentCommand : cuOSDContextCommand{
    float* d_seg;
    int seg_width, seg_height;
    float scale_x, scale_y;
    float seg_threshold;

    SegmentCommand();
};

// PolyFillCommand:
struct PolyFillCommand : cuOSDContextCommand{
    int* d_pts;
    int n_pts;

    PolyFillCommand();
};

// RGBASourceCommand:
// d_src: device pointer for incoming rgba source image
struct RGBASourceCommand : cuOSDContextCommand{
    void* d_src;
    int src_width, src_stride, src_height;
    float scale_x, scale_y;

    RGBASourceCommand();
};

// NV12SourceCommand:
// d_src0: device pointer for Y plane of incoming nv12 source image
// d_src1: device pointer for UV plane of incoming nv12 source image
// block_linear: input device pointers are block linear or pitch linear
struct NV12SourceCommand : cuOSDContextCommand{
    void* d_src0;
    void* d_src1;
    int src_width, src_stride, src_height;
    float scale_x, scale_y;
    bool block_linear;

    NV12SourceCommand();
};

void cuosd_launch_kernel(
    void* image_data0, void* image_data1, int width, int stride, int height, ImageFormat format,
    const TextLocation* text_location, const unsigned char* text_bitmap, int text_bitmap_width, const int* line_location_base,
    const unsigned char* commands, const int* commands_offset, int num_commands,
    int bounding_left, int bounding_top, int bounding_right, int bounding_bottom,
    bool have_rotatebox, const unsigned char* blur_commands, int num_blur_commands,
    void* stream
);

#endif // CUOSD_KERNEL_H