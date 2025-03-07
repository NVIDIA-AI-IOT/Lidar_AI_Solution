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
 
#include <iostream>
#include <iomanip>
#include <ctime>
#include <vector>
#include <memory>
#include <string>
#include <fstream>
#include <sstream>
#include <map>
#include <cuda_runtime.h>
#include <thread>
#include <math.h>
#include <stack>
#include <algorithm>

#include <string.h>
#include "cuosd_kernel.h"
#include "cuosd.h"

#include "memory.hpp"
#include "textbackend/backend.hpp"

using namespace std;

struct TextHostCommand : cuOSDContextCommand{
    TextCommand gputile;
    vector<unsigned long int> text;
    unsigned short font_size;
    string font_name;
    int x, y;

    TextHostCommand(const vector<unsigned long int>& text, unsigned short font_size, const char* font, int x, int y, unsigned char c0, unsigned char c1, unsigned char c2, unsigned char c3) {
        this->text = text;
        this->font_size = font_size;
        this->font_name = font;
        this->x = x;
        this->y = y;
        this->c0 = c0;
        this->c1 = c1;
        this->c2 = c2;
        this->c3 = c3;
        this->type = CommandType::Text;
    }
};

struct cuOSDContextImpl : public cuOSDContext{
    unique_ptr<Memory<TextLocation>> text_location;
    unique_ptr<Memory<int>>          line_location_base;
    vector<shared_ptr<cuOSDContextCommand>> commands;
    vector<shared_ptr<cuOSDContextCommand>> blur_commands;
    unique_ptr<Memory<unsigned char>>  gpu_commands;            // TextCommand, RectangleCommand etc.
    unique_ptr<Memory<BoxBlurCommand>>  gpu_blur_commands;            // TextCommand, RectangleCommand etc.
    unique_ptr<Memory<int>>      gpu_commands_offset;           // sizeof(TextCommand), sizeof(TextCommand) + sizeof(RectangleCommand) etc.
    shared_ptr<TextBackend> text_backend;

    #ifdef ENABLE_TEXT_BACKEND_PANGO
    cuOSDTextBackend text_backend_type = cuOSDTextBackend::PangoCairo;
    #else
    cuOSDTextBackend text_backend_type = cuOSDTextBackend::StbTrueType;
    #endif

    bool have_rotate_msaa = false;
    int bounding_left   = 0;
    int bounding_top    = 0;
    int bounding_right  = 0;
    int bounding_bottom = 0;
};

template<typename _T>
static inline unsigned char u8cast(_T value) {
    return value < 0 ? 0 : (value > 255 ? 255 : value);
}

static unsigned int inline round_down2(unsigned int num) {
    return num & (~1);
}

static TextBackendType convert_to_text_backend_type(cuOSDTextBackend backend){

    switch(backend){
        case cuOSDTextBackend::PangoCairo: return TextBackendType::PangoCairo;
        case cuOSDTextBackend::StbTrueType: return TextBackendType::StbTrueType;
        default: return TextBackendType::None;
    }
}

static inline tuple<unsigned char, unsigned char, unsigned char> make_u83(unsigned char a, unsigned char b, unsigned char c) {
    return tuple<unsigned char, unsigned char, unsigned char>(a, b, c);
}

cuOSDContext_t cuosd_context_create() {
    return new cuOSDContextImpl();
}

void cuosd_set_text_backend(cuOSDContext_t context, cuOSDTextBackend text_backend){
    static_cast<cuOSDContextImpl*>(context)->text_backend_type = text_backend;
}

void cuosd_context_destroy(cuOSDContext_t context) {
    if (context) {
        cuOSDContextImpl* p = (cuOSDContextImpl*)context;
        delete p;
    }
}

void cuosd_measure_text(cuOSDContext_t _context, const char* utf8_text, int font_size, const char* font, int* width, int* height, int* yoffset) {
    cuOSDContextImpl* context = (cuOSDContextImpl*)_context;
    if (context->text_backend == nullptr)
        context->text_backend = create_text_backend(convert_to_text_backend_type(context->text_backend_type));
    
    if(context->text_backend == nullptr){
        CUOSD_PRINT_E("There are no valid backend, please make sure your settings\n");
        return;
    }

    auto words = context->text_backend->split_utf8(utf8_text);
    if (words.empty() && strlen(utf8_text) > 0) {
        CUOSD_PRINT_E("There are some errors during converting UTF8 to Unicode.\n");
        return;
    }
    if (words.empty() || font_size <= 0) return;

    font_size = std::max(10, std::min(MAX_FONT_SIZE, font_size));
    tie(*width, *height, *yoffset) = context->text_backend->measure_text(words, font_size, font);
}

void cuosd_draw_text(
    cuOSDContext_t _context,
    const char* utf8_text,
    int font_size, const char* font, int x, int y,
    cuOSDColor border_color, cuOSDColor bg_color
) {
    cuOSDContextImpl* context = (cuOSDContextImpl*)_context;
    if (context->text_backend == nullptr)
        context->text_backend = create_text_backend(convert_to_text_backend_type(context->text_backend_type));

    if(context->text_backend == nullptr){
        CUOSD_PRINT_E("There are no valid backend, please make sure your settings\n");
        return;
    }

    auto words = context->text_backend->split_utf8(utf8_text);
    if (words.empty() && strlen(utf8_text) > 0) {
        CUOSD_PRINT_E("There are some errors during converting UTF8 to Unicode.\n");
        return;
    }
    if (words.empty() || font_size <= 0) return;

    // Scale to 3x, in order to align with nvOSD effect
    font_size = context->text_backend->uniform_font_size(font_size);
    font_size = std::max(10, std::min(MAX_FONT_SIZE, font_size));

    std::vector<shared_ptr<TextHostCommand>> commands;
    int xmargin = font_size * 0.5;
    int ymargin = font_size * 0.25;
    int background_min_x = x;
    int background_min_y = y;
    int background_max_x = x;
    int background_max_y = y;
    bool need_background_fill = bg_color.a != 0;

    auto precompute_line = [&](const std::vector<unsigned long>& line_words, const int ypos)->int{
        int width, height, yoffset;
        tie(width, height, yoffset) = context->text_backend->measure_text(line_words, font_size, font);

        if (need_background_fill) {
            background_max_x = std::max(x + width + 2 * xmargin - 1,     background_max_x);
            background_max_y = std::max(ypos + height + 2 * ymargin - 1, background_max_y);
        }
        commands.emplace_back(make_shared<TextHostCommand>(line_words, font_size, font, x + xmargin, ypos + ymargin - yoffset, border_color.r, border_color.g, border_color.b, border_color.a));
        return height;
    };

    bool need_split_lines = false;
    for(size_t i = 0; i < words.size(); ++i){

        // ignore the last charater \n
        if((words[i] & 0xFF) == '\n'){
            need_split_lines = true;
        }
    }
    
    if(!need_split_lines){
        precompute_line(words, y);

        // add rectangle cmd as background color if need to fill the background.
        if(need_background_fill){
            cuosd_draw_rectangle(_context, background_min_x, background_min_y, background_max_x, background_max_y, -1, bg_color);
        }
        context->commands.insert(context->commands.end(), commands.begin(), commands.end());
        return;
    }

    // below needed to compute the special character \n height.
    int line_text_y_pos = y;
    size_t prev_break_pos = 0;
    size_t icurrent_pos = 0;

    int width, height, yoffset;
    tie(width, height, yoffset) = context->text_backend->measure_text(context->text_backend->split_utf8("a"), font_size, font);
    int prev_break_line_height = height;
    for(; icurrent_pos < words.size(); ++icurrent_pos){

        // ignore the last charater \n
        if((words[icurrent_pos] & 0xFF) == '\n'){
            if(icurrent_pos == prev_break_pos){
                line_text_y_pos += prev_break_line_height;
            }else{
                std::vector<unsigned long> line(words.begin() + prev_break_pos, words.begin() + icurrent_pos);
                int line_height = precompute_line(line, line_text_y_pos);
                prev_break_line_height = line_height;
                line_text_y_pos += line_height;
            }
            prev_break_pos = icurrent_pos + 1;
        }
    }

    if(icurrent_pos > prev_break_pos){
        std::vector<unsigned long> line(words.begin() + prev_break_pos, words.begin() + icurrent_pos);
        precompute_line(line, line_text_y_pos);
    }

    if(!commands.empty()){
        if(need_background_fill){
            cuosd_draw_rectangle(_context, background_min_x, background_min_y, background_max_x, background_max_y, -1, bg_color);
        }
        context->commands.insert(context->commands.end(), commands.begin(), commands.end());
    }
}

void cuosd_draw_clock(
    cuOSDContext_t _context,
    cuOSDClockFormat clock_format, long time,
    int font_size, const char* font, int x, int y,
    cuOSDColor border_color, cuOSDColor bg_color
) {
    std::chrono::time_point<std::chrono::system_clock> time_now = std::chrono::system_clock::now();
    std::time_t time_now_t = std::chrono::system_clock::to_time_t(time_now);
    if (time != 0) time_now_t = time;

    std::tm now_tm = *std::localtime(&time_now_t);
    std::ostringstream oss;
    if (clock_format == cuOSDClockFormat::HHMMSS) {
        oss << std::put_time(&now_tm, "%H:%M:%S");
    } else if (clock_format == cuOSDClockFormat::YYMMDD) {
        oss << std::put_time(&now_tm, "%Y-%m-%d");
    } else if (clock_format == cuOSDClockFormat::YYMMDD_HHMMSS) {
        oss << std::put_time(&now_tm, "%Y-%m-%d %H:%M:%S");
    }
    auto utf8_str = oss.str();
    cuosd_draw_text(_context, utf8_str.c_str(), font_size, font, x, y, border_color, bg_color);
}

void cuosd_draw_rectangle(
    cuOSDContext_t _context, int left, int top, int right, int bottom, int thickness, cuOSDColor border_color, cuOSDColor bg_color
) {
    cuOSDContextImpl* context = (cuOSDContextImpl*)_context;
    int tl = min(left, right);
    int tt = min(top, bottom);
    int tr = max(left, right);
    int tb = max(top, bottom);
    left = tl; top = tt; right = tr; bottom = tb;

    if (border_color.a == 0) return;
    if (bg_color.a || thickness == -1) {
        if (thickness == -1) {
            bg_color = border_color;
        }

        auto cmd = make_shared<RectangleCommand>();
        cmd->thickness = -1;
        cmd->interpolation = false;
        cmd->c0 = bg_color.r; cmd->c1 = bg_color.g; cmd->c2 = bg_color.b; cmd->c3 = bg_color.a;

        // a   d
        // b   c
        cmd->ax1 = left; cmd->ay1 = top; cmd->dx1 = right; cmd->dy1 = top; cmd->cx1 = right; cmd->cy1 = bottom; cmd->bx1 = left; cmd->by1 = bottom;
        cmd->bounding_left  = left; cmd->bounding_right = right; cmd->bounding_top   = top; cmd->bounding_bottom = bottom;
        context->commands.emplace_back(cmd);
    }
    if (thickness == -1) return;

    auto cmd = make_shared<RectangleCommand>();
    cmd->thickness = thickness;
    cmd->interpolation = false;
    cmd->c0 = border_color.r; cmd->c1 = border_color.g; cmd->c2 = border_color.b; cmd->c3 = border_color.a;

    float half_thickness = thickness / 2.0f;
    cmd->ax2 = left + half_thickness;
    cmd->ay2 = top  + half_thickness;
    cmd->dx2 = right - half_thickness;
    cmd->dy2 = top + half_thickness;
    cmd->cx2 = right - half_thickness;
    cmd->cy2 = bottom - half_thickness;
    cmd->bx2 = left + half_thickness;
    cmd->by2 = bottom - half_thickness;

    // a   d
    // b   c
    cmd->ax1 = left - half_thickness;
    cmd->ay1 = top  - half_thickness;
    cmd->dx1 = right + half_thickness;
    cmd->dy1 = top - half_thickness;
    cmd->cx1 = right + half_thickness;
    cmd->cy1 = bottom + half_thickness;
    cmd->bx1 = left - half_thickness;
    cmd->by1 = bottom + half_thickness;

    int int_half = ceil(half_thickness);
    cmd->bounding_left  = left - int_half;
    cmd->bounding_right = right + int_half;
    cmd->bounding_top   = top - int_half;
    cmd->bounding_bottom = bottom + int_half;
    context->commands.emplace_back(cmd);
}

void cuosd_draw_boxblur(
    cuOSDContext_t _context, int left, int top, int right, int bottom, int kernel_size){
        
    cuOSDContextImpl* context = (cuOSDContextImpl*)_context;
    int tl = min(left, right);
    int tt = min(top, bottom);
    int tr = max(left, right);
    int tb = max(top, bottom);
    left = tl; top = tt; right = tr; bottom = tb;

    if(right - left < 3 || bottom - top < 3 || kernel_size < 1){
        CUOSD_PRINT_E("[cuosd_draw_boxblur] This operation will be ignored because the region of interest is too small, or the kernel is too small.");
        return;
    }
    
    auto cmd = make_shared<BoxBlurCommand>();
    cmd->kernel_size = kernel_size;
    cmd->bounding_left    = left;
    cmd->bounding_right   = right;
    cmd->bounding_top     = top;
    cmd->bounding_bottom  = bottom;
    context->blur_commands.emplace_back(cmd);
}

void cuosd_draw_rotationbox(
    cuOSDContext_t _context, int cx, int cy, int width, int height, float yaw, int thickness, cuOSDColor border_color, bool interpolation, cuOSDColor bg_color
) {
    cuOSDContextImpl* context = (cuOSDContextImpl*)_context;
    if (border_color.a == 0) return;

    // a   d
    //   o
    // b   c

    float pax = -width / 2.0f;
    float pay = -height / 2.0f;
    float pcx = +width / 2.0f;
    float pcy = +height / 2.0f;
    float pbx = pax;
    float pby = pcy;
    float pdx = pcx;
    float pdy = pay;

    float cos_angle = std::cos(yaw);
    float sin_angle = std::sin(yaw);
    float half_thickness = thickness / 2.0f;

    int angle = (yaw / M_PI * 180) + 0.5f;
    interpolation = interpolation && angle % 90 != 0;

    if (interpolation)
        context->have_rotate_msaa = true;

    if (bg_color.a || thickness == -1) {
        if (thickness == -1) {
            bg_color = border_color;
        }

        auto cmd = make_shared<RectangleCommand>();
        cmd->thickness = -1;
        cmd->interpolation = interpolation;
        cmd->c0 = bg_color.r; cmd->c1 = bg_color.g; cmd->c2 = bg_color.b; cmd->c3 = bg_color.a;

        // a   d
        // b   c
        // cosa, -sina, ox;
        // sina, cosa,  oy;
        cmd->ax1 = cos_angle * pax - sin_angle * pay + cx;
        cmd->ay1 = sin_angle * pax + cos_angle * pay + cy;
        cmd->bx1 = cos_angle * pbx - sin_angle * pby + cx;
        cmd->by1 = sin_angle * pbx + cos_angle * pby + cy;
        cmd->cx1 = cos_angle * pcx - sin_angle * pcy + cx;
        cmd->cy1 = sin_angle * pcx + cos_angle * pcy + cy;
        cmd->dx1 = cos_angle * pdx - sin_angle * pdy + cx;
        cmd->dy1 = sin_angle * pdx + cos_angle * pdy + cy;

        cmd->bounding_left   = min(min(min(cmd->ax1, cmd->bx1), cmd->cx1), cmd->dx1);
        cmd->bounding_right  = ceil(max(max(max(cmd->ax1, cmd->bx1), cmd->cx1), cmd->dx1));
        cmd->bounding_top    = min(min(min(cmd->ay1, cmd->by1), cmd->cy1), cmd->dy1);
        cmd->bounding_bottom = ceil(max(max(max(cmd->ay1, cmd->by1), cmd->cy1), cmd->dy1));
        context->commands.emplace_back(cmd);
    }
    if (thickness == -1) return;

    auto cmd = make_shared<RectangleCommand>();
    cmd->thickness = thickness;
    cmd->c0 = border_color.r; cmd->c1 = border_color.g; cmd->c2 = border_color.b; cmd->c3 = border_color.a;

    cmd->interpolation = interpolation;
    cmd->ax1 = cos_angle * (pax - half_thickness) - sin_angle * (pay - half_thickness) + cx;
    cmd->ay1 = sin_angle * (pax - half_thickness) + cos_angle * (pay - half_thickness) + cy;
    cmd->bx1 = cos_angle * (pbx - half_thickness) - sin_angle * (pby + half_thickness) + cx;
    cmd->by1 = sin_angle * (pbx - half_thickness) + cos_angle * (pby + half_thickness) + cy;
    cmd->cx1 = cos_angle * (pcx + half_thickness) - sin_angle * (pcy + half_thickness) + cx;
    cmd->cy1 = sin_angle * (pcx + half_thickness) + cos_angle * (pcy + half_thickness) + cy;
    cmd->dx1 = cos_angle * (pdx + half_thickness) - sin_angle * (pdy - half_thickness) + cx;
    cmd->dy1 = sin_angle * (pdx + half_thickness) + cos_angle * (pdy - half_thickness) + cy;

    cmd->ax2 = cos_angle * (pax + half_thickness) - sin_angle * (pay + half_thickness) + cx;
    cmd->ay2 = sin_angle * (pax + half_thickness) + cos_angle * (pay + half_thickness) + cy;
    cmd->bx2 = cos_angle * (pbx + half_thickness) - sin_angle * (pby - half_thickness) + cx;
    cmd->by2 = sin_angle * (pbx + half_thickness) + cos_angle * (pby - half_thickness) + cy;
    cmd->cx2 = cos_angle * (pcx - half_thickness) - sin_angle * (pcy - half_thickness) + cx;
    cmd->cy2 = sin_angle * (pcx - half_thickness) + cos_angle * (pcy - half_thickness) + cy;
    cmd->dx2 = cos_angle * (pdx - half_thickness) - sin_angle * (pdy + half_thickness) + cx;
    cmd->dy2 = sin_angle * (pdx - half_thickness) + cos_angle * (pdy + half_thickness) + cy;

    cmd->bounding_left   = min(min(min(cmd->ax1, cmd->bx1), cmd->cx1), cmd->dx1);
    cmd->bounding_right  = ceil(max(max(max(cmd->ax1, cmd->bx1), cmd->cx1), cmd->dx1));
    cmd->bounding_top    = min(min(min(cmd->ay1, cmd->by1), cmd->cy1), cmd->dy1);
    cmd->bounding_bottom = ceil(max(max(max(cmd->ay1, cmd->by1), cmd->cy1), cmd->dy1));
    context->commands.emplace_back(cmd);
}

void cuosd_draw_ellipse(
    cuOSDContext_t _context, int cx, int cy, int width, int height, float yaw, int thickness, cuOSDColor border_color, cuOSDColor bg_color
) {
    cuOSDContextImpl* context = (cuOSDContextImpl*)_context;
    if (border_color.a == 0) return;
    if (bg_color.a && thickness > 0) context->commands.emplace_back(make_shared<EllipseCommand>(cx, cy, width, height, yaw, -thickness, bg_color.r, bg_color.g, bg_color.b, bg_color.a));
    context->commands.emplace_back(make_shared<EllipseCommand>(cx, cy, width, height, yaw, thickness, border_color.r, border_color.g, border_color.b, border_color.a));
}

void cuosd_draw_polyline(
    cuOSDContext_t _context, int* h_pts, int* d_pts, int n_pts, int thickness, bool is_closed, cuOSDColor border_color, bool interpolation, cuOSDColor fill_color
) {
    cuOSDContextImpl* context = (cuOSDContextImpl*)_context;

    if (n_pts < 2) return;

    int bleft   = h_pts[0];
    int bright  = h_pts[0];
    int btop    = h_pts[1];
    int bbottom = h_pts[1];
    int nline   = 0;

    for (int i = 1; i < n_pts; i++) {
        int x = h_pts[2 * i];
        int y = h_pts[2 * i + 1];

        bleft   = min(x, bleft);
        bright  = max(x, bright);
        btop    = min(y, btop);
        bbottom = max(y, bbottom);
        cuosd_draw_line(context, h_pts[2 * i - 2], h_pts[2 * i - 1], x, y, thickness, border_color, interpolation); nline++;
    }

    if (n_pts > 2) {
        if (is_closed) {
            cuosd_draw_line(context, h_pts[0], h_pts[1], h_pts[2 * n_pts - 2], h_pts[2 * n_pts - 1], thickness, border_color, interpolation); nline++;
        }

        // Fill poly if alpha is not 0 and point num > 2
        if (fill_color.a) {
            auto cmd = make_shared<PolyFillCommand>();
            cmd->d_pts = d_pts;
            cmd->n_pts = n_pts;
            cmd->bounding_left   = bleft;
            cmd->bounding_right  = bright;
            cmd->bounding_top    = btop;
            cmd->bounding_bottom = bbottom;
            cmd->c0 = fill_color.r; cmd->c1 = fill_color.g; cmd->c2 = fill_color.b; cmd->c3 = fill_color.a;
            context->commands.insert(context->commands.end() - nline, cmd);
        }
    }
}

void cuosd_draw_segmentmask(
    cuOSDContext_t _context, int left, int top, int right, int bottom, int thickness, float* d_seg, int seg_width, int seg_height, float seg_threshold, cuOSDColor border_color, cuOSDColor seg_color
) {
    cuOSDContextImpl* context = (cuOSDContextImpl*)_context;
    int tl = min(left, right);
    int tt = min(top, bottom);
    int tr = max(left, right);
    int tb = max(top, bottom);
    left = tl; top = tt; right = tr; bottom = tb;

    float half_thickness = thickness / 2.0f;
    int int_half = ceil(half_thickness);

    if (border_color.a && thickness > 0) {
        auto cmd = make_shared<RectangleCommand>();
        cmd->thickness = thickness;
        cmd->interpolation = false;
        cmd->c0 = border_color.r; cmd->c1 = border_color.g; cmd->c2 = border_color.b; cmd->c3 = border_color.a;

        cmd->ax2 = left + half_thickness;
        cmd->ay2 = top  + half_thickness;
        cmd->dx2 = right - half_thickness;
        cmd->dy2 = top + half_thickness;
        cmd->cx2 = right - half_thickness;
        cmd->cy2 = bottom - half_thickness;
        cmd->bx2 = left + half_thickness;
        cmd->by2 = bottom - half_thickness;

        // a   d
        // b   c
        cmd->ax1 = left - half_thickness;
        cmd->ay1 = top  - half_thickness;
        cmd->dx1 = right + half_thickness;
        cmd->dy1 = top - half_thickness;
        cmd->cx1 = right + half_thickness;
        cmd->cy1 = bottom + half_thickness;
        cmd->bx1 = left - half_thickness;
        cmd->by1 = bottom + half_thickness;

        cmd->bounding_left  = left - int_half;
        cmd->bounding_right = right + int_half;
        cmd->bounding_top   = top - int_half;
        cmd->bounding_bottom = bottom + int_half;
        context->commands.emplace_back(cmd);
    }

    auto cmd = make_shared<SegmentCommand>();
    auto width = right - left;
    auto height = bottom - top;

    cmd->d_seg = d_seg;
    cmd->seg_width = seg_width;
    cmd->seg_height = seg_height;

    cmd->scale_x = seg_width / (width + 1e-5);
    cmd->scale_y = seg_height / (height + 1e-5);
    cmd->seg_threshold = seg_threshold;

    cmd->c0 = seg_color.r; cmd->c1 = seg_color.g; cmd->c2 = seg_color.b; cmd->c3 = seg_color.a;

    cmd->bounding_left  = left - int_half + 1;
    cmd->bounding_right = right + int_half;
    cmd->bounding_top   = top - int_half + 1;
    cmd->bounding_bottom = bottom + int_half;

    context->commands.emplace_back(cmd);
}

void cuosd_draw_line(
    cuOSDContext_t _context, int x0, int y0, int x1, int y1, int thickness, cuOSDColor color, bool interpolation
) {
    cuOSDContextImpl* context = (cuOSDContextImpl*)_context;

    float length = std::sqrt((float)((y1-y0) * (y1-y0) + (x1-x0) * (x1-x0)));
    float angle  = std::atan2((float)y1-y0, (float)x1-x0);
    float cos_angle = std::cos(angle);
    float sin_angle = std::sin(angle);
    float half_thickness = thickness / 2.0f;

    // upline
    // a    b
    // d    c
    auto cmd = make_shared<RectangleCommand>();
    cmd->ax1 = -half_thickness * cos_angle + x0 - sin_angle * half_thickness;
    cmd->ay1 = -half_thickness * sin_angle + cos_angle * half_thickness + y0;
    cmd->bx1 = (length + half_thickness) * cos_angle - sin_angle * half_thickness + x0;
    cmd->by1 = (length + half_thickness) * sin_angle + cos_angle * half_thickness + y0;
    cmd->dx1 = -half_thickness * cos_angle + x0 + sin_angle * half_thickness;
    cmd->dy1 = -half_thickness * sin_angle + y0 - cos_angle * half_thickness;
    cmd->cx1 = (length + half_thickness) * cos_angle + sin_angle * half_thickness + x0;
    cmd->cy1 = (length + half_thickness) * sin_angle - cos_angle * half_thickness + y0;

    if (x0 == x1 || y0 == y1)
        interpolation = false;

    if (interpolation)
        context->have_rotate_msaa = true;

    cmd->interpolation = interpolation;
    cmd->thickness = -1;
    cmd->c0 = color.r; cmd->c1 = color.g; cmd->c2 = color.b; cmd->c3 = color.a;

    // a   d
    // b   c
    cmd->bounding_left   = min(min(min(cmd->ax1, cmd->bx1), cmd->cx1), cmd->dx1);
    cmd->bounding_right  = ceil(max(max(max(cmd->ax1, cmd->bx1), cmd->cx1), cmd->dx1));
    cmd->bounding_top    = min(min(min(cmd->ay1, cmd->by1), cmd->cy1), cmd->dy1);
    cmd->bounding_bottom = ceil(max(max(max(cmd->ay1, cmd->by1), cmd->cy1), cmd->dy1));
    context->commands.emplace_back(cmd);
}


void cuosd_draw_arrow(
    cuOSDContext_t context, int x0, int y0, int x1, int y1, int arrow_size, int thickness, cuOSDColor color, bool interpolation
) {
    int e03_x = x0 - x1;
    int e03_y = y0 - y1;

    float e03_len = std::sqrt((float)(e03_x * e03_x) + e03_y * e03_y);
    float e03_norm_x = e03_x / e03_len;
    float e03_norm_y = e03_y / e03_len;

    float cos_theta = std::cos(3.1415926f / 12);
    float sin_theta = std::sin(3.1415926f / 12);

    //          v1
    //          *
    //        * * *
    //   v_0 *  *  * v_1
    //          *
    //          * v0
    int x_0 = x1 + int((e03_norm_x * cos_theta - e03_norm_y * sin_theta) * arrow_size);
    int y_0 = y1 + int((e03_norm_x * sin_theta + e03_norm_y * cos_theta) * arrow_size);
    int x_1 = x1 + int((e03_norm_x * cos_theta + e03_norm_y * sin_theta) * arrow_size);
    int y_1 = y1 + int((- e03_norm_x * sin_theta + e03_norm_y * cos_theta) * arrow_size);

    cuosd_draw_line(context, x0, y0, x1, y1,   thickness, color, interpolation);
    cuosd_draw_line(context, x_0, y_0, x1, y1, thickness, color, interpolation);
    cuosd_draw_line(context, x_1, y_1, x1, y1, thickness, color, interpolation);
}

void cuosd_draw_point(
    cuOSDContext_t _context, int cx, int cy, int radius, cuOSDColor color
) {
    cuosd_draw_circle(_context, cx, cy, radius, -1, color);
}

void cuosd_draw_circle(
    cuOSDContext_t _context, int cx, int cy, int radius, int thickness, cuOSDColor border_color, cuOSDColor bg_color
) {
    cuOSDContextImpl* context = (cuOSDContextImpl*)_context;
    if (bg_color.a && thickness > 0) context->commands.emplace_back(make_shared<CircleCommand>(cx, cy, radius, -thickness, bg_color.r, bg_color.g, bg_color.b, bg_color.a));
    context->commands.emplace_back(make_shared<CircleCommand>(cx, cy, radius, thickness, border_color.r, border_color.g, border_color.b, border_color.a));
}

void cuosd_draw_rgba_source(
    cuOSDContext_t _context, int left, int top, int right, int bottom, void* d_src, int src_width, int src_stride, int src_height) {
    cuOSDContextImpl* context = (cuOSDContextImpl*)_context;
    int tl = min(left, right);
    int tt = min(top, bottom);
    int tr = max(left, right);
    int tb = max(top, bottom);
    left = tl; top = tt; right = tr; bottom = tb;

    auto cmd = make_shared<RGBASourceCommand>();
    auto width = right - left;
    auto height = bottom - top;

    cmd->d_src = d_src;
    cmd->src_width = src_width;
    cmd->src_stride = src_stride;
    cmd->src_height = src_height;

    cmd->scale_x = src_width / (width + 1e-5);
    cmd->scale_y = src_height / (height + 1e-5);

    cmd->bounding_left  = left;
    cmd->bounding_right = right;
    cmd->bounding_top   = top;
    cmd->bounding_bottom = bottom;

    context->commands.emplace_back(cmd);
}

void cuosd_draw_nv12_source(
    cuOSDContext_t _context, int left, int top, int right, int bottom, void* d_src0, void* d_src1, int src_width, int src_stride, int src_height, unsigned char alpha, bool block_linear) {
    cuOSDContextImpl* context = (cuOSDContextImpl*)_context;
    int tl = min(left, right);
    int tt = min(top, bottom);
    int tr = max(left, right);
    int tb = max(top, bottom);
    left = tl; top = tt; right = tr; bottom = tb;

    auto cmd = make_shared<NV12SourceCommand>();
    auto width = right - left;
    auto height = bottom - top;

    cmd->d_src0 = d_src0;
    cmd->d_src1 = d_src1;

    cmd->src_width = src_width;
    cmd->src_stride = src_stride;
    cmd->src_height = src_height;

    cmd->scale_x = src_width / (width + 1e-5);
    cmd->scale_y = src_height / (height + 1e-5);

    cmd->block_linear = block_linear;
    cmd->c3 = alpha;

    cmd->bounding_left  = left;
    cmd->bounding_right = right;
    cmd->bounding_top   = top;
    cmd->bounding_bottom = bottom;

    context->commands.emplace_back(cmd);
}

//  elements:int = [num_element, e0 offset, e1 offset, e2 offset ,e3 offset, e offset.....]
//  elements on continue memory = [element0, element1, element2, ...]
static void cuosd_text_perper(
    cuOSDContext_t _context,
    int width, int height,
    void* _stream
) {
    cuOSDContextImpl* context   = (cuOSDContextImpl*)_context;
    cudaStream_t stream = (cudaStream_t)_stream;

    if (context->commands.empty() || context->text_backend == nullptr) return;
    for (auto& cmd : context->commands) {
        if (cmd->type == CommandType::Text) {
            auto text_cmd = static_pointer_cast<TextHostCommand>(cmd);
            context->text_backend->add_build_text(text_cmd->text, text_cmd->font_size, text_cmd->font_name.c_str());
        }
    }
    context->text_backend->build_bitmap(_stream);

    vector<vector<TextLocation>> locations;
    int total_locations = 0;
    for (auto& cmd : context->commands) {
        if (cmd->type == CommandType::Text) {
            auto text_cmd = static_pointer_cast<TextHostCommand>(cmd);
            int draw_x = text_cmd->x;
            text_cmd->gputile.bounding_left = text_cmd->x;
            text_cmd->gputile.bounding_bottom = text_cmd->y + text_cmd->font_size;
            text_cmd->gputile.bounding_top  = text_cmd->gputile.bounding_bottom;

            auto glyph_map = context->text_backend->query(text_cmd->font_name.c_str(), text_cmd->font_size);
            if(glyph_map == nullptr) continue;

            vector<TextLocation> textline_locations;
            int max_glyph_height = 0;
            for (auto& word : text_cmd->text) {
                auto meta = glyph_map->query(word);
                if (meta == nullptr)
                    continue;

                max_glyph_height = max(max_glyph_height, meta->height());
            }

            for (auto& word : text_cmd->text) {

                auto meta = glyph_map->query(word);
                if (meta == nullptr)
                    continue;

                int w = meta->width();
                int h = meta->height();
                int xadvance = meta->xadvance(text_cmd->font_size);
                if (w < 1 || h < 1) {
                    draw_x += meta->xadvance(text_cmd->font_size, true);
                    continue;
                }

                TextLocation location;
                location.image_x = draw_x;
                location.image_y = text_cmd->y + context->text_backend->compute_y_offset(max_glyph_height, h, meta, text_cmd->font_size);
                location.text_x  = meta->x_offset_on_bitmap();
                location.text_w  = w;
                location.text_h  = h;

                // Ignore if out of image area.
                if (location.image_x + location.text_w < 0 || location.image_x >= (int)width ||
                   location.image_y + location.text_h < 0 || location.image_y >= (int)height) {
                    draw_x += xadvance;
                    continue;
                }

                textline_locations.emplace_back(location);
                draw_x += xadvance;
                text_cmd->gputile.bounding_bottom = max(text_cmd->gputile.bounding_bottom, location.image_y + location.text_h);
                text_cmd->gputile.bounding_top    = min(text_cmd->gputile.bounding_top, location.image_y);
            }

            text_cmd->gputile.bounding_right = draw_x;
            text_cmd->gputile.bounding_left  = text_cmd->gputile.bounding_left;

            if (!textline_locations.empty()) {
                text_cmd->gputile.text_line_size = textline_locations.size();
                text_cmd->gputile.ilocation      = total_locations;
                text_cmd->gputile.c0             = cmd->c0;
                text_cmd->gputile.c1             = cmd->c1;
                text_cmd->gputile.c2             = cmd->c2;
                text_cmd->gputile.c3             = cmd->c3;
                text_cmd->gputile.type           = CommandType::Text;
                text_cmd->bounding_left          = text_cmd->gputile.bounding_left;
                text_cmd->bounding_top           = text_cmd->gputile.bounding_top;
                text_cmd->bounding_right         = text_cmd->gputile.bounding_right;
                text_cmd->bounding_bottom        = text_cmd->gputile.bounding_bottom;
                locations.emplace_back(textline_locations);
                total_locations += textline_locations.size();
            }
        }
    }
    if (locations.empty()) return;

    if (context->text_location == nullptr) context->text_location.reset(new Memory<TextLocation>());
    if (context->line_location_base == nullptr) context->line_location_base.reset(new Memory<int>());
    context->text_location->alloc_or_resize_to(total_locations);
    context->line_location_base->alloc_or_resize_to(locations.size() + 1);

    int ilocation = 0;
    context->line_location_base->host()[0] = 0;

    for (int i = 0; i < (int)locations.size(); ++i) {
        auto& text_line = locations[i];
        memcpy(context->text_location->host() + ilocation, text_line.data(), sizeof(TextLocation) * text_line.size());
        ilocation += text_line.size();
        context->line_location_base->host()[i + 1] = ilocation;
    }

    context->line_location_base->copy_host_to_device(stream);
    context->text_location->copy_host_to_device(stream);
}

void cuosd_apply(
    cuOSDContext_t _context,
    void* data0, void* data1, int width, int stride, int height, cuOSDImageFormat format,
    void* _stream, bool launch_and_clear
) {
    cuOSDContextImpl* context   = (cuOSDContextImpl*)_context;
    cudaStream_t stream = (cudaStream_t)_stream;

    if (context->commands.empty() && context->blur_commands.empty()) {
        // CUOSD_PRINT_W("Please check if there is anything to draw.\n");
        return;
    }

    if(!context->blur_commands.empty()){
        if (context->gpu_blur_commands == nullptr) context->gpu_blur_commands.reset(new Memory<BoxBlurCommand>());
        context->gpu_blur_commands->alloc_or_resize_to(context->blur_commands.size());
        for (int i = 0; i < (int)context->blur_commands.size(); ++i) {
            auto& cmd = context->blur_commands[i];
            memcpy((void*)(context->gpu_blur_commands->host() + i), (void*)cmd.get(), sizeof(BoxBlurCommand));
        }
        context->gpu_blur_commands->copy_host_to_device(stream);
    }

    if(!context->commands.empty()){
        cuosd_text_perper(context, width, height, _stream);

        context->bounding_left   = width;
        context->bounding_top    = height;
        context->bounding_right  = 0;
        context->bounding_bottom = 0;

        size_t byte_of_commands = 0;
        vector<unsigned int> cmd_offset(context->commands.size());
        for (int i = 0; i < (int)context->commands.size(); ++i) {
            auto& cmd = context->commands[i];
            cmd_offset[i] = byte_of_commands;

            context->bounding_left   = min(context->bounding_left,   cmd->bounding_left);
            context->bounding_top    = min(context->bounding_top,    cmd->bounding_top);
            context->bounding_right  = max(context->bounding_right,  cmd->bounding_right);
            context->bounding_bottom = max(context->bounding_bottom, cmd->bounding_bottom);

            if (cmd->type == CommandType::Text)
                byte_of_commands += sizeof(TextCommand);
            else if (cmd->type == CommandType::Rectangle)
                byte_of_commands += sizeof(RectangleCommand);
            else if (cmd->type == CommandType::Circle)
                byte_of_commands += sizeof(CircleCommand);
            else if (cmd->type == CommandType::Ellipse)
                byte_of_commands += sizeof(EllipseCommand);
            else if (cmd->type == CommandType::Segment)
                byte_of_commands += sizeof(SegmentCommand);
            else if (cmd->type == CommandType::PolyFill)
                byte_of_commands += sizeof(PolyFillCommand);
            else if (cmd->type == CommandType::RGBASource)
                byte_of_commands += sizeof(RGBASourceCommand);
            else if (cmd->type == CommandType::NV12Source)
                byte_of_commands += sizeof(NV12SourceCommand);
        }

        if (context->gpu_commands        == nullptr) context->gpu_commands.reset(new Memory<unsigned char>());
        if (context->gpu_commands_offset == nullptr) context->gpu_commands_offset.reset(new Memory<int>());

        context->gpu_commands->alloc_or_resize_to(byte_of_commands);
        context->gpu_commands_offset->alloc_or_resize_to(cmd_offset.size());
        memcpy(context->gpu_commands_offset->host(), cmd_offset.data(), sizeof(int) * cmd_offset.size());

        for (int i = 0; i < (int)context->commands.size(); ++i) {
            auto& cmd       = context->commands[i];
            unsigned char* pg_cmd = context->gpu_commands->host() + cmd_offset[i];

            if (cmd->type == CommandType::Text)
                memcpy(pg_cmd, (&(static_pointer_cast<TextHostCommand>(cmd)->gputile)), sizeof(TextCommand));
            else if (cmd->type == CommandType::Rectangle)
                memcpy(pg_cmd, cmd.get(), sizeof(RectangleCommand));
            else if (cmd->type == CommandType::Circle)
                memcpy(pg_cmd, cmd.get(), sizeof(CircleCommand));
            else if (cmd->type == CommandType::Ellipse)
                memcpy(pg_cmd, cmd.get(), sizeof(EllipseCommand));
            else if (cmd->type == CommandType::Segment)
                memcpy(pg_cmd, cmd.get(), sizeof(SegmentCommand));
            else if (cmd->type == CommandType::PolyFill)
                memcpy(pg_cmd, cmd.get(), sizeof(PolyFillCommand));
            else if (cmd->type == CommandType::RGBASource)
                memcpy(pg_cmd, cmd.get(), sizeof(RGBASourceCommand));
            else if (cmd->type == CommandType::NV12Source)
                memcpy(pg_cmd, cmd.get(), sizeof(NV12SourceCommand));
        }
        context->gpu_commands->copy_host_to_device(stream);
        context->gpu_commands_offset->copy_host_to_device(stream);
    }

    if (launch_and_clear) {
        cuosd_launch(_context, data0, data1, width, stride, height, format, stream);
        context->commands.clear();
        context->blur_commands.clear();
    }
}

void cuosd_clear(cuOSDContext_t context){
    if(context){
        ((cuOSDContextImpl*)context)->commands.clear();
        ((cuOSDContextImpl*)context)->blur_commands.clear();
    }
}

void cuosd_launch(
    cuOSDContext_t _context,
    void* data0, void* data1, int width, int stride, int height, cuOSDImageFormat format,
    void* stream
){
    cuOSDContextImpl* context  = (cuOSDContextImpl*)_context;
    if (context->commands.empty() && context->blur_commands.empty()) {
        CUOSD_PRINT_W("Please check if there is anything to draw\n");
        return;
    }

    unsigned char* text_bitmap = nullptr;
    int text_bitmap_width = 0;
    if (context->text_backend) {
        text_bitmap       = context->text_backend->bitmap_device_pointer();
        text_bitmap_width = context->text_backend->bitmap_width();
    }

    cuosd_launch_kernel(
        data0, data1, width, stride, height, (ImageFormat)format,
        context->text_location ? context->text_location->device() : nullptr,
        text_bitmap,
        text_bitmap_width,
        context->line_location_base ? context->line_location_base->device() : nullptr,
        context->gpu_commands ? context->gpu_commands->device() : nullptr,
        context->gpu_commands_offset ? context->gpu_commands_offset->device() : nullptr,
        context->commands.size(),
        context->bounding_left, context->bounding_top, context->bounding_right, context->bounding_bottom,
        context->have_rotate_msaa, 
        context->gpu_blur_commands ? (const unsigned char*)context->gpu_blur_commands->device() : nullptr, 
        context->blur_commands.size(),
        stream
    );
    checkRuntime(cudaPeekAtLastError());
}