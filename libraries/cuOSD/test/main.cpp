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
 
#include <stdio.h>
#include <vector>
#include <string>
#include <fstream>
#include <memory>
#include <map>
#include <cuda_runtime.h>
#include <string.h>

#include <cuosd.h>
#include "gpu_image.h"

static const double PI = 3.1415926535897932384626433832795;

#define	EXIT_SUCCESS	0	/* Successful exit status. */
#define	EXIT_FAILURE	1	/* Failing exit status.    */
#define	EXIT_WAIVED	    2	/* WAIVED exit status.     */

#define checkRuntime(call)  check_runtime(call, #call, __LINE__, __FILE__)

static bool inline check_runtime(cudaError_t e, const char* call, int line, const char *file) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime error %s # %s, code = %s [ %d ] in file %s:%d\n", call, cudaGetErrorString(e), cudaGetErrorName(e), e, file, line);
        return false;
    }
    return true;
}

static void init_cuda(int id) {
    int numOfGPUs = 0;
    std::vector<int> deviceIds;
    checkRuntime(cudaGetDeviceCount(&numOfGPUs));
    printf("%d GPUs found\n", numOfGPUs);

    if (!numOfGPUs)
    {
        exit(EXIT_WAIVED);
    }

    for (int devID = 0; devID < numOfGPUs; devID++)
    {
        int major = 0, minor = 0;
        checkRuntime(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, devID));
        checkRuntime(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, devID));
        printf("GPU Device %d: with compute capability %d.%d\n", devID, major, minor);
        deviceIds.push_back(devID);
    }

    printf(">>> Use GPU Device %d\n", deviceIds[id]);
    checkRuntime(cudaSetDevice(deviceIds[id]));
    checkRuntime(cudaFree(0));
}

// include h,  return is [l, h]
static int randl(int l, int h) {
    int value = rand() % (h - l + 1);
    return l + value;
}

static bool startswith(const char *s, const char *with, const char **last)
{
    while (*s++ == *with++)
    {
        if (*s == 0 || *with == 0)
            break;
    }
    if (*with == 0)
        *last = s + 1;
    return *with == 0;
}

static std::vector<std::string> split_with_tokens(const char *p, const char* tokens, bool one_token_repeat=false) {
    std::vector<std::string> output;
    int state  = 0;
    int ntoken = strlen(tokens) + 1;   // keep \0
    const char* prev  = p;
    char bufline[101] = {0};
    int n = 0;

    while(state < ntoken) {
        if (*p == tokens[state]) {
            n = std::min<int>(p - prev, sizeof(bufline) - 1);
            strncpy(bufline, prev, n);
            bufline[n] = 0;

            output.push_back(bufline);
            prev = p + 1;

            if (!one_token_repeat)
                state++;
        }

        if (*p == 0) {
            if (p - prev > 0) {
                n = std::min<int>(p - prev, sizeof(bufline) - 1);
                strncpy(bufline, prev, n);
                bufline[n] = 0;

                output.push_back(bufline);
            }
            break;
        }
        p++;
    }
    return output;
}

static void cuosd_apply(cuOSDContext_t context, gpu::Image* image, cudaStream_t stream, bool launch=true) {

    cuOSDImageFormat format = cuOSDImageFormat::None;
    if (image->format == gpu::ImageFormat::RGB) {
        format = cuOSDImageFormat::RGB;
    } else if (image->format == gpu::ImageFormat::RGBA) {
        format = cuOSDImageFormat::RGBA;
    } else if (image->format == gpu::ImageFormat::PitchLinearNV12) {
        format = cuOSDImageFormat::PitchLinearNV12;
    } else if (image->format == gpu::ImageFormat::BlockLinearNV12) {
        format = cuOSDImageFormat::BlockLinearNV12;
    }
    cuosd_apply(context,  image->data0, image->data1, image->width, image->stride, image->height, format, stream, launch);
}

static void cuosd_launch(cuOSDContext_t context, gpu::Image* image, cudaStream_t stream) {

    cuOSDImageFormat format = cuOSDImageFormat::None;
    if (image->format == gpu::ImageFormat::RGB) {
        format = cuOSDImageFormat::RGB;
    }else if (image->format == gpu::ImageFormat::RGBA) {
        format = cuOSDImageFormat::RGBA;
    } else if (image->format == gpu::ImageFormat::PitchLinearNV12) {
        format = cuOSDImageFormat::PitchLinearNV12;
    } else if (image->format == gpu::ImageFormat::BlockLinearNV12) {
        format = cuOSDImageFormat::BlockLinearNV12;
    }
    cuosd_launch(context, image->data0, image->data1, image->width, image->stride, image->height, format, stream);
}

static int simple_draw() {
    cudaStream_t stream = nullptr;
    checkRuntime(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    printf("Simple draw.\n");
    auto context = cuosd_context_create();
    gpu::Image* image = gpu::create_image(1280, 720, gpu::ImageFormat::RGB);
    gpu::set_color(image, 255, 255, 255, 255, stream);
    gpu::copy_yuvnv12_to(image, 0, 0, 1280, 720, "data/image/nv12_3840x2160.yuv", 3840, 2160, 180, stream);
    gpu::save_image(image, "input.png", stream);

    std::ifstream in("data/std-random-boxes.txt", std::ios::in);
    std::string line;
    int nline = 0;
    const char* font_name = "data/simhei.ttf";
    while(getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;

        auto words = split_with_tokens(line.c_str(), ",", true);
        if (nline == 0) {
            nline++;
            continue;
        }

        if (words[0] == "detbox") {
            if (words.size() != 9) {
                printf("Invalid number of detbox element. The accept format is [type, left, top, right, bottom, thickness, name, confidence, font_size]\n");
                break;
            }
            int left   = std::atoi(words[1].c_str());
            int top    = std::atoi(words[2].c_str());
            int right  = std::atoi(words[3].c_str());
            int bottom = std::atoi(words[4].c_str());
            int thickness = std::atoi(words[5].c_str());
            std::string name = words[6];
            std::string confidence = words[7];
            int font_size = std::atoi(words[8].c_str());

            cuosd_draw_rectangle(context, left, top, right, bottom, thickness, {0, 255, 0, 255}, {0, 0, 255, 100});
            cuosd_draw_text(context, (name + "  " + confidence).c_str(), font_size, font_name, left, top, {0, 0, 0, 255}, {255, 255, 0, 255});
        }
        nline++;
    }

    cuosd_draw_text(context, 
        "Shakespearean quotes:\n"
        "\n"
        "Words cannot express true love, loyalty behavior is the best explanation.\n"
        "Love is a woman with the ears, and if the men will love, but love is to use your eyes.\n"
        "The empty vessels make the greatest sound.\n"
        "No man or woman is worth your tears, and the one who is, won’t make you cry.\n"
        "For thy sweet love remember'd such wealth brings That then I scorn to change my state with kings.\n"
        "A sad thing in life is when you meet someone who means a lot to you, only to find out in the \nend that it was never meant to be and you just have to let go."
        , 13, "data/simhei.ttf", 10, 10, cuOSDColor{0, 255, 0, 255}, cuOSDColor{60, 60, 60, 200});
    cuosd_apply(context, image, stream);
    cuosd_context_destroy(context);

    printf("Save to output.png\n");
    gpu::save_image(image, "output.png", stream);
    checkRuntime(cudaStreamDestroy(stream));
    return 0;
}

static int polyline() {
    cudaStream_t stream = nullptr;
    checkRuntime(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    printf("Test cuosd_draw_polyline.\n");
    auto context = cuosd_context_create();
    gpu::Image* image = gpu::create_image(1280, 720, gpu::ImageFormat::PitchLinearNV12);
    gpu::set_color(image, 255, 255, 255, 255, stream);
    gpu::copy_yuvnv12_to(image, 0, 0, 1280, 720, "data/image/nv12_3840x2160.yuv", 3840, 2160, 180, stream);
    gpu::save_image(image, "input.png", stream);

    gpu::Polyline* polyline = gpu::create_polyline();

    cuosd_draw_polyline(context, polyline->h_pts, polyline->d_pts, polyline->n_pts, 6, true, {0, 255, 0, 255}, true, {0, 0, 255, 120});

    cuosd_apply(context, image, stream);
    cuosd_context_destroy(context);

    printf("Save to output.png\n");
    gpu::save_image(image, "output.png", stream);
    gpu::free_polyline(polyline);
    checkRuntime(cudaStreamDestroy(stream));
    return 0;
}

static int ellipse() {
    cudaStream_t stream = nullptr;
    checkRuntime(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    printf("Test cuosd_draw_ellipse.\n");
    auto context = cuosd_context_create();
    gpu::Image* image = gpu::create_image(1280, 720, gpu::ImageFormat::PitchLinearNV12);
    gpu::set_color(image, 255, 255, 255, 255, stream);
    gpu::copy_yuvnv12_to(image, 0, 0, 1280, 720, "data/image/nv12_3840x2160.yuv", 3840, 2160, 180, stream);
    gpu::save_image(image, "input.png", stream);

    int h = 720;
    int w = 1280;
    float pi = 3.1415925f;

    // cuosd_draw_ellipse(context, 600, 300, 200, 50, pi/2, 5, {0, 0, 255, 200}, {0, 255, 0, 0});
    for (int i = 0; i < 10; ++i)
    {
        cuosd_draw_ellipse(context, randl(0, w), randl(0, h), randl(10, 300), randl(10, 300), pi / randl(0, 360), randl(1, 10), {255, 255, 0, 200}, {0, 0, 255, 200});
    };
    cuosd_apply(context, image, stream);
    cuosd_context_destroy(context);

    printf("Save to output.png\n");
    gpu::save_image(image, "output.png", stream);
    checkRuntime(cudaStreamDestroy(stream));
    return 0;
}

static int segment() {
    cudaStream_t stream = nullptr;
    checkRuntime(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    printf("Test cuosd_draw_segmentmask.\n");
    auto context = cuosd_context_create();
    gpu::Image* image = gpu::create_image(1280, 720, gpu::ImageFormat::RGBA);
    gpu::set_color(image, 255, 255, 255, 255, stream);
    gpu::copy_yuvnv12_to(image, 0, 0, 1280, 720, "data/image/nv12_3840x2160.yuv", 3840, 2160, 180, stream);
    gpu::save_image(image, "input.png", stream);

    gpu::Segment* segment = gpu::create_segment();

    std::ifstream in("data/std-random-boxes.txt", std::ios::in);
    std::string line;
    int nline = 0;
    const char* font_name = "data/simhei.ttf";
    while(getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;

        auto words = split_with_tokens(line.c_str(), ",", true);
        if (nline == 0) {
            nline++;
            continue;
        }

        if (words[0] == "detbox") {
            if (words.size() != 9) {
                printf("Invalid number of detbox element. The accept format is [type, left, top, right, bottom, thickness, name, confidence, font_size]\n");
                break;
            }
            int left   = std::atoi(words[1].c_str());
            int top    = std::atoi(words[2].c_str());
            int right  = std::atoi(words[3].c_str());
            int bottom = std::atoi(words[4].c_str());
            int thickness = std::atoi(words[5].c_str());
            std::string name = words[6];
            std::string confidence = words[7];
            int font_size = std::atoi(words[8].c_str());

            cuosd_draw_segmentmask(context, left, top, right, bottom, thickness, segment->data, segment->width, segment->height, 0.2, {0, 255, 0, 255}, {0, 0, 255, 255});
            cuosd_draw_text(context, (name + "  " + confidence).c_str(), font_size, font_name, left, top - font_size * 3, {0, 0, 0, 255}, {255, 255, 0, 255});
        }
        nline++;
    }

    cuosd_apply(context, image, stream);
    cuosd_context_destroy(context);

    printf("Save to output.png\n");
    gpu::save_image(image, "output.png", stream);
    gpu::free_segment(segment);
    checkRuntime(cudaStreamDestroy(stream));
    return 0;
}

static int segment2() {
    cudaStream_t stream = nullptr;
    cudaEvent_t start, end;
    checkRuntime(cudaEventCreate(&end));
    checkRuntime(cudaEventCreate(&start));
    checkRuntime(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    auto context = cuosd_context_create();
    gpu::Image* image = gpu::create_image(1280, 720, gpu::ImageFormat::BlockLinearNV12);
    gpu::copy_yuvnv12_to(image, 0, 0, 1280, 720, "data/assets/sample.nv12", 1280, 720, 255, stream);
    gpu::save_image(image, "input.png", stream);

    gpu::Image* mask = gpu::create_image(1280, 720, gpu::ImageFormat::RGBA);
    gpu::copy_yuvnv12_to(mask, 0, 0, 1280, 720, "data/assets/mask.nv12", 1280, 720, 10, stream);
    gpu::save_image(mask, "mask.png", stream);

    if (mask->format == gpu::ImageFormat::RGBA) cuosd_draw_rgba_source(context, 0, 0, 1280, 720, mask->data0, mask->width, 4 * mask->width, mask->height);
    if (mask->format == gpu::ImageFormat::BlockLinearNV12) cuosd_draw_nv12_source(context, 0, 0, 1280, 720, mask->data0, mask->data1, mask->width, mask->width, mask->height, 10, true);
    if (mask->format == gpu::ImageFormat::PitchLinearNV12) cuosd_draw_nv12_source(context, 0, 0, 1280, 720, mask->data0, mask->data1, mask->width, mask->width, mask->height, 10, false);

    cuosd_apply(context, image, stream, false);

    for (int i = 0; i < 1000; ++i) {
        cuosd_launch(context, image, stream);
    }

    checkRuntime(cudaStreamSynchronize(stream));
    checkRuntime(cudaEventRecord(start, stream));

    for (int i = 0; i < 1000; ++i) {
        cuosd_launch(context, image, stream);
    }

    float gpu_time;
    checkRuntime(cudaEventRecord(end, stream));
    checkRuntime(cudaEventSynchronize(end));
    checkRuntime(cudaEventElapsedTime(&gpu_time, start, end));

    printf("draw [%dx%d-%s] mask on [%dx%d-%s] image -> performance: %.2f us\n",
            mask->width, mask->height, gpu::image_format_name(mask->format),
            image->width, image->height, gpu::image_format_name(image->format), gpu_time);

    cuosd_context_destroy(context);
    gpu::save_image(image, "output.png", stream);

    checkRuntime(cudaEventDestroy(end));
    checkRuntime(cudaEventDestroy(start));
    checkRuntime(cudaStreamDestroy(stream));

    return 0;
}

static int perf(
    int input_width, int input_height,
    gpu::ImageFormat format,
    const char* load,
    int lines,
    int rotateboxes,
    int circles,
    int rectangles,
    int texts,
    int arrows,
    int points,
    int clocks,
    int blurs,
    int seed,
    const char* save,
    const char* font,
    bool fix_pos,
    bool set_bg
) {
    cudaStream_t stream = nullptr;
    cudaEvent_t start, end;
    checkRuntime(cudaEventCreate(&end));
    checkRuntime(cudaEventCreate(&start));
    checkRuntime(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    auto context = cuosd_context_create();
    gpu::Image* image = gpu::create_image(input_width, input_height, format);
    srand(seed);

    int w = input_width;
    int h = input_height;
    for (int i = 0; i < lines; ++i)     {if (fix_pos)srand(seed); cuosd_draw_line(context, randl(0, w), randl(0, h), randl(0, w), randl(0, w), randl(1, 10), {0, 255, 0, 255});};
    for (int i = 0; i < rotateboxes; ++i)     {if (fix_pos)srand(seed); cuosd_draw_rotationbox(context, randl(0, w), randl(0, h), randl(10, 300), randl(10, 300), randl(0, 360), randl(1, 10), {255, 0, 0, 200}, true, {0, 255, 0, set_bg?(unsigned char)128:(unsigned char)0});};
    for (int i = 0; i < circles; ++i)   {if (fix_pos)srand(seed); cuosd_draw_circle(context, randl(0, w), randl(0, h), randl(10, 100), randl(1, 10), {255, 0, 128, 128}, {0, 0, 255, 100});};
    for (int i = 0; i < rectangles; ++i){if (fix_pos)srand(seed); cuosd_draw_rectangle(context, randl(0, w), randl(0, h), randl(0, w), randl(0, h), 5, {255, 128, 0, 128}, {0, 0, 255, 100});};
    for (int i = 0; i < texts; ++i)     {if (fix_pos)srand(seed); cuosd_draw_text(context, "cuOSD你好", 25, font, randl(0, w), randl(0, h), {0, 0, 255, 128});};
    for (int i = 0; i < arrows; ++i)    {if (fix_pos)srand(seed); cuosd_draw_arrow(context, randl(0, w), randl(0, h), randl(0, w), randl(0, h), randl(10, 50), randl(1, 10), {255, 0, 0, 200});};
    for (int i = 0; i < points; ++i)    {if (fix_pos)srand(seed); cuosd_draw_point(context, randl(0, w), randl(0, h), randl(10, 100), {0, 255, 0, 255});};
    for (int i = 0; i < clocks; ++i)    {if (fix_pos)srand(seed); cuosd_draw_clock(context, cuOSDClockFormat::YYMMDD_HHMMSS, time(0), 25, font, randl(0, w), randl(0, h), {0, 128, 255, 255});};
    for (int i = 0; i < blurs; ++i)     {
        if (fix_pos)srand(seed); 
        int x = randl(0, w);
        int y = randl(0, h);
        int bw = randl(50, 200);
        int bh = randl(50, 200);
        cuosd_draw_boxblur(context, x, y, x + bw, y + bh);
    };

    if (load) {
        std::ifstream in(load, std::ios::in);
        std::string line;
        int nline = 0;
        while(getline(in, line)) {
            if (line.empty() || line[0] == '#') continue;

            auto words = split_with_tokens(line.c_str(), ",", true);
            if (nline == 0) {
                if (words.size() != 2) {
                    printf("Invalid number of head element. Accepted format is [width],[height]\n");
                    break;
                }
                nline++;
                continue;
            }

            if (words[0] == "detbox") {

                if (words.size() != 9) {
                    printf("Invalid number of detbox element. Accepted format is [type, left, top, right, bottom, thickness, name, confidence, font_size]\n");
                    break;
                }
                rectangles++;
                texts++;

                int left   = std::atoi(words[1].c_str());
                int top    = std::atoi(words[2].c_str());
                int right  = std::atoi(words[3].c_str());
                int bottom = std::atoi(words[4].c_str());
                int thickness = std::atoi(words[5].c_str());
                std::string name = words[6];
                std::string confidence = words[7];
                int font_size = std::atoi(words[8].c_str());

                cuosd_draw_rectangle(context, left, top, right, bottom, thickness, {0, 255, 0, 255}, {0, 0, 255, 100});
                cuosd_draw_text(context, (name + "  " + confidence).c_str(), font_size, font, left, top - font_size * 1.3 - thickness / 2 - 3, {0, 0, 0, 255}, {255, 255, 0, 255});
            }
            nline++;
        }
    }

    char prefix[1000] = {0};
    char* pprefix     = prefix;
    if (lines)       pprefix += sprintf(pprefix, "line=%d ", lines);
    if (rotateboxes) pprefix += sprintf(pprefix, "rotatebox=%d ", rotateboxes);
    if (circles)     pprefix += sprintf(pprefix, "circle=%d ", circles);
    if (rectangles)  pprefix += sprintf(pprefix, "rectangle=%d ", rectangles);
    if (texts)       pprefix += sprintf(pprefix, "text=%d ", texts);
    if (arrows)      pprefix += sprintf(pprefix, "arrow=%d ", arrows);
    if (points)      pprefix += sprintf(pprefix, "point=%d ", points);
    if (clocks)      pprefix += sprintf(pprefix, "clock=%d ", clocks);
    if (blurs)       pprefix += sprintf(pprefix, "blur=%d ", blurs);
    cuosd_apply(context, image, stream, false);

    for (int i = 0; i < 1000; ++i) {
        cuosd_launch(context, image, stream);
    }

    checkRuntime(cudaStreamSynchronize(stream));
    checkRuntime(cudaEventRecord(start, stream));

    for (int i = 0; i < 1000; ++i)
        cuosd_launch(context, image, stream);

    float gpu_time;
    checkRuntime(cudaEventRecord(end, stream));
    checkRuntime(cudaEventSynchronize(end));
    checkRuntime(cudaEventElapsedTime(&gpu_time, start, end));

    if (save) {
        // gpu::set_color(image, 255, 255, 255, 255, stream);
        gpu::copy_yuvnv12_to(image, 0, 0, input_width, input_height, "data/image/nv12_3840x2160.yuv", 3840, 2160, 180, stream);
        cuosd_launch(context, image, stream);
        gpu::save_image(image, save, stream);
    }

    cuosd_context_destroy(context);
    checkRuntime(cudaStreamDestroy(stream));
    checkRuntime(cudaEventDestroy(start));
    checkRuntime(cudaEventDestroy(end));
    printf("%dx%d/%s:%s-> performance: %.2f us\n", input_width, input_height, gpu::image_format_name(format), prefix, gpu_time);
    return 0;
}

static bool parse_input(const char* p, int& input_width, int& input_height, gpu::ImageFormat& input_format) {
    auto params = split_with_tokens(p, "x/");
    if (params.size() != 3) {printf("Unknow format [%s], options is [width]x[height]/[format]\n", p); return false;}

    std::map<std::string, gpu::ImageFormat> format_map{
        {"BL", gpu::ImageFormat::BlockLinearNV12},
        {"PL", gpu::ImageFormat::PitchLinearNV12},
        {"RGBA", gpu::ImageFormat::RGBA},
        {"RGB", gpu::ImageFormat::RGB}
    };

    if (format_map.find(params[2]) == format_map.end()){printf("Unknow format [%s], options is [BL, PL, RGBA, RGB]\n", params[2].c_str()); return false;}
    input_width  = std::atoi(params[0].c_str());
    input_height = std::atoi(params[1].c_str());
    input_format = format_map[params[2]];
    return true;
}

static void help()
{
    printf(
        "Usage: \n"
        "     ./cuosd simple\n"
        "     ./cuosd comp --line\n"
        "     ./cuosd perf --input=3840x2160/BL --font=data/my.nvfont --line=100 --rotatebox=100 --circle=100 --rectangle=100 --text=100 --arrow=100 --point=100 --clock=100 --save=output.png --seed=31\n"
        "     ./cuosd perf --input=1280x720/BL --font=data/my.nvfont --load=data/std-random-boxes.txt --save=output.png --seed=31\n"
        "\n"
        "Command List:\n"
        "  ./cuosd simple\n"
        "    Simple image rendering and save result to jpg file.\n"
        "\n\n"
        "  ./cuosd comp --line\n"
        "    Benchmark test of drawing 100 lines using the same configuration as nvOSD.\n"
        "\n\n"
        "  ./cuosd perf --input=3840x2160/BL --font=data/my.nvfont --line=100 --rotatebox=100 --circle=100 --rectangle=100 --text=100 --arrow=100 --point=100 --clock=100 --save=output.png --seed=31\n"
        "  ./cuosd perf --input=1280x720/BL --font=data/my.nvfont --load=data/std-random-boxes.txt --save=output.png --seed=31\n"
        "    Perf test for given config.\n"
        "\n"
        "    Prameters:\n"
        "    --input:  Set input size and format, Syntax format is: [width]x[height]/[format]\n"
        "              format can be 'BL', 'PL', 'RGBA' \n"
        "    --load:      Load elements from file to rendering pipeline.\n"
        "    --line:      Add lines to rendering pipeline\n"
        "    --rotatebox: Add rototebox to rendering pipeline\n"
        "    --circle:    Add circles to rendering pipeline\n"
        "    --rectangle: Add rectangles to rendering pipeline\n"
        "    --text:      Add texts to rendering pipeline\n"
        "    --arrow:     Add arrows to rendering pipeline\n"
        "    --point:     Add points to rendering pipeline\n"
        "    --clock:     Add clock to rendering pipeline\n"
        "    --blur:      Add boxblur to rendering pipeline\n"
        "    --save:      Sets the path of the output. default does not save the output\n"
        "    --font:      Sets the font file used for text contexting.\n"
        "    --fix-pos:   All elements of the same kind use the same coordinates, not random\n"
        "    --seed:      Set seed number for random engine\n"
    );
    exit(EXIT_SUCCESS);
}

static int perf(int argc, char **argv) {
    const char *save  = nullptr;
    const char *value = nullptr;
    const char *font  = "data/simfang.ttf";
    const char *load  = nullptr;
    int input_width = 1920;
    int input_height = 1080;
    gpu::ImageFormat format = gpu::ImageFormat::RGBA;
    int lines = 0;
    int rotateboxes = 0;
    int circles = 0;
    int rectangles = 0;
    int texts = 0;
    int arrows = 0;
    int points = 0;
    int clocks = 0;
    int blurs  = 0;
    int seed = 31;
    bool fix_pos = false;
    bool parse_failed = false;
    bool has_input  = false;
    bool set_bg = false;

    for (int i = 2; i < argc; ++i) {
        if (startswith(argv[i], "--input", &value)) {
            has_input = true;
            parse_failed |= !parse_input(value, input_width, input_height, format);
        } else if (startswith(argv[i], "--load", &value)) {
            load = value;
        } else if (startswith(argv[i], "--line", &value)) {
            lines = atoi(value);
        } else if (startswith(argv[i], "--rotatebox", &value)) {
            rotateboxes = atoi(value);
        } else if (startswith(argv[i], "--circle", &value)) {
            circles = atoi(value);
        } else if (startswith(argv[i], "--rectangle", &value)) {
            rectangles = atoi(value);
        } else if (startswith(argv[i], "--text", &value)) {
            texts = atoi(value);
        } else if (startswith(argv[i], "--arrow", &value)) {
            arrows = atoi(value);
        } else if (startswith(argv[i], "--point", &value)) {
            points = atoi(value);
        } else if (startswith(argv[i], "--clock", &value)) {
            clocks = atoi(value);
        } else if (startswith(argv[i], "--blur", &value)) {
            blurs = atoi(value);
        } else if (startswith(argv[i], "--save", &value)) {
            save = value;
        } else if (startswith(argv[i], "--font", &value)) {
            font = value;
        } else if (startswith(argv[i], "--seed", &value)) {
            seed = atoi(value);
        } else if (startswith(argv[i], "--fix-pos", &value)) {
            fix_pos = true;
        } else if (startswith(argv[i], "--set-bg", &value)) {
            set_bg = true;
        } else {
            help();
        }
    }

    if (parse_failed || !has_input) help();
    return perf(
        input_width, input_height, format, load, lines, rotateboxes,
        circles, rectangles, texts, arrows, points, clocks, blurs,
        seed, save, font, fix_pos, set_bg
    );
}

static int comp(int argc, char **argv) {
    int input_width = 1920;
    int input_height = 1080;
    const char *value = nullptr;
    bool lines = false;
    bool circles = false;
    bool rectangles = false;
    bool texts = false;
    bool arrows = false;
    bool set_bg = false;
    bool interp = false;

    for (int i = 2; i < argc; ++i) {
        if (startswith(argv[i], "--line", &value)) {
            lines = true;
        } else if (startswith(argv[i], "--circle", &value)) {
            circles = true;
        } else if (startswith(argv[i], "--rectangle", &value)) {
            rectangles = true;
        } else if (startswith(argv[i], "--text", &value)) {
            texts = true;
        } else if (startswith(argv[i], "--arrow", &value)) {
            arrows = true;
        } else if (startswith(argv[i], "--device", &value)) {
            init_cuda(atoi(value));
        } else if (startswith(argv[i], "--set-bg", &value)) {
            set_bg = true;
        } else if (startswith(argv[i], "--interp", &value)) {
            interp = true;
        } else {
            help();
        }
    }

    std::vector<gpu::ImageFormat> format_vec{
        gpu::ImageFormat::RGB,
        gpu::ImageFormat::RGBA,
        gpu::ImageFormat::BlockLinearNV12,
        gpu::ImageFormat::PitchLinearNV12
    };

    const char* font = "data/simfang.ttf";
    for (int i = 0; i < (int)format_vec.size(); i++) {
        gpu::ImageFormat format = format_vec[i];
        cudaStream_t stream = nullptr;
        cudaEvent_t start, end;
        checkRuntime(cudaEventCreate(&end));
        checkRuntime(cudaEventCreate(&start));
        checkRuntime(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

        auto context = cuosd_context_create();
        // cuosd_set_text_backend(context, cuOSDTextBackend::PangoCairo);
        gpu::Image* image = gpu::create_image(input_width, input_height, format);
        gpu::set_color(image, 255, 255, 255, 255, stream);
        gpu::copy_yuvnv12_to(image, 0, 0, input_width, input_height, "data/image/nv12_3840x2160.yuv", 3840, 2160, 180, stream);
        gpu::save_image(image, "input.png", stream);

        auto draw_elements = [&](){
            if (rectangles) for (int i = 0; i < 100; ++i) cuosd_draw_rectangle(context, i * 12, i * 6, i * 12 + 100, i * 6 + 300, 5, {255, 0, 0, 255}, set_bg ? cuOSDColor({0, 0, 255, 100}) : cuOSDColor({0, 0, 0, 0}));
            // if (texts) for (int i = 0; i < 100; ++i) cuosd_draw_text(context, "oooooooooooooooooooooooooooooooooooooooooooooooooooo", 18, 30, 30 + i * 6, {255, 0, 255, 255}, {0, 0, 255, 100});
            if (lines) for (int i = 0; i < 100; ++i) cuosd_draw_line(context, 800, 100 + i * 6, 300, i * 6, 4, {0, 255, 204, 255}, interp);
            if (circles) for (int i = 0; i < 100; ++i) cuosd_draw_circle(context, 500, 100 + i * 6, 100, 1, {0, 0, 255, 255}, set_bg ? cuOSDColor({0, 0, 255, 100}) : cuOSDColor({0, 0, 0, 0}));
            if (arrows) for (int i = 0; i < 100; ++i) cuosd_draw_arrow(context, 500, 100 + i * 6, 1000, 200 + i * 6, 35, 4, {0, 255, 204, 255}, interp);
            if (texts) for (int i = 0; i < 100; ++i) {
                if (i<30) {
                    cuosd_draw_text(context, "欢迎使用cuOSD!", 18, font, 30, 60 + i * 30, {255, 255, 0, 255}, {0, 0, 0, 255});
                }
                else if (i<60) {
                    cuosd_draw_text(context, "欢迎使用cuOSD!", 18, font, 600, 60 + (i - 30) * 30, {255, 255, 0, 255}, {0, 0, 0, 255});
                }
                else {
                    cuosd_draw_text(context, "欢迎使用cuOSD!", 18, font, 1200, 60 + (i - 60) * 20, {255, 255, 0, 255}, {0, 0, 0, 255});
                }
            }
            cuosd_apply(context, image, stream);
        };

        for (int i = 0; i < 1000; ++i) {
            draw_elements();
        }

        checkRuntime(cudaStreamSynchronize(stream));
        checkRuntime(cudaEventRecord(start, stream));

        for (int i = 0; i < 1000; ++i) {
            draw_elements();
        }

        float gpu_time;
        checkRuntime(cudaEventRecord(end, stream));
        checkRuntime(cudaEventSynchronize(end));
        checkRuntime(cudaEventElapsedTime(&gpu_time, start, end));

        gpu::set_color(image, 255, 255, 255, 128, stream);
        gpu::copy_yuvnv12_to(image, 0, 0, input_width, input_height, "data/image/nv12_3840x2160.yuv", 3840, 2160, 180, stream);
        draw_elements();

        printf("%dx%d/%s-> performance: %.2f us\n", input_width, input_height, gpu::image_format_name(format), gpu_time);
        gpu::save_image(image, "output.png", stream);
        cuosd_context_destroy(context);
        checkRuntime(cudaEventDestroy(end));
        checkRuntime(cudaEventDestroy(start));
        checkRuntime(cudaStreamDestroy(stream));
    }
    return 0;
}

int main(int argc, char **argv)
{
    const char* cmd   = nullptr;
    if (argc < 2)
        help();

    cmd = argv[1];
    if (strcmp(cmd, "perf") == 0) {
        return perf(argc, argv);
    } else if (strcmp(cmd, "simple") == 0) {
        return simple_draw();
    } else if (strcmp(cmd, "segment") == 0) {
        return segment();
    } else if (strcmp(cmd, "segment2") == 0) {
        return segment2();
    } else if (strcmp(cmd, "ellipse") == 0) {
        return ellipse();
    } else if (strcmp(cmd, "polyline") == 0) {
        return polyline();
    } else if (strcmp(cmd, "comp") == 0) {
        return comp(argc, argv);
    } else {
        help();
    }
}