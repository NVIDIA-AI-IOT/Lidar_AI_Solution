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
#include <functional>
#include <unordered_map>
#include <memory>
#include "yuv_to_rgb.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

using namespace std;

#define checkRuntime(call) check_runtime(call, #call, __LINE__, __FILE__)

bool __inline__ check_runtime(cudaError_t e, const char *call, int line, const char *file)
{
    if (e != cudaSuccess)
    {
        std::fprintf(stderr, "CUDA Runtime error %s # %s, code = %s [ %d ] in file %s:%d\n", call, cudaGetErrorString(e), cudaGetErrorName(e), e, file, line);
        return false;
    }
    return true;
}

shared_ptr<YUVHostImage> load_yuv(YUVFormat format)
{   
    string path = "workspace/data/nv12_3840x2160.yuv";
    if(format == YUVFormat::YUV422Packed_YUYV)
        path = "workspace/data/yuyv_3840x2160_yuyv422.yuv";
        // path = "workspace/data/aa.yuv";

    if(format == YUVFormat::NV12BlockLinear)
        format = YUVFormat::NV12PitchLinear;

    YUVHostImage *yuvhost = read_yuv(path, 3840, 2160, format);
    return shared_ptr<YUVHostImage>(yuvhost, free_yuv_host_image);
}

int run(
    bool perf, const char *save,
    int input_width, int input_height, int input_batch, YUVFormat input_format,
    int output_width, int output_height, PixelLayout output_layout, DataType output_dtype, Interpolation interp
){
    auto yuvhost = load_yuv(input_format);
    if (yuvhost == nullptr)
        return -1;

    FillColor color;
    // memset(&color, 0, sizeof(color));
    color.color[0] = 0;
    color.color[1] = 255;
    color.color[2] = 128;

    cudaStream_t stream = nullptr;
    cudaEvent_t start, end;
    checkRuntime(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    float gpu_time;
    auto input = create_yuv_gpu_image(input_width, input_height, input_batch, input_format);
    auto output = create_rgb_gpu_image(output_width, output_height, input_batch, output_layout, output_dtype);
    for (int ibatch = 0; ibatch < input_batch; ++ibatch)
        copy_yuv_host_to_gpu(yuvhost.get(), input, ibatch, input_width, input_height, stream);

    batched_convert_yuv_to_rgb(input, output, output->width, output->height, 0, 0, color, 0, 0, 0, 1, 1, 1, interp, stream);

    if (perf)
    {
        checkRuntime(cudaEventCreate(&start));
        checkRuntime(cudaEventCreate(&end));

        // warmup
        for (int i = 0; i < 1000; ++i)
            batched_convert_yuv_to_rgb(input, output, output->width, output->height, 0, 0, color, 0, 0, 0, 1, 1, 1, interp, stream);

        checkRuntime(cudaStreamSynchronize(stream));
        checkRuntime(cudaEventRecord(start, stream));
        for (int i = 0; i < 1000; ++i)
            batched_convert_yuv_to_rgb(input, output, output->width, output->height, 0, 0, color, 0, 0, 0, 1, 1, 1, interp, stream);

        checkRuntime(cudaEventRecord(end, stream));
        checkRuntime(cudaEventSynchronize(end));
        checkRuntime(cudaEventElapsedTime(&gpu_time, start, end));

        printf("[%s] %dx%dx%d/%s to %dx%d/%s/%s performance: %.2f us\n",
               interp_name(interp), input_width, input_height, input_batch, yuvformat_name(input_format),
               output_width, output_height, dtype_name(output_dtype), pixel_layout_name(output_layout),
               gpu_time);
        checkRuntime(cudaEventDestroy(start));
        checkRuntime(cudaEventDestroy(end));
    }

    if (save && strlen(save) > 0)
    {
        printf("Save to %s\n", save);
        save_rgbgpu_to_file(save, output, stream);
    }

    checkRuntime(cudaStreamDestroy(stream));
    free_rgb_gpu_image(output);
    free_yuv_gpu_image(input);
    return 0;
}

bool startswith(const char *s, const char *with, const char **last)
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

vector<string> split_with_tokens(const char *p, const char* tokens){

    vector<string> output;
    int state  = 0;
    int ntoken = strlen(tokens) + 1;   // keep \0
    const char* prev  = p;
    char bufline[101] = {0};
    int n = 0;
    while(state < ntoken){
        if(*p == tokens[state]){
            n = std::min<int>(p - prev, sizeof(bufline) - 1);
            strncpy(bufline, prev, n);
            bufline[n] = 0;

            output.push_back(bufline);
            state++;
            prev = p + 1;
        }

        if(*p == 0) break;
        p++;
    }
    return output;
}

bool parse_input(const char* p, int& input_width, int& input_height, int& input_batch, YUVFormat& input_format){

    auto params = split_with_tokens(p, "xx/");
    if(params.size() != 4) return false;

    unordered_map<string, YUVFormat> format_map{
        {"BL", YUVFormat::NV12BlockLinear},
        {"PL", YUVFormat::NV12PitchLinear},
        {"YUYV", YUVFormat::YUV422Packed_YUYV}
    };

    if(format_map.find(params[3]) == format_map.end()){printf("Unknow format [%s], options is [BL, PL, YUYV]\n", params[3].c_str()); return false;}
    input_width  = std::atoi(params[0].c_str());
    input_height = std::atoi(params[1].c_str());
    input_batch  = std::atoi(params[2].c_str());
    input_format = format_map[params[3]];
    return true;
}

bool parse_output(const char* p, int& output_width, int& output_height, PixelLayout& output_layout, DataType& output_dtype){

    auto params = split_with_tokens(p, "x//");
    if(params.size() != 4) return false;

    output_width = std::atoi(params[0].c_str());
    output_height = std::atoi(params[1].c_str());

    unordered_map<string, DataType> dtype_map{
        {"uint8",   DataType::Uint8},
        {"int8",   DataType::Int8},
        {"float32", DataType::Float32},
        {"float16", DataType::Float16}
    };

    unordered_map<string, PixelLayout> layout_map{
        {"NCHW16_BGR", PixelLayout::NCHW16_BGR},
        {"NCHW16_RGB", PixelLayout::NCHW16_RGB},
        {"NCHW32_BGR", PixelLayout::NCHW32_BGR},
        {"NCHW32_RGB", PixelLayout::NCHW32_RGB},
        {"NCHW4_BGR", PixelLayout::NCHW4_BGR},
        {"NCHW4_RGB", PixelLayout::NCHW4_RGB},
        {"NCHW_BGR",   PixelLayout::NCHW_BGR},
        {"NCHW_RGB",   PixelLayout::NCHW_RGB},
        {"NHWC_BGR",   PixelLayout::NHWC_BGR},
        {"NHWC_RGB",   PixelLayout::NHWC_RGB}
    };

    if(dtype_map.find(params[2])  == dtype_map.end())  {printf("Unknow dtype [%s], options is [uint8, float32, float16]\n", params[2].c_str()); return false;}
    if(layout_map.find(params[3]) == layout_map.end()) {printf("Unknow layout [%s], options is [NCHW16_BGR, NCHW16_RGB, NCHW_BGR, NCHW_RGB, NHWC_BGR, NHWC_RGB]\n", params[3].c_str()); return false;}
    output_dtype  = dtype_map[params[2]];
    output_layout = layout_map[params[3]];
    return true;
}

bool parse_interp(const char* p, Interpolation& interp){

    unordered_map<string, Interpolation> interp_map{
        {"nearest",  Interpolation::Nearest},
        {"bilinear", Interpolation::Bilinear}
    };

    if(interp_map.find(p)  == interp_map.end()){printf("Unknow interpolation [%s], options is [nearest, bilinear]\n", p); return false;}
    interp = interp_map[p];
    return true;
}

void help()
{
    printf(
        "Usage: ./yuvtorgb --input=3840x2160x1/BL --output=1280x720/uint8/NCHW_RGB --interp=nearest --save=tensor.binary --perf\n"
        "\n"
        "parameters:\n"
        "    --input:  Set input size and format, Syntax format is: [width]x[height]x[batch]/[format]\n"
        "              format can be 'BL' or 'PL' or 'YUYV' \n"
        "    --output: Set output size and layout, Syntax format is: [width]x[height]/[dtype]/[layout]\n"
        "              dtype can be 'int8', 'uint8', 'float16' or 'float32'\n"
        "              layout can be one of the following: NCHW_RGB NCHW_BGR NHWC_RGB NHWC_BGR for GPU, NCHW16_RGB NCHW16_BGR for DLA\n"
        "    --interp: Set rescale mode. Here's the choice 'nearest' or 'bilinear', default is nearest\n"
        "    --save:   Sets the path of the output. default does not save the output\n"
        "    --perf:   Launch performance test with 1000x warmup and 1000x iteration\n"
    );
    exit(0);
}

int main(int argc, char **argv)
{
    bool perf = false;
    const char *save  = nullptr;
    const char *value = nullptr;
    int input_width, input_height, input_batch;
    int output_width, output_height;
    YUVFormat input_format   = YUVFormat::NoneEnum;
    PixelLayout output_layout = PixelLayout::NoneEnum;
    DataType output_dtype     = DataType::NoneEnum;
    Interpolation interp      = Interpolation::Nearest;

    if (argc < 3)
        help();

    bool parse_failed = false;
    bool has_input  = false;
    bool has_output = false;
    for(int i = 1; i < argc; ++i){
        if (startswith(argv[i], "--input", &value)){
            has_input = true;
            parse_failed |= !parse_input(value, input_width, input_height, input_batch, input_format);
        }else if (startswith(argv[i], "--output", &value)){
            has_output = true;
            parse_failed |= !parse_output(value, output_width, output_height, output_layout, output_dtype);
        }else if(startswith(argv[i], "--interp", &value)){
            parse_failed |= !parse_interp(value, interp);
        }else if(startswith(argv[i], "--perf", &value)){
            perf = true;
        }else if(startswith(argv[i], "--save", &value)){
            save = value;
        }else if(startswith(argv[i], "--help", &value)){
            help();
        }else{
            help();
        }
    }

    if(parse_failed || !has_input || !has_output) help();
    return run(perf, save, input_width, input_height, input_batch, input_format, output_width, output_height, output_layout, output_dtype, interp);
}