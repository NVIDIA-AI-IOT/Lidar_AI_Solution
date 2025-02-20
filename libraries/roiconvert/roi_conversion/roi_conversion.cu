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
#include <mutex>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "roi_conversion.hpp"

#define checkRuntime(call)  check_runtime(call, #call, __LINE__, __FILE__)
#define half2short(h)   (*(unsigned short*)&h)

namespace roiconv{

struct ProblemPerBlock{
    int output_x0, output_x1, output_y0, output_y1;
    unsigned int input_task_id;

    ProblemPerBlock() = default;
    ProblemPerBlock(int output_x0_, int output_y0_, int output_x1_, int output_y1_, unsigned int input_task_id_)
        :output_x0(output_x0_), output_x1(output_x1_), output_y0(output_y0_), output_y1(output_y1_), input_task_id(input_task_id_){}
};

typedef unsigned char uint8_t;

template<typename _T>struct AsUnion4{};
template<>struct AsUnion4<uint8_t>{typedef uchar4  type;};
template<>struct AsUnion4<float>  {typedef float4  type;};
template<>struct AsUnion4<__half> {typedef ushort4 type;};
template<>struct AsUnion4<int8_t> {typedef char4   type;};
template<>struct AsUnion4<int32_t> {typedef int4   type;};
template<>struct AsUnion4<uint32_t> {typedef uint4   type;};

template<typename _T>struct AsUnion3{};
template<>struct AsUnion3<uint8_t>{typedef uchar3  type;};
template<>struct AsUnion3<float>  {typedef float3  type;};
template<>struct AsUnion3<__half> {typedef ushort3 type;};
template<>struct AsUnion3<int8_t> {typedef char3   type;};
template<>struct AsUnion3<int32_t> {typedef int3   type;};
template<>struct AsUnion3<uint32_t> {typedef uint3   type;};

template<OutputDType _T>struct AsPODType{};
template<>struct AsPODType<OutputDType::Uint8>   {typedef uint8_t type;};
template<>struct AsPODType<OutputDType::Float32> {typedef float   type;};
template<>struct AsPODType<OutputDType::Float16> {typedef __half  type;};
// template<>struct AsPODType<OutputDType::Int8>    {typedef int8_t type;};
// template<>struct AsPODType<OutputDType::Int32>   {typedef int32_t type;};
// template<>struct AsPODType<OutputDType::Uint32>  {typedef uint32_t type;};

enum class Parallel : unsigned int{
    None        = 0,
    SinglePixel = 1,
    FourPixel   = 2
};

static __device__ __forceinline__ uchar3 make3(uint8_t v0, uint8_t v1, uint8_t v2){return make_uchar3(v0, v1, v2);}
// static __device__ __forceinline__ char3 make3(int8_t v0, int8_t v1, int8_t v2){return make_char3(v0, v1, v2);}
static __device__ __forceinline__ float3 make3(float v0, float v1, float v2){return make_float3(v0, v1, v2);}
static __device__ __forceinline__ ushort3 make3(__half v0, __half v1, __half v2){return make_ushort3(half2short(v0), half2short(v1), half2short(v2)); }
// static __device__ __forceinline__ int3 make3(int v0, int v1, int v2){return make_int3(v0, v1, v2); }
// static __device__ __forceinline__ uint3 make3(uint32_t v0, uint32_t v1, uint32_t v2){return make_uint3(v0, v1, v2); }

#define INTER_RESIZE_COEF_BITS 11
#define INTER_RESIZE_COEF_SCALE (1 << INTER_RESIZE_COEF_BITS)
#define CAST_BITS (INTER_RESIZE_COEF_BITS << 1)

template<typename _T>
static __forceinline__ __device__ _T limit(_T value, _T low, _T high){
    return value < low ? low : (value > high ? high : value);
}

template<typename _T>
static __device__ __forceinline__ uint8_t u8cast(_T value){
    return value < 0 ? 0 : (value >= 255 ? 255 : uint8_t(value));
}

template<typename _T>
static __device__ __forceinline__ int8_t s8cast(_T value){
    return value <= -128 ? -128 : (value >= 127 ? 127 : int8_t(value));
}

template<typename _T>struct Saturate{};
template<>struct Saturate<int8_t>   {__device__ __forceinline__ static int8_t cast(float x){return s8cast(x);}};
template<>struct Saturate<uint8_t>  {__device__ __forceinline__ static uint8_t cast(float x){return u8cast(x);}};
template<>struct Saturate<float>    {__device__ __forceinline__ static float cast(float x){return x;}};
template<>struct Saturate<__half>   {__device__ __forceinline__ static __half cast(float x){return __half(x);}};
template<>struct Saturate<int32_t>   {__device__ __forceinline__ static int32_t cast(float x){return int32_t(x);}};
template<>struct Saturate<uint32_t>   {__device__ __forceinline__ static uint32_t cast(float x){return uint32_t(x);}};

static bool __inline__ check_runtime(cudaError_t e, const char* call, int line, const char *file){
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime error %s # %s, code = %s [ %d ] in file %s:%d\n", call, cudaGetErrorString(e), cudaGetErrorName(e), e, file, line);
        return false;
    }
    return true;
}

template<typename T, OutputFormat _Layout>
struct DataWriter{};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////// NHWC RGB
template<typename T>
struct DataWriter<T, OutputFormat::HWC_RGB>{
    static __device__ __forceinline__ void call(T* pdst, T r, T g, T b, int x, int y, int width, int height){
        *(typename AsUnion3<T>::type*)(pdst + (y * width + x) * 3) = make3(r, g, b);
    }
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////// NHWC BGR
template<typename T>
struct DataWriter<T, OutputFormat::HWC_BGR>{
    static __device__ __forceinline__ void call(T* pdst, T r, T g, T b, int x, int y, int width, int height){
        *(typename AsUnion3<T>::type*)(pdst + (y * width + x) * 3) = make3(b, g, r);
    }
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////// CHW RGB
template<typename T>
struct DataWriter<T, OutputFormat::CHW_RGB>{
    static __device__ __forceinline__ void call(T* pdst, T r, T g, T b, int x, int y, int width, int height){
        *(pdst + ((0 * height + y) * width + x)) = r;
        *(pdst + ((1 * height + y) * width + x)) = g;
        *(pdst + ((2 * height + y) * width + x)) = b;
    }
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////// CHW BGR
template<typename T>
struct DataWriter<T, OutputFormat::CHW_BGR>{
    static __device__ __forceinline__ void call(T* pdst, T r, T g, T b, int x, int y, int width, int height){
        *(pdst + ((0 * height + y) * width + x)) = b;
        *(pdst + ((1 * height + y) * width + x)) = g;
        *(pdst + ((2 * height + y) * width + x)) = r;
    }
};

template<typename T>
struct DataWriter<T, OutputFormat::Gray>{
    static __device__ __forceinline__ void call(T* pdst, T r, T g, T b, int x, int y, int width, int height){
        *(pdst + y * width + x) = r;
    }
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////// CHWx RGB
template<typename T, int SC>
struct CHWxRGBWriter{
    static __device__ __forceinline__ void call(T* pdst, T r, T g, T b, int x, int y, int width, int height){

        // for CHW(S): tensor.view(N, C/S, S, H, W).transpose(0, 1, 3, 4, 2)
        *(typename AsUnion3<T>::type*)(pdst + (y * width + x) * SC) = make3(r, g, b);
    }
};

template<typename T, int SC>
struct CHWxBGRWriter{
    static __device__ __forceinline__ void call(T* pdst, T r, T g, T b, int x, int y, int width, int height){

        // for CHW(S): tensor.view(N, C/S, S, H, W).transpose(0, 1, 3, 4, 2)
        *(typename AsUnion3<T>::type*)(pdst + (y * width + x) * SC) = make3(b, g, r);
    }
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////// CHWx RGB/BGR
template<typename T>
struct DataWriter<T, OutputFormat::CHW16_RGB>{
    static __device__ __forceinline__ void call(T* pdst, T r, T g, T b, int x, int y, int width, int height){
        CHWxRGBWriter<T, 16>::call(pdst, r, g, b, x, y, width, height);
    }
};

template<typename T>
struct DataWriter<T, OutputFormat::CHW16_BGR>{
    static __device__ __forceinline__ void call(T* pdst, T r, T g, T b, int x, int y, int width, int height){
        CHWxBGRWriter<T, 16>::call(pdst, r, g, b, x, y, width, height);
    }
};

template<typename T>
struct DataWriter<T, OutputFormat::CHW32_RGB>{
    static __device__ __forceinline__ void call(T* pdst, T r, T g, T b, int x, int y, int width, int height){
        CHWxRGBWriter<T, 32>::call(pdst, r, g, b, x, y, width, height);
    }
};

template<typename T>
struct DataWriter<T, OutputFormat::CHW32_BGR>{
    static __device__ __forceinline__ void call(T* pdst, T r, T g, T b, int x, int y, int width, int height){
        CHWxBGRWriter<T, 32>::call(pdst, r, g, b, x, y, width, height);
    }
};

template<typename T>
struct DataWriter<T, OutputFormat::CHW4_RGB>{
    static __device__ __forceinline__ void call(T* pdst, T r, T g, T b, int x, int y, int width, int height){
        CHWxRGBWriter<T, 4>::call(pdst, r, g, b, x, y, width, height);
    }
};

template<typename T>
struct DataWriter<T, OutputFormat::CHW4_BGR>{
    static __device__ __forceinline__ void call(T* pdst, T r, T g, T b, int x, int y, int width, int height){
        CHWxBGRWriter<T, 4>::call(pdst, r, g, b, x, y, width, height);
    }
};
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static __device__ unsigned int __forceinline__ round_down2(unsigned int num){
    return num & (~1);
}

template<typename T, OutputFormat output_format>
struct Normalizer{
    static __device__ void __forceinline__ call(
        uint8_t ir, uint8_t ig, uint8_t ib, T& r, T& g, T& b,
        float alphas[3], float betas[3]
    ){
        r = Saturate<T>::cast(ir * alphas[0] + betas[0]);
        g = Saturate<T>::cast(ig * alphas[1] + betas[1]);
        b = Saturate<T>::cast(ib * alphas[2] + betas[2]);
    }
};

template<typename T>
struct Normalizer<T, OutputFormat::Gray>{
    static __device__ void __forceinline__ call(
        uint8_t ir, uint8_t ig, uint8_t ib, T& r, T& g, T& b,
        float alphas[3], float betas[3]
    ){
        // 0.299 ∙ Red + 0.587 ∙ Green + 0.114 ∙ Blue
        float gray = ir * 0.299f + ig * 0.587f + ib * 0.114f;
        r = Saturate<T>::cast(gray * alphas[0] + betas[0]);
    }
};

static __device__ void __forceinline__ yuv2rgb(
    int y, int u, int v, uint8_t& r, uint8_t& g, uint8_t& b
){
    int iyval = 1220542 * max(0, y - 16);
    r = u8cast((iyval + 1673527*(v - 128)                      + (1 << 19)) >> 20);
    g = u8cast((iyval - 852492*(v - 128) - 409993*(u - 128)    + (1 << 19)) >> 20);
    b = u8cast((iyval                      + 2116026*(u - 128) + (1 << 19)) >> 20);
}

template<InputFormat format>
static __device__ void __forceinline__ load_pixel(
    const void* const planes[3],
    int x, int y, int down_x, int width, int stride, uint8_t& r, uint8_t& g, uint8_t& b
);

template<>
__device__ void __forceinline__ load_pixel<InputFormat::RGBA>(
    const void* const planes[3],
    int x, int y, int down_x, int width, int stride, uint8_t& r, uint8_t& g, uint8_t& b
){
    uchar4 data = *(const uchar4*)((const char*)planes[0] + y * stride + x * 4);
    r = data.x;
    g = data.y;
    b = data.z;
}

template<>
__device__ void __forceinline__ load_pixel<InputFormat::RGB>(
    const void* const planes[3],
    int x, int y, int down_x, int width, int stride, uint8_t& r, uint8_t& g, uint8_t& b
){
    uchar3 data = *(const uchar3*)((const char*)planes[0] + y * stride + x * 3);
    r = data.x;
    g = data.y;
    b = data.z;
}

template<>
__device__ void __forceinline__ load_pixel<InputFormat::YUVI420Separated>(
    const void* const planes[3],
    int x, int y, int down_x, int width, int stride, uint8_t& r, uint8_t& g, uint8_t& b
){
    uint8_t yv = *((const unsigned char*)planes[0] + y * width + x);
    uint8_t uv = *((const unsigned char*)planes[1] + (y / 2) * (width / 2) + (x / 2));
    uint8_t vv = *((const unsigned char*)planes[2] + (y / 2) * (width / 2) + (x / 2));
    yuv2rgb(yv, uv, vv, r, g, b);
}

// BL sample pixel implmentation
template<>
__device__ void __forceinline__ load_pixel<InputFormat::NV12BlockLinear>(
    const void* const planes[3],
    int x, int y, int down_x, int width, int stride, uint8_t& r, uint8_t& g, uint8_t& b
){
    uint8_t yv = tex2D<uint8_t>((cudaTextureObject_t)planes[0],   x,          y    );
    // If chroma bytes per pixel = 1.
    // uint8_t uv = tex2D<uint8_t>((cudaTextureObject_t)chroma, down_x + 0, y / 2);
    // uint8_t vv = tex2D<uint8_t>((cudaTextureObject_t)chroma, down_x + 1, y / 2);
    // yuv2rgb(yv, uv, vv, r, g, b);

    // If chroma bytes per pixel = 2.
    uchar2 uv  = tex2D<uchar2>((cudaTextureObject_t)planes[1], x / 2, y / 2);
    yuv2rgb(yv, uv.x, uv.y, r, g, b);
}

// PL sample pixel implmentation
template<>
__device__ void __forceinline__ load_pixel<InputFormat::NV12PitchLinear>(
    const void* const planes[3],
    int x, int y, int down_x, int width, int stride, uint8_t& r, uint8_t& g, uint8_t& b
){
    uint8_t yv = *((const unsigned char*)planes[0] + y * stride + x);
    uint8_t uv = *((const unsigned char*)planes[1] + (y / 2) * stride + down_x + 0);
    uint8_t vv = *((const unsigned char*)planes[1] + (y / 2) * stride + down_x + 1);
    yuv2rgb(yv, uv, vv, r, g, b);
}

//     Y U Y V Y U Y V      Y Y Y Y Y Y      Y Y Y Y Y Y      Y Y Y Y Y Y      Y Y Y Y Y Y
//     Y U Y V Y U Y V      Y Y Y Y Y Y      Y Y Y Y Y Y      Y Y Y Y Y Y      Y Y Y Y Y Y
//     Y U Y V Y U Y V      Y Y Y Y Y Y      Y Y Y Y Y Y      Y Y Y Y Y Y      Y Y Y Y Y Y
//     Y U Y V Y U Y V      Y Y Y Y Y Y      Y Y Y Y Y Y      Y Y Y Y Y Y      Y Y Y Y Y Y
//     Y U Y V Y U Y V      U U U U U U      V V V V V V      U V U V U V      V U V U V U
//     Y U Y V Y U Y V      V V V V V V      U U U U U U      U V U V U V      V U V U V U
//        - YUYV -           - I420 -          - YV12 -         - NV12 -         - NV21 -

// YUV422Packed_YUYV sample pixel implmentation
template<>
__device__ void __forceinline__ load_pixel<InputFormat::YUV422Packed_YUYV>(
    const void* const planes[3],
    int x, int y, int down_x, int width, int stride, uint8_t& r, uint8_t& g, uint8_t& b
){
    // 0, 1, 2, 3, 4, 5, 6, 7
    // 0, 0, 2, 2, 4, 4, 6, 6
    // Y, U, Y, V, Y, U, Y, V
    // 0,    1,    2,    3
    uchar4 yuv = *(uchar4*)((const uint8_t*)planes[0] + y * stride + (x / 2) * 4);
    if(x == down_x){
        yuv2rgb(yuv.x, yuv.y, yuv.w, r, g, b);
    }else{
        yuv2rgb(yuv.z, yuv.y, yuv.w, r, g, b);
    }
}

template<InputFormat input_format, Interpolation interp>
struct LoadPixel{};

// BL sample pixel implmentation
template<InputFormat format>
struct LoadPixel<format, Interpolation::Nearest>{
    static __device__ void __forceinline__ call(
        float x, float y, 
        uint8_t& r, uint8_t& g, uint8_t& b, const Task& task
    ){
        // In some cases, the floating point precision will lead to miscalculation of the value, 
        // making the result not exactly match with opencv, 
        // so here you need to add eps as precision compensation
        //
        // A special case is when the input is 3840 and the output is 446, x = 223:
        // const int SCrc_x_double = 223.0  * (3840.0  / 446.0);            // -> 1920
        // const int SCrc_x_float  = 223.0f * (3840.0f / 446.0f);           // -> 1919
        // const int SCrc_x_float  = 223.0f * (3840.0f / 446.0f) + 1e-5;    // -> 1920
        //
        // !!! If you want to use the double for sx/sy, you'll get a 2x speed drop
        const float eps = 1e-5;
        int ix = static_cast<int>(x + eps);
        int iy = static_cast<int>(y + eps);
        if(ix >= task.x0 && ix < task.x1 && iy >= task.y0 && iy < task.y1){
            load_pixel<format>(task.input_planes, ix, iy, round_down2(ix), task.input_width, task.input_stride, r, g, b);
        }else{
            r = task.fillcolor[0]; g = task.fillcolor[1]; b = task.fillcolor[2];
        }
    }
};

template<InputFormat format>
struct LoadPixel<format, Interpolation::Bilinear>{
    static __device__ void __forceinline__ call(
        float x, float y, 
        uint8_t& r, uint8_t& g, uint8_t& b, const Task& task
    ){
        uint8_t rs[4], gs[4], bs[4];
        int x_low  = floorf(x);
        int y_low  = floorf(y);
        int x_high = x_low + 1;
        int y_high = y_low + 1;

        int ly = rint((y - y_low) * INTER_RESIZE_COEF_SCALE);
        int lx = rint((x - x_low) * INTER_RESIZE_COEF_SCALE);
        int hy = INTER_RESIZE_COEF_SCALE - ly;
        int hx = INTER_RESIZE_COEF_SCALE - lx;

        load_pixel<format>(task.input_planes, x_low,  y_low,  round_down2(x_low),  task.input_width, task.input_stride, rs[0], gs[0], bs[0]);
        if(x_high >= task.x0 && x_high < task.x1)
            load_pixel<format>(task.input_planes, x_high, y_low,  round_down2(x_high), task.input_width, task.input_stride, rs[1], gs[1], bs[1]);

        if(y_high >= task.y0 && y_high < task.y1)
            load_pixel<format>(task.input_planes, x_low,  y_high, round_down2(x_low),  task.input_width, task.input_stride, rs[2], gs[2], bs[2]);

        if(x_high >= task.x0 && x_high < task.x1 && y_high >= task.y0 && y_high < task.y1)
            load_pixel<format>(task.input_planes, x_high, y_high, round_down2(x_high), task.input_width, task.input_stride, rs[3], gs[3], bs[3]);

        r = ( ((hy * ((hx * rs[0] + lx * rs[1]) >> 4)) >> 16) + ((ly * ((hx * rs[2] + lx * rs[3]) >> 4)) >> 16) + 2 )>>2;
        g = ( ((hy * ((hx * gs[0] + lx * gs[1]) >> 4)) >> 16) + ((ly * ((hx * gs[2] + lx * gs[3]) >> 4)) >> 16) + 2 )>>2;
        b = ( ((hy * ((hx * bs[0] + lx * bs[1]) >> 4)) >> 16) + ((ly * ((hx * bs[2] + lx * bs[3]) >> 4)) >> 16) + 2 )>>2;
    }
};

template<InputFormat input_format, typename output_dtype, OutputFormat output_format, Interpolation interp>
static __global__ void convert_roi_kernel(Task* tasks, const int num_task, ProblemPerBlock* problems, const int num_problem){

    #pragma unroll
    for(int jk = 0; jk < 4; ++jk){
        int iproblem = blockIdx.x * 4 + jk;
        if(iproblem >= num_problem) continue;
        ProblemPerBlock problem = problems[iproblem];

        int x = threadIdx.x + problem.output_x0;
        int y = threadIdx.y + problem.output_y0;
        if(x >= problem.output_x1 || y >= problem.output_y1) continue;
        
        Task task = tasks[problem.input_task_id];
        float fx = x + 0.5f;
        float fy = y + 0.5f;
        float ifx = fx * task.affine_matrix[0] + fy * task.affine_matrix[1] + task.affine_matrix[2] + task.x0 - 0.5f;
        float ify = fx * task.affine_matrix[3] + fy * task.affine_matrix[4] + task.affine_matrix[5] + task.y0 - 0.5f;

        uint8_t ir = task.fillcolor[0], ig = task.fillcolor[1], ib = task.fillcolor[2];
        if(ifx >= task.x0 - 0.5f && ifx < task.x1 + 0.5f && ify >= task.y0 - 0.5f && ify < task.y1 + 0.5f){
            ifx = max(float(task.x0), min(task.x1 - 1.0f, ifx));
            ify = max(float(task.y0), min(task.y1 - 1.0f, ify));
            LoadPixel<input_format, interp>::call(ifx, ify, ir, ig, ib, task);
        }

        output_dtype r, g, b;
        Normalizer<output_dtype, output_format>::call(ir, ig, ib, r, g, b, task.alpha, task.beta);
        DataWriter<output_dtype, output_format>::call(
            (output_dtype*)task.output, r, g, b, x, y, task.output_width, task.output_height
        );
    }
}

template<InputFormat input_format, OutputDType out_dtype, OutputFormat layout, Interpolation interp>
static bool batched_convert_roi_impl(Task* tasks, const int num_task, ProblemPerBlock* problems, const int num_problem, cudaStream_t stream){
    using output_dtype = typename AsPODType<out_dtype>::type;

    // HxW per block
    dim3 dim_block(32, 32);
    dim3 dim_grid((num_problem + 3) / 4);
    convert_roi_kernel<input_format, output_dtype, layout, interp> <<<dim_grid, dim_block, 0, stream>>>(
        tasks, num_task, problems, num_problem
    );
    return checkRuntime(cudaPeekAtLastError());
}

typedef bool(*batched_convert_roi_impl_function)(Task* tasks, const int num_task, ProblemPerBlock* problems, const int num_problem, cudaStream_t stream);

// If you want to modify this part of the code, 
// please note that the order of the enumerated types must match the integer values of this type 
// (note: that the order starts from 1)

#define DefineInputFormat(...)                                                    \
    batched_convert_roi_impl<InputFormat::NV12BlockLinear, __VA_ARGS__>,   \
    batched_convert_roi_impl<InputFormat::NV12PitchLinear, __VA_ARGS__>,   \
    batched_convert_roi_impl<InputFormat::YUV422Packed_YUYV, __VA_ARGS__>, \
    batched_convert_roi_impl<InputFormat::YUVI420Separated, __VA_ARGS__>,  \
    batched_convert_roi_impl<InputFormat::RGBA, __VA_ARGS__>,              \
    batched_convert_roi_impl<InputFormat::RGB, __VA_ARGS__>,

#define DefineDType(...)                                               \
    DefineInputFormat(OutputDType::Uint8, __VA_ARGS__)                    \
    DefineInputFormat(OutputDType::Float32, __VA_ARGS__)                  \
    DefineInputFormat(OutputDType::Float16, __VA_ARGS__)                  
    // DefineInputFormat(OutputDType::Int8, __VA_ARGS__)                     \
    // DefineInputFormat(OutputDType::Int32, __VA_ARGS__)                    \
    // DefineInputFormat(OutputDType::Uint32, __VA_ARGS__)

#define DefineLayout(...)                                            \
    DefineDType(OutputFormat::CHW_RGB, __VA_ARGS__)                  \
    DefineDType(OutputFormat::CHW_BGR, __VA_ARGS__)                  \
    DefineDType(OutputFormat::HWC_RGB, __VA_ARGS__)                  \
    DefineDType(OutputFormat::HWC_BGR, __VA_ARGS__)                  \
    DefineDType(OutputFormat::CHW16_RGB, __VA_ARGS__)                \
    DefineDType(OutputFormat::CHW16_BGR, __VA_ARGS__)                \
    DefineDType(OutputFormat::CHW32_RGB, __VA_ARGS__)                \
    DefineDType(OutputFormat::CHW32_BGR, __VA_ARGS__)                \
    DefineDType(OutputFormat::CHW4_RGB, __VA_ARGS__)                 \
    DefineDType(OutputFormat::CHW4_BGR, __VA_ARGS__)                 \
    DefineDType(OutputFormat::Gray, __VA_ARGS__)

#define DefineInterp                                            \
    DefineLayout(Interpolation::Nearest)                        \
    DefineLayout(Interpolation::Bilinear)              

#define DefineAllFunction   DefineInterp

template<typename T>struct EnumCount{};
template<> struct EnumCount<InputFormat>{static const unsigned int value = 6;};
template<> struct EnumCount<OutputDType>{static const unsigned int value = 3;};
template<> struct EnumCount<OutputFormat>{static const unsigned int value = 11;};
template<> struct EnumCount<Interpolation>{static const unsigned int value = 2;};

static const batched_convert_roi_impl_function func_list[] = {
    DefineAllFunction
    nullptr
};

static bool batch_convert_rois(Task* tasks, const unsigned int num_task, ProblemPerBlock* problems, const int num_problem, InputFormat input_format, OutputDType output_dtype, OutputFormat output_format, Interpolation interpolation, cudaStream_t stream){
    int iformat = (int)input_format - 1;
    int odtype  = (int)output_dtype    - 1;
    int olayout = (int)output_format   - 1;
    int iinterp = (int)interpolation - 1;
    int index = ((iinterp * EnumCount<OutputFormat>::value + olayout) * EnumCount<OutputDType>::value + odtype) * EnumCount<InputFormat>::value + iformat;
    if(
        iformat < 0 || iformat >= EnumCount<InputFormat>::value ||
        odtype < 0  || odtype >= EnumCount<OutputDType>::value ||
        olayout < 0 || olayout >= EnumCount<OutputFormat>::value ||
        iinterp < 0 || iinterp >= EnumCount<Interpolation>::value ||
        index < 0   || index >= sizeof(func_list) / sizeof(func_list[0]) - 1
    ){
        fprintf(stderr, "Unsupported configure %d.\n", index);
        return false;
    }
    batched_convert_roi_impl_function func = func_list[index];
    return func(tasks, num_task, problems, num_problem, stream);
}

template<typename T>
class PinnedMemoryAlloctor{
public:
    typedef T value_type;
 
    PinnedMemoryAlloctor() = default;

    template<class U>
    constexpr PinnedMemoryAlloctor(const PinnedMemoryAlloctor<U>&) noexcept {}
 
    [[nodiscard]] T* allocate(std::size_t n){
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T))
            throw std::bad_array_new_length();

        T* p = nullptr;
        if (checkRuntime(cudaMallocHost(&p, sizeof(T) * n))){
            return p;
        }
        throw std::bad_alloc();
        return nullptr;
    }
 
    void deallocate(T* p, std::size_t n) noexcept{
        checkRuntime(cudaFreeHost(p));
    }
};

static void inverse_affine_matrix(const float src[6], float dst[6]){
    float a = src[0], b = src[1], c = src[2];
    float d = src[3], e = src[4], f = src[5];
    float r = a * e - b * d;
    if(r != 0) r = 1.0f / r;
    dst[0] = e * r; dst[1] = -b * r; dst[2] = (b * f - c * e) * r;
    dst[3] = -d * r; dst[4] = a * r; dst[5] = -(a * f - c * d) * r;
}

void Task::resize_affine(){
    float roi_width  = x1 - x0;
    float roi_height = y1 - y0;
    memset(affine_matrix, 0, sizeof(affine_matrix));
    affine_matrix[0] = output_width / roi_width;
    affine_matrix[4] = output_height / roi_height;
}

void Task::center_resize_affine(){
    float roi_width  = x1 - x0;
    float roi_height = y1 - y0;
    float sx = output_width / roi_width;
    float sy = output_height / roi_height;
    float min_scale = std::min(sx, sy);
    memset(affine_matrix, 0, sizeof(affine_matrix));
    affine_matrix[0] = min_scale;
    affine_matrix[2] = (output_width - roi_width * min_scale) * 0.5f;
    affine_matrix[4] = min_scale;
    affine_matrix[5] = (output_height - roi_height * min_scale) * 0.5f;
}

class ROIConversionImpl : public ROIConversion{
public:
    ROIConversionImpl(){
        tasks_.reserve(256);
        tasks_cached_.reserve(256);
        problems_.reserve(4096 * 8);
        problems_cached_.reserve(4096 * 8);
    }

    virtual ~ROIConversionImpl(){
        if(gpu_tasks_ != nullptr){
            checkRuntime(cudaFree(gpu_tasks_));
            gpu_tasks_ = nullptr;
            gpu_tasks_capacity_ = 0;
        }

        if(gpu_problems_ != nullptr){
            checkRuntime(cudaFree(gpu_problems_));
            gpu_problems_ = nullptr;
            gpu_problems_capacity_ = 0;
        }
        tasks_cached_.clear();
        tasks_.clear();

        problems_cached_.clear();
        problems_.clear();
    }

    virtual void add(const Task& task) override{
        tasks_.emplace_back(task);
    }

    void convert_to_problems(){
        const int x_block_size = 32;
        const int y_block_size = 32;
        unsigned int total_problem_size_added = 0;
        for(unsigned int i = 0; i < (unsigned int)tasks_.size(); ++i){
            auto& task = tasks_[i];
            inverse_affine_matrix(task.affine_matrix, task.affine_matrix);

            int xmin = std::min(task.x0, task.x1);
            int xmax = std::max(task.x0, task.x1);
            int ymin = std::min(task.y0, task.y1);
            int ymax = std::max(task.y0, task.y1);
            task.x0 = std::max(0, std::min(xmin, task.input_width - 1));
            task.y0 = std::max(0, std::min(ymin, task.input_height - 1));
            task.x1 = std::max(0, std::min(xmax, task.input_width));
            task.y1 = std::max(0, std::min(ymax, task.input_height));

            const int ngrid_x = (task.output_width + x_block_size - 1) / x_block_size;
            const int ngrid_y = (task.output_height + y_block_size - 1) / y_block_size;
            total_problem_size_added += ngrid_x * ngrid_y;
        }
        problems_.resize(total_problem_size_added);

        unsigned int index_problem = 0;
        for(unsigned int i = 0; i < (unsigned int)tasks_.size(); ++i){
            auto& task = tasks_[i];
            const int ngrid_x = (task.output_width + x_block_size - 1) / x_block_size;
            const int ngrid_y = (task.output_height + y_block_size - 1) / y_block_size;

            for(int iy = 0; iy < ngrid_y; ++iy){
                for(int ix = 0; ix < ngrid_x; ++ix){
                    const int x0 = ix * x_block_size;
                    const int y0 = iy * y_block_size;
                    problems_[index_problem++] = ProblemPerBlock(
                        x0, y0,
                        std::min(x0 + x_block_size, task.output_width),
                        std::min(y0 + y_block_size, task.output_height),
                        i
                    );
                }
            }
        }
    }

    bool data_preparation(void* stream, bool clear){
        if(tasks_.empty()) return true;

        this->convert_to_problems();
        if(problems_.empty()){
            tasks_.clear();
            return false;
        }

        {
            if(gpu_tasks_capacity_ < tasks_.size()){
                if(!checkRuntime(cudaFree(gpu_tasks_))) return false;

                gpu_tasks_capacity_ = std::max(size_t((double)tasks_.size() * 1.2), size_t(256));
                if(!checkRuntime(cudaMalloc(&gpu_tasks_, gpu_tasks_capacity_ * sizeof(Task)))) return false;
            }

            if(gpu_problems_capacity_ < problems_.size()){
                if(!checkRuntime(cudaFree(gpu_problems_))) return false;

                gpu_problems_capacity_ = std::max(size_t((double)problems_.size() * 1.2), size_t(4096));
                if(!checkRuntime(cudaMalloc(&gpu_problems_, gpu_problems_capacity_ * sizeof(ProblemPerBlock)))) return false;
            }
            
            tasks_cached_.resize(tasks_.size());
            memcpy(tasks_cached_.data(), tasks_.data(), sizeof(Task) * tasks_.size());
            
            problems_cached_.resize(problems_.size());
            memcpy(problems_cached_.data(), problems_.data(), sizeof(ProblemPerBlock) * problems_.size());

            if(clear){
                tasks_.clear();
                problems_.clear();
            }
        };

        if(!checkRuntime(cudaMemcpyAsync(gpu_tasks_, tasks_cached_.data(), sizeof(Task) * tasks_cached_.size(), cudaMemcpyHostToDevice, static_cast<cudaStream_t>(stream)))){
            return false;
        }
        return checkRuntime(cudaMemcpyAsync(gpu_problems_, problems_cached_.data(), sizeof(ProblemPerBlock) * problems_cached_.size(), cudaMemcpyHostToDevice, static_cast<cudaStream_t>(stream)));
    }

    virtual bool run(InputFormat input_format, OutputDType output_dtype, OutputFormat output_format, Interpolation interpolation, void* stream, bool sync, bool clear) override{
        if(!this->data_preparation(stream, clear)) return false;
        
        bool ok = batch_convert_rois(
            gpu_tasks_, (unsigned int)tasks_cached_.size(), 
            gpu_problems_, (unsigned int)problems_cached_.size(),
            input_format, output_dtype, output_format, interpolation, 
            static_cast<cudaStream_t>(stream)
        );
        if(!ok) return false;

        if(sync){
            ok = checkRuntime(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)));
        }
        return ok;
    }
    
private:
    std::vector<Task, PinnedMemoryAlloctor<Task>> tasks_cached_;
    std::vector<Task> tasks_;
    std::vector<ProblemPerBlock, PinnedMemoryAlloctor<ProblemPerBlock>> problems_cached_;
    std::vector<ProblemPerBlock> problems_;
    Task* gpu_tasks_ = nullptr;
    size_t gpu_tasks_capacity_ = 0;
    ProblemPerBlock* gpu_problems_ = nullptr;
    size_t gpu_problems_capacity_ = 0;
};

std::shared_ptr<ROIConversion> create(){
    std::shared_ptr<ROIConversionImpl> output(new ROIConversionImpl());
    return output;
}

}; // namespace roiconv
