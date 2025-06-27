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

#include <stdint.h>
#include <stdio.h>
#include <fstream>
#include <cuda_runtime.h>
#include "gpu_image.h"
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "cuosd.h"

namespace gpu{

    #define checkRuntime(call)  check_runtime(call, #call, __LINE__, __FILE__)

    static bool inline check_runtime(cudaError_t e, const char* call, int line, const char *file) {
        if (e != cudaSuccess) {
            fprintf(stderr, "CUDA Runtime error %s # %s, code = %s [ %d ] in file %s:%d\n", call, cudaGetErrorString(e), cudaGetErrorName(e), e, file, line);
            return false;
        }
        return true;
    }

    template<typename _T>
    static __host__ __device__ __forceinline__ unsigned char u8cast(_T value) {
        return value < 0 ? 0 : (value > 255 ? 255 : value);
    }

    static __host__ __device__ unsigned int __forceinline__ round_down2(unsigned int num) {
        return num & (~1);
    }

    static void rgb2yuv(unsigned char r, unsigned char g, unsigned char b, unsigned char& y, unsigned char& u, unsigned char& v) {

        y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
        u = ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
        v = ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
    }

    static void __host__ __device__ __forceinline__ yuv2rgb(unsigned char y, unsigned char u, unsigned char v, unsigned char& r, unsigned char& g, unsigned char& b) {

        int c = ((int)y - 16) * 298;
        int d =  (int)u - 128;
        int e =  (int)v - 128;
        r = u8cast(( (c + 409 * e + 128) ) >> 8);
        g = u8cast(( (c - 100 * d - 208 * e + 128) ) >> 8);
        b = u8cast(( (c + 516 * d + 128) ) >> 8);
    }

    const char* image_format_name(ImageFormat format) {
        switch(format) {
        case ImageFormat::RGB: return "RGB";
        case ImageFormat::RGBA: return "RGBA";
        case ImageFormat::BlockLinearNV12: return "BlockLinearNV12";
        case ImageFormat::PitchLinearNV12: return "PitchLinearNV12";
        default: return "UnknowImageFormat";
        }
    }

    // Create image using size and format
    Image* create_image(int width, int height, ImageFormat format) {

        Image* output  = new Image();
        output->width  = width;
        output->height = height;
        output->format = format;

        if (format == ImageFormat::RGB) {
            output->stride = output->width * 3;
            checkRuntime(cudaMalloc(&output->data0, output->stride * output->height));
        } else if (format == ImageFormat::RGBA) {
            output->stride = output->width * 4;
            checkRuntime(cudaMalloc(&output->data0, output->stride * output->height));
        } else if (format == ImageFormat::PitchLinearNV12) {
            output->stride = output->width;
            if (output->width % 2 != 0 || output->height % 2 != 0) {
                fprintf(stderr, "Invalid image size(%d, %d) for NV12\n", output->width, output->height);
                delete output;
                return nullptr;
            }
            checkRuntime(cudaMalloc(&output->data0, output->stride * output->height));
            checkRuntime(cudaMalloc(&output->data1, output->stride * output->height / 2));
        } else if (format == ImageFormat::BlockLinearNV12) {
            output->stride = output->width;
            if (output->width % 2 != 0 || output->height % 2 != 0) {
                fprintf(stderr, "Invalid image size(%d, %d) for NV12\n", output->width, output->height);
                delete output;
                return nullptr;
            }
            cudaChannelFormatDesc planeDesc = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
            checkRuntime(cudaMallocArray((cudaArray_t*)&output->reserve0, &planeDesc, output->stride, height));
            checkRuntime(cudaMallocArray((cudaArray_t*)&output->reserve1, &planeDesc, output->stride, height / 2));

            cudaResourceDesc luma_desc = {};
            luma_desc.resType         = cudaResourceTypeArray;
            luma_desc.res.array.array = (cudaArray_t)output->reserve0;
            checkRuntime(cudaCreateSurfaceObject((cudaSurfaceObject_t*)&output->data0, &luma_desc));

            cudaResourceDesc chroma_desc = {};
            chroma_desc.resType         = cudaResourceTypeArray;
            chroma_desc.res.array.array = (cudaArray_t)output->reserve1;
            checkRuntime(cudaCreateSurfaceObject((cudaSurfaceObject_t*)&output->data1, &chroma_desc));
        } else {
            fprintf(stderr, "Unsupport format %d\n", (int)format);
            delete output;
            output = nullptr;
        }
        return output;
    }

    Segment* create_segment() {
        Segment* output  = new Segment();
        output->width = 10; output->height = 10;
        checkRuntime(cudaMalloc(&output->data, output->width * output->height * sizeof(float)));
        std::vector<float> diamond;
        diamond.insert(diamond.end(), { 0,  0,  0,  0, 0.2,0.2, 0,  0,  0,  0 });
        diamond.insert(diamond.end(), { 0,  0,  0, 0.2,0.3,0.3,0.2, 0,  0,  0 });
        diamond.insert(diamond.end(), { 0,  0, 0.2,0.3,0.4,0.4,0.3,0.2, 0,  0 });
        diamond.insert(diamond.end(), { 0, 0.2,0.3,0.4,0.5,0.5,0.4,0.3,0.2, 0 });
        diamond.insert(diamond.end(), {0.2,0.3,0.4,0.5,0.5,0.5,0.5,0.4,0.3,0.2});
        diamond.insert(diamond.end(), {0.2,0.3,0.4,0.5,0.5,0.5,0.5,0.4,0.3,0.2});
        diamond.insert(diamond.end(), { 0, 0.2,0.3,0.4,0.5,0.5,0.4,0.3,0.2, 0 });
        diamond.insert(diamond.end(), { 0,  0, 0.2,0.3,0.4,0.4,0.3,0.2, 0,  0 });
        diamond.insert(diamond.end(), { 0,  0,  0, 0.2,0.3,0.3,0.2, 0,  0,  0 });
        diamond.insert(diamond.end(), { 0,  0,  0,  0, 0.2,0.2, 0,  0,  0,  0 });
        checkRuntime(cudaMemcpy(output->data, diamond.data(), output->width * output->height * sizeof(float), cudaMemcpyHostToDevice));
        return output;
    }

    void free_segment(Segment* segment) {
        if (segment->data) {
            checkRuntime(cudaFree(segment->data));
        }
        segment->width = 0;
        segment->height = 0;
    }

    Polyline* create_polyline() {
        Polyline* output = new Polyline();
        std::vector<Point> points;
        // points.push_back(Point({ 100, 200 }));
        // points.push_back(Point({ 600, 100 }));
        // points.push_back(Point({ 350, 300 }));
        // points.push_back(Point({ 600, 500 }));
        // points.push_back(Point({ 300, 500 }));
        points = {{ 20, 600 }, { 100, 100 }, {100, 300}, {90, 300}, {90, 320}, {150, 320}, {150, 300}, {130, 300}, {130, 100}, { 500, 20 }, { 600, 600 }};

        output->n_pts = points.size();
        output->h_pts = (int *)malloc(output->n_pts * 2 * sizeof(int));
        memcpy(output->h_pts, points.data(), output->n_pts * 2 * sizeof(int));
        checkRuntime(cudaMalloc(&output->d_pts, output->n_pts * 2 * sizeof(int)));
        checkRuntime(cudaMemcpy(output->d_pts, points.data(), output->n_pts * 2 * sizeof(int), cudaMemcpyHostToDevice));
        return output;
    }

    void free_polyline(Polyline* polyline) {
        if (polyline->d_pts) {
            checkRuntime(cudaFree(polyline->d_pts));
        }
        if (polyline->h_pts) {
            free(polyline->h_pts);
        }
        polyline->n_pts = 0;
    }

    static __global__ void set_yuv_pl_color(unsigned char* ydata, unsigned char* uvdata, int w, int line, int h, unsigned char y, unsigned char u, unsigned char v) {

        int ix = (blockDim.x * blockIdx.x + threadIdx.x) * 2;
        int iy = (blockDim.y * blockIdx.y + threadIdx.y) * 2;

        if (ix >= w-1 || iy >= h-1) return;

        ydata[(iy + 0) * line + ix + 0] = y;
        ydata[(iy + 0) * line + ix + 1] = y;
        ydata[(iy + 1) * line + ix + 0] = y;
        ydata[(iy + 1) * line + ix + 1] = y;
        uvdata[(iy / 2) * line + ix + 0] = u;
        uvdata[(iy / 2) * line + ix + 1] = v;
    }

    static __global__ void set_yuv_bl_color(cudaSurfaceObject_t ydata, cudaSurfaceObject_t uvdata, int w, int line, int h, unsigned char y, unsigned char u, unsigned char v) {

        int ix = (blockDim.x * blockIdx.x + threadIdx.x) * 2;
        int iy = (blockDim.y * blockIdx.y + threadIdx.y) * 2;

        if (ix >= w-1 || iy >= h-1) return;

        surf2Dwrite<unsigned char>(y, ydata, ix, iy);
        surf2Dwrite<unsigned char>(y, ydata, ix+1, iy);
        surf2Dwrite<unsigned char>(y, ydata, ix, iy+1);
        surf2Dwrite<unsigned char>(y, ydata, ix+1, iy+1);
        surf2Dwrite<unsigned char>(u, uvdata, ix, iy/2);
        surf2Dwrite<unsigned char>(v, uvdata, ix+1, iy/2);
    }

    static __global__ void set_rgba_color(unsigned char* pdata, int w, int line, int h, unsigned char r, unsigned char g, unsigned char b, unsigned char a) {

        int ix = (blockDim.x * blockIdx.x + threadIdx.x);
        int iy = (blockDim.y * blockIdx.y + threadIdx.y);

        if (ix >= w || iy >= h) return;

        pdata[iy * line + ix * 4 + 0] = r;
        pdata[iy * line + ix * 4 + 1] = g;
        pdata[iy * line + ix * 4 + 2] = b;
        pdata[iy * line + ix * 4 + 3] = a;
    }

    static __global__ void mask_rgba_alpha(unsigned char* pdata, int w, int h, unsigned char r, unsigned char g, unsigned char b, unsigned char a) {

        int ix = (blockDim.x * blockIdx.x + threadIdx.x);
        int iy = (blockDim.y * blockIdx.y + threadIdx.y);

        if (ix >= w || iy >= h) return;

        if (pdata[(iy * w + ix) * 4 + 0] == r &&
            pdata[(iy * w + ix) * 4 + 1] == g &&
            pdata[(iy * w + ix) * 4 + 2] == b) {
            pdata[(iy * w + ix) * 4 + 3] = 0;
        }
        else {
            pdata[(iy * w + ix) * 4 + 3] = a;
        }
    }

    static __global__ void set_rgb_color(unsigned char* pdata, int w, int line, int h, unsigned char r, unsigned char g, unsigned char b) {

        int ix = (blockDim.x * blockIdx.x + threadIdx.x);
        int iy = (blockDim.y * blockIdx.y + threadIdx.y);

        if (ix >= w || iy >= h) return;

        pdata[iy * line + ix * 3 + 0] = r;
        pdata[iy * line + ix * 3 + 1] = g;
        pdata[iy * line + ix * 3 + 2] = b;
    }

    void set_color(Image* image, unsigned char r, unsigned char g, unsigned char b, unsigned char a, void* _stream) {

        cudaStream_t stream = (cudaStream_t)_stream;
        if (image->format == ImageFormat::RGB) {

            dim3 block(32, 32);
            dim3 grid((image->width + block.x - 1) / block.x, (image->height + block.y - 1) / block.y);
            set_rgb_color<<<grid, block, 0, stream>>>((unsigned char*)image->data0, image->width, image->stride, image->height, r, g, b);
        } else if (image->format == ImageFormat::RGBA) {

            dim3 block(32, 32);
            dim3 grid((image->width + block.x - 1) / block.x, (image->height + block.y - 1) / block.y);
            set_rgba_color<<<grid, block, 0, stream>>>((unsigned char*)image->data0, image->width, image->stride, image->height, r, g, b, a);
        } else if (image->format == ImageFormat::PitchLinearNV12) {

            dim3 block(32, 32);
            dim3 grid(((image->width + 1) / 2 + block.x - 1) / block.x, ((image->height + 1) / 2 + block.y - 1) / block.y);
            rgb2yuv(r, g, b, r, g, b);
            set_yuv_pl_color<<<grid, block, 0, stream>>>((unsigned char*)image->data0, (unsigned char*)image->data1, image->width, image->stride, image->height, r, g, b);
        } else if (image->format == ImageFormat::BlockLinearNV12) {

            dim3 block(32, 32);
            dim3 grid(((image->width + 1) / 2 + block.x - 1) / block.x, ((image->height + 1) / 2 + block.y - 1) / block.y);
            rgb2yuv(r, g, b, r, g, b);
            set_yuv_bl_color<<<grid, block, 0, stream>>>((cudaSurfaceObject_t)image->data0, (cudaSurfaceObject_t)image->data1, image->width, image->stride, image->height, r, g, b);
        }
    }

    static Image* load_yuvnv12(const char* file, int width, int height, cudaStream_t stream) {

        if ((width % 2 != 0) || (height % 2 != 0)) {
            fprintf(stderr, "Unsupport resolution. %d x %d\n", width, height);
            return nullptr;
        }

        std::fstream infile(file, std::ios::binary | std::ios::in);
        if (!infile) {
            fprintf(stderr, "Failed to open: %s\n", file);
            return nullptr;
        }

        infile.seekg(0, std::ios::end);

        // check yuv size
        size_t file_size   = infile.tellg();
        size_t y_area      = width * height;
        size_t except_size = y_area * 3 / 2;
        if (file_size != except_size) {
            fprintf(stderr, "Wrong size of yuv image : %lu bytes, expected %lu bytes\n", file_size, except_size);
            return nullptr;
        }

        unsigned char* host_memory = nullptr;
        checkRuntime(cudaMallocHost(&host_memory, except_size));

        infile.seekg(0, std::ios::beg);
        if (!infile.read((char*)host_memory, except_size).good()) {
            fprintf(stderr, "Failed to read %lu byte data\n", y_area);
            checkRuntime(cudaFreeHost(host_memory));
            return nullptr;
        }

        Image* output = create_image(width, height, ImageFormat::PitchLinearNV12);
        if (output == nullptr) {
            checkRuntime(cudaFreeHost(host_memory));
            return nullptr;
        }

        checkRuntime(cudaMemcpyAsync(output->data0, host_memory,                  width * height,     cudaMemcpyHostToDevice, stream));
        checkRuntime(cudaMemcpyAsync(output->data1, host_memory + width * height, width * height / 2, cudaMemcpyHostToDevice, stream));
        checkRuntime(cudaStreamSynchronize(stream));
        checkRuntime(cudaFreeHost(host_memory));
        return output;
    }

    static __global__ void copy_nv12_to_pl(
        unsigned char* ydata, unsigned char* uvdata, int dst_x, int dst_y, int dst_w, int dst_h, int dst_width, int dst_height,
        unsigned char* nv12_y, unsigned char* nv12_uv, int nv12_w, int nv12_h
    ) {
        int ix = (blockDim.x * blockIdx.x + threadIdx.x) + dst_x;
        int iy = (blockDim.y * blockIdx.y + threadIdx.y) + dst_y;
        if (ix >= dst_w || iy >= dst_h) return;

        int nx = ix * nv12_w / (float)dst_w;
        int ny = iy * nv12_h / (float)dst_h;
        unsigned char value_y = nv12_y [(ny + 0) * nv12_w + nx    ];
        unsigned char value_u = nv12_uv[(ny / 2) * nv12_w + round_down2(nx) + 0];
        unsigned char value_v = nv12_uv[(ny / 2) * nv12_w + round_down2(nx) + 1];

        ix += dst_x;
        iy += dst_y;
        ydata [(iy + 0) * dst_width + ix + 0] = value_y;
        uvdata[(iy / 2) * dst_width + round_down2(ix) + 0] = value_u;
        uvdata[(iy / 2) * dst_width + round_down2(ix) + 1] = value_v;
    }

    static __global__ void copy_nv12_to_bl(
        cudaSurfaceObject_t ydata, cudaSurfaceObject_t uvdata, int dst_x, int dst_y, int dst_w, int dst_h, int dst_width, int dst_height,
        unsigned char* nv12_y, unsigned char* nv12_uv, int nv12_w, int nv12_h
    ) {
        int ix = (blockDim.x * blockIdx.x + threadIdx.x);
        int iy = (blockDim.y * blockIdx.y + threadIdx.y);
        if (ix >= dst_w || iy >= dst_h) return;

        int nx = ix * nv12_w / (float)dst_w;
        int ny = iy * nv12_h / (float)dst_h;
        unsigned char value_y = nv12_y [(ny + 0) * nv12_w + nx    ];
        unsigned char value_u = nv12_uv[(ny / 2) * nv12_w + round_down2(nx) + 0];
        unsigned char value_v = nv12_uv[(ny / 2) * nv12_w + round_down2(nx) + 1];

        ix += dst_x;
        iy += dst_y;
        surf2Dwrite<unsigned char>(value_y, ydata, ix, iy);
        surf2Dwrite<unsigned char>(value_u, uvdata, round_down2(ix), iy/2);
        surf2Dwrite<unsigned char>(value_v, uvdata, round_down2(ix)+1, iy/2);
    }

    static __global__ void copy_nv12_to_rgba(
        unsigned char* rgba, int dst_x, int dst_y, int dst_w, int dst_h, int dst_width, int dst_height,
        unsigned char* nv12_y, unsigned char* nv12_uv, int nv12_w, int nv12_h, unsigned char yuvalpha
    ) {
        int ix = (blockDim.x * blockIdx.x + threadIdx.x);
        int iy = (blockDim.y * blockIdx.y + threadIdx.y);

        if (ix >= dst_w || iy >= dst_h) return;

        int nx = ix * nv12_w / (float)dst_w;
        int ny = iy * nv12_h / (float)dst_h;
        unsigned char value_y = nv12_y [(ny + 0) * nv12_w + nx    ];
        unsigned char value_u = nv12_uv[(ny / 2) * nv12_w + round_down2(nx) + 0];
        unsigned char value_v = nv12_uv[(ny / 2) * nv12_w + round_down2(nx) + 1];

        ix += dst_x;
        iy += dst_y;
        yuv2rgb(value_y, value_u, value_v, rgba[(iy * dst_width + ix) * 4 + 0], rgba[(iy * dst_width + ix) * 4 + 1], rgba[(iy * dst_width + ix) * 4 + 2]);
        rgba[(iy * dst_width + ix) * 4 + 3] = yuvalpha;
    }

    static __global__ void copy_nv12_to_rgb(
        unsigned char* rgb, int dst_x, int dst_y, int dst_w, int dst_h, int dst_width, int dst_height,
        unsigned char* nv12_y, unsigned char* nv12_uv, int nv12_w, int nv12_h
    ) {
        int ix = (blockDim.x * blockIdx.x + threadIdx.x);
        int iy = (blockDim.y * blockIdx.y + threadIdx.y);

        if (ix >= dst_w || iy >= dst_h) return;

        int nx = ix * nv12_w / (float)dst_w;
        int ny = iy * nv12_h / (float)dst_h;
        unsigned char value_y = nv12_y [(ny + 0) * nv12_w + nx    ];
        unsigned char value_u = nv12_uv[(ny / 2) * nv12_w + round_down2(nx) + 0];
        unsigned char value_v = nv12_uv[(ny / 2) * nv12_w + round_down2(nx) + 1];

        ix += dst_x;
        iy += dst_y;
        yuv2rgb(value_y, value_u, value_v, rgb[(iy * dst_width + ix) * 3 + 0], rgb[(iy * dst_width + ix) * 3 + 1], rgb[(iy * dst_width + ix) * 3 + 2]);
    }

    void mask_rgba_alpha(Image* image, unsigned char r, unsigned char g, unsigned char b, unsigned char a, void* _stream) {
        cudaStream_t stream = (cudaStream_t)_stream;
        if (image->data0 == nullptr) return;

        if (image->format == ImageFormat::RGBA) {

            dim3 block(32, 32);
            dim3 grid((image->width + block.x - 1) / block.x, (image->height + block.y - 1) / block.y);
            mask_rgba_alpha<<<grid, block, 0, stream>>>((unsigned char*)image->data0, image->width, image->height, r, g, b, a);
        }
        else {
            fprintf(stderr, "This API is only used for RGBA image.\n");
        }
        checkRuntime(cudaStreamSynchronize(stream));
    }

    void copy_yuvnv12_to(Image* image, int dst_x, int dst_y, int dst_w, int dst_h, const char* yuvnv12file, int yuvwidth, int yuvheight, unsigned char yuvalpha, void* _stream) {

        cudaStream_t stream = (cudaStream_t)_stream;
        Image* yuv = load_yuvnv12(yuvnv12file, yuvwidth, yuvheight, stream);
        if (yuv == nullptr) return;

        if (image->format == ImageFormat::RGB) {

            dim3 block(32, 32);
            dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y);
            copy_nv12_to_rgb<<<grid, block, 0, stream>>>(
                (unsigned char*)image->data0, dst_x, dst_y, dst_w, dst_h, image->width, image->height,
                (unsigned char*)yuv->data0, (unsigned char*)yuv->data1, yuv->width, yuv->height
            );
        } else if (image->format == ImageFormat::RGBA) {

            dim3 block(32, 32);
            dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y);
            copy_nv12_to_rgba<<<grid, block, 0, stream>>>(
                (unsigned char*)image->data0, dst_x, dst_y, dst_w, dst_h, image->width, image->height,
                (unsigned char*)yuv->data0, (unsigned char*)yuv->data1, yuv->width, yuv->height, yuvalpha
            );
        } else if (image->format == ImageFormat::PitchLinearNV12) {

            dim3 block(32, 32);
            dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y);
            copy_nv12_to_pl<<<grid, block, 0, stream>>>(
                (unsigned char*)image->data0, (unsigned char*)image->data1, dst_x, dst_y, dst_w, dst_h, image->width, image->height,
                (unsigned char*)yuv->data0, (unsigned char*)yuv->data1, yuv->width, yuv->height
            );
        } else if (image->format == ImageFormat::BlockLinearNV12) {

            dim3 block(32, 32);
            dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y);
            copy_nv12_to_bl<<<grid, block, 0, stream>>>(
                (cudaSurfaceObject_t)image->data0, (cudaSurfaceObject_t)image->data1, dst_x, dst_y, dst_w, dst_h, image->width, image->height,
                (unsigned char*)yuv->data0, (unsigned char*)yuv->data1, yuv->width, yuv->height
            );
        }
        checkRuntime(cudaStreamSynchronize(stream));
        free_image(yuv);
    }

    void convert_nv12_to_rgb(const unsigned char* nv12, unsigned char* rgb, int width, int height);
    void convert_nv12_to_rgb(const unsigned char* nv12, unsigned char* rgb, int width, int height) {

        for (unsigned int y = 0; y < (unsigned int)height; ++y) {

            const uint8_t* yptr  = nv12 + y * width;
            const uint8_t* uvptr = nv12 + width * height + (y >> 1) * width;
            uint8_t* rgbptr      = rgb + y * width * 3;
            for (unsigned int x = 0; x < (unsigned int)width; ++x, ++yptr, rgbptr += 3) {

                unsigned int mod_x = x & (~1);
                short c = (short)yptr[0] - 16;
                short d = (short)uvptr[mod_x + 0] - 128;
                short e = (short)uvptr[mod_x + 1] - 128;
                rgbptr[0] = u8cast((298 * c + 409 * e + 128) >> 8);
                rgbptr[1] = u8cast((298 * c - 100 * d - 208 * e + 128) >> 8);
                rgbptr[2] = u8cast((298 * c + 516 * d + 128) >> 8);
            }
        }
    }

    // Save image to file, file format is png if rgba, otherwise jpg
    bool save_image(Image* image, const char* file, void* _stream) {

        cudaStream_t stream = (cudaStream_t)_stream;
        if (image->format == ImageFormat::RGB) {
            unsigned char* pdata = nullptr;
            checkRuntime(cudaMallocHost(&pdata, image->stride * image->height));
            checkRuntime(cudaMemcpyAsync(pdata, image->data0, image->stride * image->height, cudaMemcpyDeviceToHost, stream));
            checkRuntime(cudaStreamSynchronize(stream));
            stbi_write_jpg(file, image->width, image->height, 3, pdata, 100);
            checkRuntime(cudaFreeHost(pdata));
            return true;
        } else if (image->format == ImageFormat::RGBA) {
            unsigned char* pdata = nullptr;
            checkRuntime(cudaMallocHost(&pdata, image->stride * image->height));
            checkRuntime(cudaMemcpyAsync(pdata, image->data0, image->stride * image->height, cudaMemcpyDeviceToHost, stream));
            checkRuntime(cudaStreamSynchronize(stream));
            stbi_write_png(file, image->width, image->height, 4, pdata, image->stride);
            checkRuntime(cudaFreeHost(pdata));
            return true;
        } else if (image->format == ImageFormat::PitchLinearNV12) {
            unsigned char* pdata   = nullptr;
            unsigned char* rgbdata = nullptr;
            checkRuntime(cudaMallocHost(&pdata,   image->width * image->height * 3 / 2));
            checkRuntime(cudaMallocHost(&rgbdata, image->width * image->height * 3));
            checkRuntime(cudaMemcpyAsync(pdata, image->data0, image->width * image->height, cudaMemcpyDeviceToHost, stream));
            checkRuntime(cudaMemcpyAsync(pdata + image->width * image->height, image->data1, image->width * image->height / 2, cudaMemcpyDeviceToHost, stream));
            checkRuntime(cudaStreamSynchronize(stream));
            convert_nv12_to_rgb(pdata, rgbdata, image->width, image->height);
            stbi_write_jpg(file, image->width, image->height, 3, rgbdata, 100);
            checkRuntime(cudaFreeHost(pdata));
            checkRuntime(cudaFreeHost(rgbdata));
            return true;
        } else if (image->format == ImageFormat::BlockLinearNV12) {
            unsigned char* pdata   = nullptr;
            unsigned char* rgbdata = nullptr;
            checkRuntime(cudaMallocHost(&pdata,   image->width * image->height * 3 / 2));
            checkRuntime(cudaMallocHost(&rgbdata, image->width * image->height * 3));
            checkRuntime(cudaMemcpy2DFromArrayAsync(
                pdata, image->width,
                (cudaArray_t)image->reserve0, 0, 0, image->width, image->height, cudaMemcpyDeviceToHost, stream
            ));
            checkRuntime(cudaMemcpy2DFromArrayAsync(
                pdata + image->width * image->height, image->width,
                (cudaArray_t)image->reserve1, 0, 0, image->width, image->height / 2, cudaMemcpyDeviceToHost, stream
            ));
            checkRuntime(cudaStreamSynchronize(stream));
            convert_nv12_to_rgb(pdata, rgbdata, image->width, image->height);
            stbi_write_jpg(file, image->width, image->height, 3, rgbdata, 100);
            checkRuntime(cudaFreeHost(pdata));
            checkRuntime(cudaFreeHost(rgbdata));
            return true;
        }
        return false;
    }

    // Free image pointer
    void free_image(Image* image) {
        if (image == nullptr) return;

        if (image->format == ImageFormat::RGB) {
            if (image->data0) checkRuntime(cudaFree(image->data0));
        } else if (image->format == ImageFormat::RGBA) {
            if (image->data0) checkRuntime(cudaFree(image->data0));
        } else if (image->format == ImageFormat::PitchLinearNV12) {
            if (image->data0) checkRuntime(cudaFree(image->data0));
            if (image->data1) checkRuntime(cudaFree(image->data1));
        } else if (image->format == ImageFormat::BlockLinearNV12) {
            if (image->data0) checkRuntime(cudaDestroySurfaceObject((cudaSurfaceObject_t)image->data0));
            if (image->data1) checkRuntime(cudaDestroySurfaceObject((cudaSurfaceObject_t)image->data1));
            if (image->reserve0) checkRuntime(cudaFreeArray((cudaArray_t)image->reserve0));
            if (image->reserve1) checkRuntime(cudaFreeArray((cudaArray_t)image->reserve1));
        }
        delete image;
    }
};
