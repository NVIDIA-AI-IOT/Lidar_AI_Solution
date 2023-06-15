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
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "yuv_to_rgb.hpp"

using namespace std;

#define checkRuntime(call)  check_runtime(call, #call, __LINE__, __FILE__)

template<typename _T>struct AsDataType{};
template<>struct AsDataType<uint8_t>{static const DataType type = DataType::Uint8;};
template<>struct AsDataType<int8_t> {static const DataType type = DataType::Int8;};
template<>struct AsDataType<float>  {static const DataType type = DataType::Float32;};
template<>struct AsDataType<__half> {static const DataType type = DataType::Float16;};

template<DataType _T>struct AsPODType{};
template<>struct AsPODType<DataType::Uint8>   {typedef uint8_t type;};
template<>struct AsPODType<DataType::Float32> {typedef float   type;};
template<>struct AsPODType<DataType::Float16> {typedef __half  type;};
template<>struct AsPODType<DataType::Int8>    {typedef int8_t  type;};

static bool __inline__ check_runtime(cudaError_t e, const char* call, int line, const char *file){
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA Runtime error %s # %s, code = %s [ %d ] in file %s:%d\n", call, cudaGetErrorString(e), cudaGetErrorName(e), e, file, line);
        return false;
    }
    return true;
}

size_t dtype_sizeof(DataType dtype){
    switch(dtype){
    case DataType::Float32: return sizeof(AsPODType<DataType::Float32>::type);
    case DataType::Float16: return sizeof(AsPODType<DataType::Float16>::type);
    case DataType::Uint8:   return sizeof(AsPODType<DataType::Uint8>::type);
    case DataType::Int8:    return sizeof(AsPODType<DataType::Int8>::type);
    default: return 0;
    }
}

const char* pixel_layout_name(PixelLayout layout){

    switch(layout){
    case PixelLayout::NCHW16_BGR: return "NCHW16_BGR";
    case PixelLayout::NCHW16_RGB: return "NCHW16_RGB";
    case PixelLayout::NCHW32_BGR: return "NCHW32_BGR";
    case PixelLayout::NCHW32_RGB: return "NCHW32_RGB";
    case PixelLayout::NCHW4_BGR: return "NCHW4_BGR";
    case PixelLayout::NCHW4_RGB: return "NCHW4_RGB";
    case PixelLayout::NCHW_BGR: return "NCHW_BGR";
    case PixelLayout::NCHW_RGB: return "NCHW_RGB";
    case PixelLayout::NHWC_BGR: return "NHWC_BGR";
    case PixelLayout::NHWC_RGB: return "NHWC_RGB";
    default: return "UnknowLayout";
    }
}

const char* yuvformat_name(YUVFormat format){

    switch(format){
    case YUVFormat::NV12PitchLinear: return "NV12PitchLinear";
    case YUVFormat::NV12BlockLinear: return "NV12BlockLinear";
    case YUVFormat::YUV422Packed_YUYV: return "YUV422Packed_YUYV";
    default: return "UnknowYUVFormat";
    }
}

const char* interp_name(Interpolation interp){

    switch(interp){
    case Interpolation::Nearest:  return "Nearest";
    case Interpolation::Bilinear: return "Bilinear";
    default: return "UnknowInterpolation";
    }
}

const char* dtype_name(DataType dtype){

    switch(dtype){
    case DataType::Float32: return "Float32";
    case DataType::Float16: return "Float16";
    case DataType::Uint8:   return "Uint8";
    case DataType::Int8:   return "Int8";
    default: return "UnknowDataType";
    }
}

void free_yuv_host_image(YUVHostImage* p){
    if(p){
        if(p->data) delete p->data;
        delete p;
    }
}

YUVHostImage* read_yuv(const string& file, int width, int height, YUVFormat format){

    if((width % 2 != 0) || (height % 2 != 0)){
        std::fprintf(stderr, "Unsupport resolution. %d x %d\n", width, height);
        return nullptr;
    }

    fstream infile(file, ios::binary | ios::in);
    if(!infile){
        std::fprintf(stderr, "Failed to open: %s\n", file.c_str());
        return nullptr;
    }

    infile.seekg(0, ios::end);

    // check yuv size
    size_t file_size   = infile.tellg();
    size_t y_area      = width * height;
    size_t except_size = y_area * 3 / 2;
    size_t stride      = width;
    if(format == YUVFormat::YUV422Packed_YUYV){
        except_size = y_area * 2;
        stride      = width * 2;
    }

    if(file_size != except_size){
        std::fprintf(stderr, "Wrong size of yuv image : %lld bytes, expected %lld bytes\n", file_size, except_size);
        return nullptr;
    }

    YUVHostImage* output = new YUVHostImage();
    output->width  = width;
    output->height = height;
    output->stride = stride;
    output->y_area = y_area;
    output->data   = new uint8_t[except_size];
    output->format = format;

    infile.seekg(0, ios::beg);
    if(!infile.read((char*)output->data, except_size).good()){
        free_yuv_host_image(output);
        std::fprintf(stderr, "Failed to read %lld byte data\n", y_area);
        return nullptr;
    }
    return output;
}

void free_yuv_gpu_image(YUVGPUImage* p){
    if(p){
        if(p->format == YUVFormat::NV12PitchLinear){
            if(p->chroma) checkRuntime(cudaFree(p->chroma));
            if(p->luma)   checkRuntime(cudaFree(p->luma));
        }else if(p->format == YUVFormat::NV12BlockLinear){
            if(p->chroma) checkRuntime(cudaDestroyTextureObject((cudaTextureObject_t)p->chroma));
            if(p->luma)   checkRuntime(cudaDestroyTextureObject((cudaTextureObject_t)p->luma));
            if(p->chroma_array) checkRuntime(cudaFreeArray((cudaArray_t)p->chroma_array));
            if(p->luma_array)   checkRuntime(cudaFreeArray((cudaArray_t)p->luma_array));
        }else if(p->format == YUVFormat::YUV422Packed_YUYV){
            if(p->luma) checkRuntime(cudaFree(p->luma));
        }
        delete p;
    }
}

YUVGPUImage* create_yuv_gpu_image(int width, int height, int batch_size, YUVFormat format){

    YUVGPUImage* output = new YUVGPUImage();
    output->width  = width;
    output->height = height;
    output->batch  = batch_size;
    output->format = format;
    output->stride = width;

    if(format == YUVFormat::NV12PitchLinear){
        checkRuntime(cudaMalloc(&output->luma,   width * height * batch_size));
        checkRuntime(cudaMalloc(&output->chroma, width * height / 2 * batch_size));
    }else if(format == YUVFormat::NV12BlockLinear){
        cudaChannelFormatDesc YplaneDesc = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
        checkRuntime(cudaMallocArray((cudaArray_t*)&output->luma_array,   &YplaneDesc, width, height * batch_size, 0));

        // One pixel of the uv channel contains 2 bytes
        cudaChannelFormatDesc UVplaneDesc = cudaCreateChannelDesc(8, 8, 0, 0, cudaChannelFormatKindUnsigned);
        checkRuntime(cudaMallocArray((cudaArray_t*)&output->chroma_array, &UVplaneDesc, width / 2, height / 2 * batch_size, 0));

        cudaResourceDesc luma_desc = {};
        luma_desc.resType         = cudaResourceTypeArray;
        luma_desc.res.array.array = (cudaArray_t)output->luma_array;

        cudaTextureDesc texture_desc = {};
        texture_desc.filterMode = cudaFilterModePoint;
        texture_desc.readMode   = cudaReadModeElementType;
        checkRuntime(cudaCreateTextureObject((cudaTextureObject_t*)&output->luma, &luma_desc, &texture_desc, NULL));

        cudaResourceDesc chroma_desc = {};
        chroma_desc.resType         = cudaResourceTypeArray;
        chroma_desc.res.array.array = (cudaArray_t)output->chroma_array;
        checkRuntime(cudaCreateTextureObject((cudaTextureObject_t*)&output->chroma, &chroma_desc, &texture_desc, NULL));
    }else if(format == YUVFormat::YUV422Packed_YUYV){
        output->stride = width * 2;
        checkRuntime(cudaMalloc(&output->luma, output->stride * height * batch_size));
    }
    return output;
}

void copy_yuv_host_to_gpu(const YUVHostImage* host, YUVGPUImage* gpu, unsigned int ibatch, unsigned int crop_width, unsigned int crop_height, cudaStream_t stream){

    if(crop_width > host->width || crop_height > host->height){
        std::fprintf(stderr, "Failed to copy, invalid crop size %d x %d is larger than %d x %d\n", crop_width, crop_height, host->width, host->height);
        return;
    }

    if(crop_width > gpu->width || crop_height > gpu->height){
        std::fprintf(stderr, "Failed to copy, invalid crop size %d x %d is larger than %d x %d\n", crop_width, crop_height, gpu->width, gpu->height);
        return;
    }

    if(ibatch >= gpu->batch){
        std::fprintf(stderr, "Invalid ibatch %d is larger than %d, index out of range.\n", ibatch, gpu->batch);
        return;
    }

    if(host->format == YUVFormat::YUV422Packed_YUYV){
        if(gpu->format != YUVFormat::YUV422Packed_YUYV){
            std::fprintf(stderr, "Copied images should have the same format. host is %s, gpu is %s\n", yuvformat_name(host->format), yuvformat_name(gpu->format));
            return;
        }
    }

    if(gpu->format == YUVFormat::NV12PitchLinear){
        checkRuntime(cudaMemcpy2DAsync(gpu->luma + ibatch * gpu->stride * gpu->height,   gpu->stride, host->data,              host->stride,
            crop_width, crop_height,     cudaMemcpyHostToDevice, stream));
        checkRuntime(cudaMemcpy2DAsync(gpu->chroma + ibatch * gpu->stride * gpu->height / 2, gpu->stride, host->data + host->y_area, host->stride,
            crop_width, crop_height / 2, cudaMemcpyHostToDevice, stream));
    }else if(gpu->format == YUVFormat::NV12BlockLinear){
        checkRuntime(cudaMemcpy2DToArrayAsync((cudaArray_t)gpu->luma_array,   0, ibatch * gpu->height,     host->data,              host->stride,
            crop_width, crop_height,     cudaMemcpyHostToDevice, stream));
        checkRuntime(cudaMemcpy2DToArrayAsync((cudaArray_t)gpu->chroma_array, 0, ibatch * gpu->height / 2, host->data + host->y_area, host->stride,
            crop_width, crop_height / 2, cudaMemcpyHostToDevice, stream));
    }else if(gpu->format == YUVFormat::YUV422Packed_YUYV){
        checkRuntime(cudaMemcpy2DAsync(gpu->luma + ibatch * gpu->stride * gpu->height,   gpu->stride, host->data,              host->stride,
            crop_width * 2, crop_height,     cudaMemcpyHostToDevice, stream));
    }
}

void free_rgb_gpu_image(RGBGPUImage* p){
    if(p){
        if(p->data) checkRuntime(cudaFree(p->data));
        delete p;
    }
}

RGBGPUImage* create_rgb_gpu_image(int width, int height, int batch, PixelLayout layout, DataType dtype){

    RGBGPUImage* output = new RGBGPUImage();
    int channel = 0;
    int stride  = 0;
    
    switch(layout){
    case PixelLayout::NHWC_RGB:
    case PixelLayout::NHWC_BGR:
        channel = 3; stride = width * channel;
        break;
    case PixelLayout::NCHW_RGB:
    case PixelLayout::NCHW_BGR:
        channel = 3; stride = width;
        break;
    case PixelLayout::NCHW4_RGB:
    case PixelLayout::NCHW4_BGR:
        channel = 4; stride = width * channel;
        break;
    case PixelLayout::NCHW16_RGB:
    case PixelLayout::NCHW16_BGR:
        channel = 16; stride = width * channel;
        break;
    case PixelLayout::NCHW32_RGB:
    case PixelLayout::NCHW32_BGR:
        channel = 32; stride = width * channel;
        break;
    }

    auto bytes = width * height * channel * batch * dtype_sizeof(dtype);
    checkRuntime(cudaMalloc(&output->data, bytes));
    output->width   = width;
    output->height  = height;
    output->batch   = batch;
    output->channel = channel;
    output->stride  = stride;
    output->layout  = layout;
    output->dtype   = dtype;
    return output;
}

bool save_rgbgpu_to_file(const string& file, RGBGPUImage* gpu, cudaStream_t stream){

    unsigned int header[] = {0xAABBCCEF, gpu->width, gpu->height, gpu->channel, gpu->batch, (unsigned int)gpu->layout, (unsigned int)gpu->dtype};
    std::fstream fout(file, ios::binary | ios::out);
    if(!fout.good()){
        std::fprintf(stderr, "Can not open %s\n", file.c_str());
        return false;
    }
    fout.write((char*)header, sizeof(header));

    size_t num_element    = gpu->width * gpu->height * gpu->channel * gpu->batch;
    size_t sizeof_element = dtype_sizeof(gpu->dtype);
    uint8_t* phost   = new uint8_t[num_element * sizeof_element];
    checkRuntime(cudaMemcpyAsync(phost, gpu->data, num_element * sizeof_element, ::cudaMemcpyDeviceToHost, stream));
    checkRuntime(cudaStreamSynchronize(stream));
    fout.write((char*)phost, num_element * sizeof_element);

    delete [] phost;
    return fout.good();
}

void batched_convert_yuv_to_rgb(
    YUVGPUImage* input, RGBGPUImage* output, 
    int scaled_width, int scaled_height,
    int output_xoffset, int output_yoffset, FillColor fillcolor,
    float mean0,  float mean1,  float mean2, 
    float scale0, float scale1, float scale2,
    Interpolation interp,
    cudaStream_t stream
){
    batched_convert_yuv_to_rgb(
        input->luma, input->chroma, input->width, input->stride, input->height, input->batch, input->format,
        scaled_width, scaled_height, output_xoffset, output_yoffset, fillcolor,
        output->data, output->width, output->stride, output->height, output->dtype, output->layout, interp,
        mean0, mean1, mean2, scale0, scale1, scale2, stream
    );
}