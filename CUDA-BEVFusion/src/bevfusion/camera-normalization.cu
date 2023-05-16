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

#include <cuda_fp16.h>

#include "camera-normalization.hpp"
#include "common/check.hpp"
#include "common/launch.cuh"

namespace bevfusion {
namespace camera {

#define INTER_RESIZE_COEF_BITS 11
#define INTER_RESIZE_COEF_SCALE (1 << INTER_RESIZE_COEF_BITS)
#define CAST_BITS (INTER_RESIZE_COEF_BITS << 1)

template <typename _T>
static __forceinline__ __device__ _T limit(_T value, _T low, _T high) {
  return value < low ? low : (value > high ? high : value);
}

struct half3 {
  half x, y, z;

  __device__ __host__ half3(half x, half y, half z) : x(x), y(y), z(z) {}
};

NormMethod NormMethod::mean_std(const float mean[3], const float std[3], float alpha, float beta, ChannelType channel_type) {
  NormMethod out;
  out.type = NormType::MeanStd;
  out.alpha = alpha;
  out.beta = beta;
  out.channel_type = channel_type;
  memcpy(out.mean, mean, sizeof(out.mean));
  memcpy(out.std, std, sizeof(out.std));
  return out;
}

NormMethod NormMethod::alpha_beta(float alpha, float beta, ChannelType channel_type) {
  NormMethod out;
  out.type = NormType::AlphaBeta;
  out.alpha = alpha;
  out.beta = beta;
  out.channel_type = channel_type;
  return out;
}

NormMethod NormMethod::None() { return NormMethod(); }

template <Interpolation interpolation>
static __device__ uchar3 load_pixel(const uchar3* image, int x, int y, int tix, int tiy, float sx, float sy, int width,
                                    int height);

template <>
__device__ uchar3 load_pixel<Interpolation::Nearest>(const uchar3* image, int x, int y, int tox, int toy, float sx, float sy,
                                                     int width, int height) {
  // In some cases, the floating point precision will lead to miscalculation of the value,
  // making the result not exactly match with opencv,
  // so here you need to add eps as precision compensation
  //
  // A special case is when the input is 3840 and the output is 446, x = 223:
  // const int src_x_double = 223.0  * (3840.0  / 446.0);            // -> 1920
  // const int src_x_float  = 223.0f * (3840.0f / 446.0f);           // -> 1919
  // const int src_x_float  = 223.0f * (3840.0f / 446.0f) + 1e-5;    // -> 1920
  //
  // !!! If you want to use the double for sx/sy, you'll get a 2x speed drop
  const float eps = 1e-5;
  int ix = (x + tox) * sx + eps;
  int iy = (y + toy) * sy + eps;
  return image[iy * width + ix];
}

template <>
__device__ uchar3 load_pixel<Interpolation::Bilinear>(const uchar3* image, int x, int y, int tox, int toy, float sx, float sy,
                                                      int width, int height) {
  uchar3 rgb[4];
  float src_x = (x + tox + 0.5f) * sx - 0.5f;
  float src_y = (y + toy + 0.5f) * sy - 0.5f;
  int y_low = floorf(src_y);
  int x_low = floorf(src_x);
  int y_high = limit(y_low + 1, 0, height - 1);
  int x_high = limit(x_low + 1, 0, width - 1);
  y_low = limit(y_low, 0, height - 1);
  x_low = limit(x_low, 0, width - 1);

  int ly = rint((src_y - y_low) * INTER_RESIZE_COEF_SCALE);
  int lx = rint((src_x - x_low) * INTER_RESIZE_COEF_SCALE);
  int hy = INTER_RESIZE_COEF_SCALE - ly;
  int hx = INTER_RESIZE_COEF_SCALE - lx;

  rgb[0] = image[y_low * width + x_low];
  rgb[1] = image[y_low * width + x_high];
  rgb[2] = image[y_high * width + x_low];
  rgb[3] = image[y_high * width + x_high];

  uchar3 output;
  output.x =
      (((hy * ((hx * rgb[0].x + lx * rgb[1].x) >> 4)) >> 16) + ((ly * ((hx * rgb[2].x + lx * rgb[3].x) >> 4)) >> 16) + 2) >> 2;
  output.y =
      (((hy * ((hx * rgb[0].y + lx * rgb[1].y) >> 4)) >> 16) + ((ly * ((hx * rgb[2].y + lx * rgb[3].y) >> 4)) >> 16) + 2) >> 2;
  output.z =
      (((hy * ((hx * rgb[0].z + lx * rgb[1].z) >> 4)) >> 16) + ((ly * ((hx * rgb[2].z + lx * rgb[3].z) >> 4)) >> 16) + 2) >> 2;
  return output;
}

template <NormType norm_type>
static __device__ half3 normalize_value(const uchar3& pixel, const NormMethod& method);

template <>
__device__ half3 normalize_value<NormType::Nothing>(const uchar3& pixel, const NormMethod& method) {
  return half3(pixel.x, pixel.y, pixel.z);
}

template <>
__device__ half3 normalize_value<NormType::AlphaBeta>(const uchar3& pixel, const NormMethod& method) {
  return half3(pixel.x * method.alpha + method.beta, pixel.y * method.alpha + method.beta, pixel.z * method.alpha + method.beta);
}

template <>
__device__ half3 normalize_value<NormType::MeanStd>(const uchar3& pixel, const NormMethod& method) {
  return half3((pixel.x * method.alpha - method.mean[0]) / method.std[0] + method.beta,
               (pixel.y * method.alpha - method.mean[1]) / method.std[1] + method.beta,
               (pixel.z * method.alpha - method.mean[2]) / method.std[2] + method.beta);
}

template <typename OutputType>
static __device__ void store_output(const half3& normed, void* output, int icamera, int ix, int iy, int nx, int ny);

template <>
__device__ void store_output<half>(const half3& normed, void* output, int icamera, int ix, int iy, int nx, int ny) {
  half* planar_pointer = (half*)output + icamera * ny * nx * 3;
  planar_pointer[(0 * ny + iy) * nx + ix] = normed.x;
  planar_pointer[(1 * ny + iy) * nx + ix] = normed.y;
  planar_pointer[(2 * ny + iy) * nx + ix] = normed.z;
}

template <NormType norm_type, Interpolation interpolation, typename OutputType>
static __global__ void normalize_to_planar_kernel(int nx, int ny, int nz, float sx, float sy, int crop_x, int crop_y,
                                                  uchar3* imgs, int image_width, int image_height, void* output,
                                                  NormMethod method) {
  int ix = cuda_2d_x;
  int iy = cuda_2d_y;
  if (ix >= nx || iy >= ny) return;

  int icamera = blockIdx.z;
  uchar3* img = imgs + image_width * image_height * icamera;
  uchar3 pixel = load_pixel<interpolation>(img, ix, iy, crop_x, crop_y, sx, sy, image_width, image_height);

  if (method.channel_type == ChannelType::Invert) {
    unsigned char t = pixel.z;
    pixel.z = pixel.x;
    pixel.x = t;
  }

  half3 normed = normalize_value<norm_type>(pixel, method);
  store_output<OutputType>(normed, output, icamera, ix, iy, nx, ny);
}

typedef void (*normalize_to_planar_kernel_fn)(int nx, int ny, int nz, float sx, float sy, int crop_x_, int crop_y_, uchar3* imgs,
                                              int image_width, int image_height, void* output, NormMethod method);

#define DefineNormType(...)                                                                                               \
  normalize_to_planar_kernel<NormType::Nothing, __VA_ARGS__>, normalize_to_planar_kernel<NormType::MeanStd, __VA_ARGS__>, \
      normalize_to_planar_kernel<NormType::AlphaBeta, __VA_ARGS__>,

#define DefineInterpolation(...) \
  DefineNormType(Interpolation::Nearest, __VA_ARGS__) DefineNormType(Interpolation::Bilinear, __VA_ARGS__)

#define DefineDataType DefineInterpolation(half)

#define DefineAllFunction DefineDataType

static const normalize_to_planar_kernel_fn func_list[] = {DefineAllFunction nullptr};

class NormalizationImplement : public Normalization {
 public:
  virtual ~NormalizationImplement() {
    if (raw_images_) checkRuntime(cudaFree(raw_images_));
    if (normalize_images_) checkRuntime(cudaFree(normalize_images_));
  }

  bool init(const NormalizationParameter& param) {
    this->param_ = param;

    int resized_width = static_cast<int>(param.image_width * param.resize_lim);
    int resized_height = static_cast<int>(param.image_height * param.resize_lim);
    this->crop_x_ = (resized_width - param.output_width) / 2;
    this->crop_y_ = resized_height - param.output_height;
    this->sx_ = 1.0f / param.resize_lim;
    this->sy_ = 1.0f / param.resize_lim;

    checkRuntime(cudaMalloc(&raw_images_, param.image_width * param.image_height * 3 * param.num_camera * sizeof(unsigned char)));
    checkRuntime(cudaMalloc(&normalize_images_, param.output_width * param.output_height * 3 * param.num_camera * sizeof(half)));
    return true;
  }

  virtual nvtype::half* forward(const unsigned char** images, void* stream) override {
    size_t index = (size_t)param_.interpolation * 3 + (size_t)param_.method.type;
    Assertf(index < sizeof(func_list) / sizeof(func_list[0]) - 1, "Invalid configure index: %d", static_cast<int>(index));

    auto normalize_to_planar_kernel_function = func_list[index];
    cudaStream_t _stream = static_cast<cudaStream_t>(stream);
    size_t bytes_image = param_.image_width * param_.image_height * 3 * sizeof(unsigned char);

    for (int icamera = 0; icamera < param_.num_camera; ++icamera)
      checkRuntime(
          cudaMemcpyAsync(raw_images_ + icamera * bytes_image, images[icamera], bytes_image, cudaMemcpyHostToDevice, _stream));

    cuda_2d_launch(normalize_to_planar_kernel_function, _stream, param_.output_width, param_.output_height, param_.num_camera,
                   sx_, sy_, crop_x_, crop_y_, reinterpret_cast<uchar3*>(raw_images_), param_.image_width, param_.image_height,
                   normalize_images_, param_.method);

    return reinterpret_cast<nvtype::half*>(normalize_images_);
  }

 private:
  NormalizationParameter param_;
  float sx_ = 1.0f;
  float sy_ = 1.0f;
  int crop_x_ = 0;
  int crop_y_ = 0;
  half* normalize_images_ = nullptr;
  unsigned char* raw_images_ = nullptr;
};

std::shared_ptr<Normalization> create_normalization(const NormalizationParameter& param) {
  std::shared_ptr<NormalizationImplement> instance(new NormalizationImplement());
  if (!instance->init(param)) {
    instance.reset();
  }
  return instance;
}

};  // namespace camera
};  // namespace bevfusion