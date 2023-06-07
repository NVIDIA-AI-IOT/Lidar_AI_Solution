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
#include <cuda_runtime.h>
#include <string.h>

#include <algorithm>
#include <numeric>
#include <unordered_map>

#include "check.hpp"
#include "launch.cuh"
#include "tensor.hpp"

namespace nv {

using namespace std;

#define DISPATCH_BY_TYPES(dtype, ...)                  \
  [&]() {                                              \
    switch (dtype) {                                   \
      case DataType::Float32: {                        \
        using scalar_t = float;                        \
        return __VA_ARGS__();                          \
      }                                                \
      case DataType::Float16: {                        \
        using scalar_t = half;                         \
        return __VA_ARGS__();                          \
      }                                                \
      case DataType::UInt32: {                         \
        using scalar_t = uint32_t;                     \
        return __VA_ARGS__();                          \
      }                                                \
      case DataType::UInt64: {                         \
        using scalar_t = uint64_t;                     \
        return __VA_ARGS__();                          \
      }                                                \
      case DataType::UInt8: {                          \
        using scalar_t = uint8_t;                      \
        return __VA_ARGS__();                          \
      }                                                \
      case DataType::Int32: {                          \
        using scalar_t = int32_t;                      \
        return __VA_ARGS__();                          \
      }                                                \
      case DataType::Int64: {                          \
        using scalar_t = int64_t;                      \
        return __VA_ARGS__();                          \
      }                                                \
      case DataType::Int8: {                           \
        using scalar_t = int8_t;                       \
        return __VA_ARGS__();                          \
      }                                                \
      case DataType::Int16: {                          \
        using scalar_t = short;                        \
        return __VA_ARGS__();                          \
      }                                                \
      case DataType::UInt16: {                         \
        using scalar_t = unsigned short;               \
        return __VA_ARGS__();                          \
      }                                                \
      default: {                                       \
        using scalar_t = float;                        \
        Assertf(false, "Unknow dtype %d", (int)dtype); \
        return __VA_ARGS__();                          \
      }                                                \
    }                                                  \
  }();

static inline float _native_half2float(const unsigned short h) {
  unsigned int sign = ((static_cast<unsigned int>(h) >> 15U) & 1U);
  unsigned int exponent = ((static_cast<unsigned int>(h) >> 10U) & 0x1fU);
  unsigned int mantissa = ((static_cast<unsigned int>(h) & 0x3ffU) << 13U);
  float f(0.f);
  if (exponent == 0x1fU) { /* NaN or Inf */
    /* discard sign of a NaN */
    sign = ((mantissa != 0U) ? (sign >> 1U) : sign);
    mantissa = ((mantissa != 0U) ? 0x7fffffU : 0U);
    exponent = 0xffU;
  } else if (exponent == 0U) { /* Denorm or Zero */
    if (mantissa != 0U) {
      unsigned int msb;
      exponent = 0x71U;
      do {
        msb = (mantissa & 0x400000U);
        mantissa <<= 1U; /* normalize */
        --exponent;
      } while (msb == 0U);
      mantissa &= 0x7fffffU; /* 1.mantissa is implicit */
    }
  } else {
    exponent += 0x70U;
  }
  unsigned int u = ((sign << 31U) | (exponent << 23U) | mantissa);
  memcpy(&f, &u, sizeof(f));
  return f;
}

template <typename _T>
static __global__ void arange_kernel_device(size_t num, _T* pdata) {
  int index = cuda_linear_index;
  if (index < num) {
    pdata[index] = index;
  }
}

template <typename _T>
static void arange_kernel_host(size_t num, _T* pdata) {
  for (size_t index = 0; index < num; ++index) pdata[index] = index;
}

template <>
void arange_kernel_host<half>(size_t num, half* pdata) {
  for (size_t index = 0; index < num; ++index) pdata[index] = half((int)index);
}

template <typename _AData, typename _BData>
static __global__ void any_to_any_device(size_t num, _AData* input, _BData* output) {
  int index = cuda_linear_index;
  if (index < num) {
    output[index] = input[index];
  }
}

template <typename _BData>
static __global__ void any_to_any_device(size_t num, int64_t* input, _BData* output) {
  int index = cuda_linear_index;
  if (index < num) {
    output[index] = (int)input[index];
  }
}

template <typename _BData>
static __global__ void any_to_any_device(size_t num, uint64_t* input, _BData* output) {
  int index = cuda_linear_index;
  if (index < num) {
    output[index] = (int)input[index];
  }
}

template <>
std::string format_shape(const std::vector<int64_t>& shape) {
  char buf[200] = {0};
  char* p = buf;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i + 1 < shape.size())
      p += sprintf(p, "%ld x ", shape[i]);
    else
      p += sprintf(p, "%ld", shape[i]);
  }
  return buf;
}

template <>
std::string format_shape(const std::vector<int>& shape) {
  char buf[200] = {0};
  char* p = buf;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i + 1 < shape.size())
      p += sprintf(p, "%d x ", shape[i]);
    else
      p += sprintf(p, "%d", shape[i]);
  }
  return buf;
}

template <typename T>
std::vector<int64_t> to_int64(const std::vector<T>& array) {
  std::vector<int64_t> output(array.size());
  for (size_t i = 0; i < array.size(); ++i) output[i] = static_cast<int64_t>(array[i]);

  return output;
}

const char* dtype_string(DataType dtype) {
  switch (dtype) {
    case DataType::Float32:
      return "Float32";
    case DataType::Float16:
      return "Float16";
    case DataType::Int32:
      return "Int32";
    case DataType::Int64:
      return "Int64";
    case DataType::UInt64:
      return "UInt64";
    case DataType::UInt32:
      return "UInt32";
    case DataType::Int8:
      return "Int8";
    case DataType::UInt8:
      return "UInt8";
    case DataType::Int16:
      return "Int16";
    case DataType::UInt16:
      return "UInt16";
    default:
      return "Unknow";
  }
}

size_t dtype_bytes(DataType dtype) {
  switch (dtype) {
    case DataType::Float32:
      return sizeof(float);
    case DataType::Float16:
      return sizeof(unsigned short);
    case DataType::Int32:
      return sizeof(int);
    case DataType::Int64:
      return sizeof(int64_t);
    case DataType::UInt64:
      return sizeof(uint64_t);
    case DataType::UInt32:
      return sizeof(unsigned int);
    case DataType::Int8:
      return sizeof(char);
    case DataType::UInt8:
      return sizeof(unsigned char);
    case DataType::Int16:
      return sizeof(short);
    case DataType::UInt16:
      return sizeof(unsigned short);
    default:
      return 0ul;
  }
}

TensorData::~TensorData() { TensorData::free(); }

void TensorData::free() {
  if (data && owner) {
    if (device) {
      checkRuntime(cudaFree(data));
    } else {
      checkRuntime(cudaFreeHost(data));
    }
  }
  data = nullptr;
  owner = false;
  bytes = 0;
  dtype = DataType::None;
  device = false;
}

TensorData* TensorData::reference_new(void* data, size_t bytes, DataType dtype, bool device) {
  TensorData* output = new TensorData();
  output->owner = false;
  output->data = data;
  output->dtype = dtype;
  output->bytes = bytes;
  output->device = device;
  return output;
}

void TensorData::reference(void* data, size_t bytes, DataType dtype, bool device) {
  TensorData::free();
  this->owner = false;
  this->data = data;
  this->dtype = dtype;
  this->bytes = bytes;
  this->device = device;
}

TensorData* TensorData::create(size_t bytes, DataType dtype, bool device) {
  TensorData* output = new TensorData();
  output->owner = true;
  output->dtype = dtype;
  output->bytes = bytes;
  output->device = device;

  if (device)
    checkRuntime(cudaMalloc(&output->data, bytes));
  else
    checkRuntime(cudaMallocHost(&output->data, bytes));
  return output;
}

Tensor::Tensor(std::vector<int64_t> shape, DataType dtype, bool device) {
  size_t volumn = std::accumulate(shape.begin(), shape.begin() + shape.size(), 1, std::multiplies<int>());
  size_t bytes = volumn * dtype_bytes(dtype);
  this->shape = shape;
  this->numel = volumn;
  this->ndim = shape.size();
  this->data.reset(TensorData::create(bytes, dtype, device));
}

Tensor::Tensor(std::vector<int32_t> shape, DataType dtype, bool device) : Tensor(to_int64(shape), dtype, device) {}

void Tensor::create_(vector<int64_t> shape, DataType dtype, bool device) {
  this->release();
  size_t volumn = std::accumulate(shape.begin(), shape.begin() + shape.size(), 1, std::multiplies<int>());
  size_t bytes = volumn * dtype_bytes(dtype);
  this->shape = shape;
  this->numel = volumn;
  this->ndim = shape.size();
  this->data.reset(TensorData::create(bytes, dtype, device));
}

void Tensor::release() {
  this->shape.clear();
  this->data.reset();
  this->numel = 0;
  this->ndim = 0;
}

Tensor Tensor::create(vector<int64_t> shape, DataType dtype, bool device) { return Tensor(shape, dtype, device); }
Tensor Tensor::create(vector<int32_t> shape, DataType dtype, bool device) { return create(to_int64(shape), dtype, device); }

void Tensor::reference(void* data, vector<int64_t> shape, DataType dtype, bool device) {
  if (this->data == nullptr) {
    this->data.reset(new TensorData());
  }

  size_t volumn = std::accumulate(shape.begin(), shape.begin() + shape.size(), 1, std::multiplies<int>());
  size_t bytes = volumn * dtype_bytes(dtype);
  this->data->reference(data, bytes, dtype, device);
  this->numel = volumn;
  this->shape = shape;
  this->ndim = shape.size();
}

void Tensor::reference(void* data, vector<int32_t> shape, DataType dtype, bool device) {
  reference(data, to_int64(shape), dtype, device);
}

Tensor Tensor::from_data(void* data, vector<int64_t> shape, DataType dtype, bool device, void* stream) {
  Tensor output = Tensor::create(shape, dtype, device);
  if (device) {
    checkRuntime(cudaMemcpyAsync(output.ptr(), data, output.bytes(), cudaMemcpyDeviceToDevice, (cudaStream_t)stream));
  } else {
    checkRuntime(cudaMemcpyAsync(output.ptr(), data, output.bytes(), cudaMemcpyHostToHost, (cudaStream_t)stream));
  }
  return output;
}

Tensor Tensor::from_data(void* data, vector<int32_t> shape, DataType dtype, bool device, void* stream) {
  return from_data(data, to_int64(shape), dtype, device, stream);
}

Tensor Tensor::from_data_reference(void* data, vector<int64_t> shape, DataType dtype, bool device) {
  Tensor output;
  output.reference(data, shape, dtype, device);
  return output;
}

Tensor Tensor::from_data_reference(void* data, vector<int32_t> shape, DataType dtype, bool device) {
  return from_data_reference(data, to_int64(shape), dtype, device);
}

void Tensor::to_device_(void* stream_) {
  cudaStream_t stream = (cudaStream_t)stream_;
  if (!this->device() && !this->empty()) {
    shared_ptr<TensorData> newdata(TensorData::create(this->bytes(), this->dtype(), true));
    checkRuntime(cudaMemcpyAsync(newdata->data, this->ptr(), this->bytes(), cudaMemcpyHostToDevice, stream));
    checkRuntime(cudaStreamSynchronize(stream));
    this->data = newdata;
  }
}

Tensor Tensor::to_device(void* stream_) const {
  if (!this->device() && !this->empty()) {
    cudaStream_t stream = (cudaStream_t)stream_;
    Tensor output(shape, this->dtype(), true);
    checkRuntime(cudaMemcpyAsync(output.ptr(), this->ptr(), this->bytes(), cudaMemcpyHostToDevice, stream));
    checkRuntime(cudaStreamSynchronize(stream));
    return output;
  }
  return *this;
}

void Tensor::to_host_(void* stream_) {
  cudaStream_t stream = (cudaStream_t)stream_;
  if (this->device() && !this->empty()) {
    shared_ptr<TensorData> newdata(TensorData::create(this->bytes(), this->dtype(), false));
    checkRuntime(cudaMemcpyAsync(newdata->data, this->ptr(), this->bytes(), cudaMemcpyDeviceToHost, stream));
    checkRuntime(cudaStreamSynchronize(stream));
    this->data = newdata;
  }
}

Tensor Tensor::to_host(void* stream_) const {
  if (this->device() && !this->empty()) {
    cudaStream_t stream = (cudaStream_t)stream_;
    Tensor output(shape, this->dtype(), false);
    checkRuntime(cudaMemcpyAsync(output.ptr(), this->ptr(), this->bytes(), cudaMemcpyDeviceToHost, stream));
    checkRuntime(cudaStreamSynchronize(stream));
    return output;
  }
  return *this;
}

void Tensor::arange(void* _stream) {
  if (this->empty()) return;

  cudaStream_t stream = (cudaStream_t)_stream;
  if (this->device()) {
    DISPATCH_BY_TYPES(this->dtype(),
                      [&] { cuda_linear_launch(arange_kernel_device, stream, this->numel, this->ptr<scalar_t>()); });
  } else {
    DISPATCH_BY_TYPES(this->dtype(), [&] { arange_kernel_host(this->numel, this->ptr<scalar_t>()); });
  }
}

void Tensor::memset(unsigned char value, void* stream) {
  if (this->empty()) return;

  if (this->device()) {
    checkRuntime(cudaMemsetAsync(this->ptr(), value, this->bytes(), (cudaStream_t)stream));
  } else {
    ::memset(this->ptr(), value, this->bytes());
  }
}

Tensor Tensor::loadbinary(const std::string& file, std::vector<int64_t> shape, DataType dtype, bool device) {
  FILE* f = fopen(file.c_str(), "rb");
  if (f == nullptr) return Tensor();

  fseek(f, 0, SEEK_END);
  size_t fsize = ftell(f);
  fseek(f, 0, SEEK_SET);

  int num_implicit_dim = 0;
  size_t volumn_explicit_dim = 1;
  int i_implicit_dim = 0;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] == -1) {
      num_implicit_dim++;
      i_implicit_dim = i;
    } else
      volumn_explicit_dim *= shape[i];
  }

  if (num_implicit_dim > 0) {
    if (num_implicit_dim != 1) {
      printf("%d implicit dimensions are found, Only one implicit dimension can be supported.\n",
             num_implicit_dim);
      fclose(f);
      return Tensor();
    }

    size_t explicit_bytes = dtype_bytes(dtype) * volumn_explicit_dim;
    if (fsize % explicit_bytes != 0) {
      printf(
          "Cannot be calculated the implicit dimension, and the file bytes[%ld] cannot be divided "
          "by the size of the "
          "explicit dimension bytes[%ld]\n",
          fsize, explicit_bytes);
      fclose(f);
      return Tensor();
    }
    shape[i_implicit_dim] = fsize / explicit_bytes;
    volumn_explicit_dim *= shape[i_implicit_dim];
  }

  size_t bytes = dtype_bytes(dtype) * volumn_explicit_dim;
  if (bytes != fsize) {
    printf(
        "Cannot be loaded by the specified shape, The file has %ld byte and the shape requires %ld "
        "byte\n",
        fsize, bytes);
    fclose(f);
    return Tensor();
  }

  vector<unsigned char> host_data(bytes);
  if (fread(host_data.data(), 1, bytes, f) != bytes) {
    printf("Failed to read %ld bytes in file: %s\n", bytes, file.c_str());
    fclose(f);
    return Tensor();
  }
  fclose(f);

  Tensor output = Tensor::create(shape, dtype, device);
  if (device) {
    checkRuntime(cudaMemcpy(output.ptr(), host_data.data(), bytes, cudaMemcpyHostToDevice));
  } else {
    checkRuntime(cudaMemcpy(output.ptr(), host_data.data(), bytes, cudaMemcpyHostToHost));
  }
  checkRuntime(cudaDeviceSynchronize());
  return output;
}

Tensor Tensor::load(const std::string& file, bool device) {
  FILE* f = fopen(file.c_str(), "rb");
  if (f == nullptr) return Tensor();

  int head[3];
  if (fread(head, 1, sizeof(head), f) == 0) {
    printf("This is invalid tensor file %s\n", file.c_str());
    fclose(f);
    return Tensor();
  }

  if (head[0] != 0x33ff1101) {
    printf("This is invalid tensor file %s\n", file.c_str());
    fclose(f);
    return Tensor();
  }

  int ndim = head[1];
  int dtypei = head[2];
  int dims[16];

  if (fread(dims, 1, ndim * sizeof(int), f) == 0) {
    printf("This is invalid tensor file %s\n", file.c_str());
    fclose(f);
    return Tensor();
  }

  vector<int64_t> shape(ndim);
  std::transform(dims, dims + ndim, shape.begin(), [](int x) { return x; });

  int volumn = std::accumulate(dims, dims + ndim, 1, std::multiplies<int>());
  DataType dtype = (DataType)dtypei;
  size_t bytes = dtype_bytes(dtype) * volumn;
  vector<unsigned char> host_data(bytes);

  if (fread(host_data.data(), 1, bytes, f) == 0) {
    printf("This is invalid tensor file %s\n", file.c_str());
    fclose(f);
    return Tensor();
  }

  fclose(f);

  Tensor output = Tensor::create(shape, dtype, device);
  if (device) {
    checkRuntime(cudaMemcpy(output.ptr(), host_data.data(), bytes, cudaMemcpyHostToDevice));
  } else {
    checkRuntime(cudaMemcpy(output.ptr(), host_data.data(), bytes, cudaMemcpyHostToHost));
  }
  checkRuntime(cudaDeviceSynchronize());
  return output;
}

void Tensor::print(const char* prefix, size_t offset, size_t num_per_line, size_t lines) const {
  printf("%s[%s] %s%s", prefix, dtype_string(dtype()), format_shape(shape).c_str(), lines == 1 ? ": " : ": \n");

  if (this->empty()) {
    printf("empty.\n");
    return;
  }

  shared_ptr<TensorData> tensor_data = this->data;
  if (this->device()) {
    tensor_data = shared_ptr<TensorData>(TensorData::create(this->bytes(), this->dtype(), false));
    checkRuntime(cudaMemcpy(tensor_data->data, this->ptr(), this->bytes(), cudaMemcpyDeviceToHost));
  }

  size_t num_print = min(lines * num_per_line, numel);
  if (this->dtype() == DataType::Float32) {
    for (size_t i = 0; i < num_print; ++i) {
      printf("%.3f ", *((float*)tensor_data->data + offset + i));
      if ((i + 1) % num_per_line == 0) printf("\n");
    }
  } else if (this->dtype() == DataType::Float16) {
    for (size_t i = 0; i < num_print; ++i) {
      printf("%.3f ", _native_half2float(*((unsigned short*)tensor_data->data + offset + i)));
      if ((i + 1) % num_per_line == 0) printf("\n");
    }
  } else if (this->dtype() == DataType::Int32 || this->dtype() == DataType::UInt32) {
    for (size_t i = 0; i < num_print; ++i) {
      printf("%d ", *((int*)tensor_data->data + offset + i));
      if ((i + 1) % num_per_line == 0) printf("\n");
    }
  } else if (this->dtype() == DataType::Int64 || this->dtype() == DataType::UInt64) {
    for (size_t i = 0; i < num_print; ++i) {
      printf("%ld ", *((int64_t*)tensor_data->data + offset + i));
      if ((i + 1) % num_per_line == 0) printf("\n");
    }
  } else if (this->dtype() == DataType::Int16 || this->dtype() == DataType::UInt16) {
    for (size_t i = 0; i < num_print; ++i) {
      printf("%d ", *((short*)tensor_data->data + offset + i));
      if ((i + 1) % num_per_line == 0) printf("\n");
    }
  }
  if (num_print % num_per_line != 0) printf("\n");
}

void Tensor::copy_from_host(const void* data, void* stream) {
  if (this->empty()) return;

  cudaStream_t _stream = static_cast<cudaStream_t>(stream);
  if (this->device()) {
    checkRuntime(cudaMemcpyAsync(this->ptr(), data, this->bytes(), cudaMemcpyHostToDevice, _stream));
  } else {
    checkRuntime(cudaMemcpyAsync(this->ptr(), data, this->bytes(), cudaMemcpyHostToHost, _stream));
  }
}

void Tensor::copy_from_device(const void* data, void* stream) {
  if (this->empty()) return;

  cudaStream_t _stream = static_cast<cudaStream_t>(stream);
  if (this->device()) {
    checkRuntime(cudaMemcpyAsync(this->ptr(), data, this->bytes(), cudaMemcpyDeviceToDevice, _stream));
  } else {
    checkRuntime(cudaMemcpyAsync(this->ptr(), data, this->bytes(), cudaMemcpyDeviceToHost, _stream));
  }
}

Tensor Tensor::clone(void* _stream) const {
  Tensor output = *this;
  cudaStream_t stream = (cudaStream_t)_stream;
  if (this->device() && !this->empty()) {
    shared_ptr<TensorData> newdata(TensorData::create(this->bytes(), this->dtype(), true));
    checkRuntime(cudaMemcpyAsync(newdata->data, this->ptr(), this->bytes(), cudaMemcpyDeviceToDevice, stream));
    checkRuntime(cudaStreamSynchronize(stream));
    output.data = newdata;
  } else if (!this->device() && !this->empty()) {
    shared_ptr<TensorData> newdata(TensorData::create(this->bytes(), this->dtype(), false));
    checkRuntime(cudaMemcpyAsync(newdata->data, this->ptr(), this->bytes(), cudaMemcpyHostToHost, stream));
    checkRuntime(cudaStreamSynchronize(stream));
    output.data = newdata;
  }
  return output;
}

Tensor Tensor::to_half(void* _stream) const {
  Tensor output;
  cudaStream_t stream = (cudaStream_t)_stream;

  if (this->empty()) return output;
  if (this->dtype() == DataType::Float16) return this->clone(stream);
  if (!this->device()) {
    printf("Unsupport non device convertion.\n");
    return output;
  }
  output = Tensor::create(this->shape, DataType::Float16);

  DISPATCH_BY_TYPES(this->dtype(), [&] {
    cuda_linear_launch(any_to_any_device, stream, this->numel, this->ptr<scalar_t>(), output.ptr<half>());
    checkRuntime(cudaStreamSynchronize(stream));
  });
  return output;
}

Tensor Tensor::to_float(void* _stream) const {
  Tensor output;
  cudaStream_t stream = (cudaStream_t)_stream;

  if (this->empty()) return output;
  if (this->dtype() == DataType::Float32) return this->clone(stream);
  if (!this->device()) {
    printf("Unsupport non device convertion.\n");
    return output;
  }
  output = Tensor::create(this->shape, DataType::Float32);

  DISPATCH_BY_TYPES(this->dtype(), [&] {
    cuda_linear_launch(any_to_any_device, stream, this->numel, this->ptr<scalar_t>(), output.ptr<float>());
    checkRuntime(cudaStreamSynchronize(stream));
  });
  return output;
}

bool Tensor::save(const std::string& file, void* stream_) const {
  cudaStream_t stream = (cudaStream_t)stream_;
  FILE* f = fopen(file.c_str(), "wb");
  if (f == nullptr) {
    printf("Failed to open %s\n", file.c_str());
    return false;
  }

  int head[] = {0x33ff1101, (int)this->shape.size(), (int)this->dtype()};
  int dims[16];
  std::transform(this->shape.begin(), this->shape.end(), dims, [](int64_t i) -> int { return i; });

  fwrite(head, 1, sizeof(head), f);
  fwrite(dims, 1, this->shape.size() * sizeof(int), f);

  if (this->device()) {
    std::vector<char> host_data(this->bytes());
    checkRuntime(cudaMemcpyAsync(host_data.data(), this->ptr(), this->bytes(), cudaMemcpyDeviceToHost, stream));
    checkRuntime(cudaStreamSynchronize(stream));
    fwrite(host_data.data(), 1, this->bytes(), f);
  } else {
    fwrite(this->ptr(), 1, this->bytes(), f);
  }
  fclose(f);
  return true;
}

void Tensor::self_byte_check(size_t type_bytes) const {
  size_t self_bytes = dtype_bytes(this->dtype());
  if (self_bytes != type_bytes) {
    this->print("This");
    Assertf(self_bytes == type_bytes,
            "Failed to check the data type, your code may have a logic error. The type of this tensor is %d bytes, but the "
            "pointer is %d bytes.",
            static_cast<int>(self_bytes), static_cast<int>(type_bytes));
  }
}

bool Tensor::save(const Tensor& tensor, const std::string& file, void* stream) { return tensor.save(file, stream); }

};  // namespace nv