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
 
#ifndef __SPCONV_TENSOR_HPP__
#define __SPCONV_TENSOR_HPP__

#include <assert.h>
#include <stdarg.h>
#include <stdlib.h>

#include <memory>
#include <string>
#include <vector>

namespace spconv {

#define Exported __attribute__((visibility("default")))

enum class DataType : int {
  None = 0,
  Int32 = 1,
  Float16 = 2,
  Float32 = 3,
  Int64 = 4,
  UInt64 = 5,
  UInt32 = 6,
  Int8 = 7,
  UInt8 = 8
};

struct TensorData {
  void *data = nullptr;
  DataType dtype = DataType::None;
  size_t bytes = 0;
  size_t capacity = 0;
  bool device = true;
  bool owner = false;

  bool empty() const { return data == nullptr; }
  void reference(void *data, size_t bytes, DataType dtype, bool device);
  void rearrange_and_realloc(size_t bytes, DataType dtype, bool device);
  void free();
  virtual ~TensorData();

  static TensorData *reference_new(void *data, size_t bytes, DataType dtype, bool device);
  static TensorData *create(size_t bytes, DataType dtype, bool device);
};

struct Tensor {
  std::vector<int64_t> shape;
  std::shared_ptr<TensorData> data;
  size_t numel = 0;
  size_t ndim = 0;

  template <typename T>
  T *ptr() const {
    return data ? (T *)data->data : nullptr;
  }
  void *ptr() const { return data ? data->data : nullptr; }

  template <typename T>
  T *begin() const {
    return data ? (T *)data->data : nullptr;
  }
  template <typename T>
  T *end() const {
    return data ? (T *)data->data + numel : nullptr;
  }

  Exported int64_t size(int index) const { return shape[index]; }
  Exported size_t bytes() const { return data ? data->bytes : 0; }
  Exported bool empty() const { return data == nullptr || data->empty(); }
  Exported DataType dtype() const { return data ? data->dtype : DataType::None; }
  Exported bool device() const { return data ? data->device : false; }
  Exported void reference(void *data, std::vector<int64_t> shape, DataType dtype,
                          bool device = true);
  Exported void to_device_(void *stream = nullptr);
  Exported void to_host_(void *stream = nullptr);
  Exported Tensor to_device(void *stream = nullptr);
  Exported Tensor to_host(void *stream = nullptr);
  Exported Tensor clone(void *stream = nullptr);
  Exported Tensor to_half(void *stream = nullptr);
  Exported float absmax(void *stream = nullptr);
  Exported void print(const std::string &prefix = "Tensor", size_t offset = 0,
                      size_t num_per_line = 10, size_t lines = 1) const;
  Exported void memset(unsigned char value = 0, void *stream = nullptr);
  Exported void arange(size_t num, void *stream = nullptr);
  Exported bool save(const std::string &file, void *stream = nullptr) const;
  Exported void create_(const std::vector<int64_t> &shape, DataType dtype, bool device = true);

  Exported Tensor() = default;
  Exported Tensor(std::vector<int64_t> shape, DataType dtype, bool device = true);

  Exported static Tensor create(std::vector<int64_t> shape, DataType dtype, bool device = true);
  Exported static Tensor from_data(void *data, std::vector<int64_t> shape, DataType dtype,
                                   bool device = true, void *stream = nullptr);
  Exported static Tensor from_data_reference(void *data, std::vector<int64_t> shape, DataType dtype,
                                             bool device = true);
  Exported static Tensor load(const std::string &file, bool device = true);
  Exported static Tensor load_from_raw(const std::string &file, std::vector<int64_t> shape,
                                       DataType dtype, bool device = true);
  Exported static bool save(const Tensor &tensor, const std::string &file, void *stream = nullptr);
};

Exported std::vector<int64_t> toshape(const std::vector<int> &ishape);
Exported float native_half2float(const unsigned short h);
Exported const char *dtype_string(DataType dtype);
Exported size_t dtype_bytes(DataType dtype);
template <typename _T>
Exported std::string format_shape(const std::vector<_T> &shape, bool space = true);

};  // namespace spconv

#endif  // #ifndef __SPCONV_TENSOR_HPP__