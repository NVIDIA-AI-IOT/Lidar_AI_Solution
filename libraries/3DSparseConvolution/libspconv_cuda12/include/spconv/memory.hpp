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
 
#ifndef __SPCONV_MEMORY_HPP__
#define __SPCONV_MEMORY_HPP__

#include <memory>
#include <string>
#include <unordered_map>

#include "check.hpp"

namespace spconv {

class PinnedMemoryData {
 public:
  inline void *ptr() const { return ptr_; }
  inline size_t bytes() const { return bytes_; }
  inline bool empty() const { return ptr_ == nullptr; }
  virtual ~PinnedMemoryData() { free_memory(nullptr); }
  PinnedMemoryData() = default;
  PinnedMemoryData(const std::string &name) { this->name_ = name; }

  void alloc_or_resize_to(size_t nbytes, cudaStream_t stream) {
    if (capacity_ < nbytes) {
      // dprintf("%s Free old %d, malloc new %d bytes.\n", name_.c_str(), capacity_, nbytes);
      free_memory(stream);
      checkRuntime(cudaMallocHost(&ptr_, nbytes));
      capacity_ = nbytes;
    }
    bytes_ = nbytes;
  }

  void alloc(size_t nbytes, cudaStream_t stream) { alloc_or_resize_to(nbytes, stream); }

  void resize(size_t nbytes) {
    if (capacity_ < nbytes) {
      Assertf(false, "%s Failed to resize memory to %ld bytes. capacity = %ld", name_.c_str(),
              nbytes, capacity_);
    }
    bytes_ = nbytes;
  }

  void free_memory(cudaStream_t stream) {
    if (ptr_) {
      checkRuntime(cudaFreeHost(ptr_));
      ptr_ = nullptr;
      capacity_ = 0;
      bytes_ = 0;
    }
  }

 private:
  void *ptr_ = nullptr;
  size_t bytes_ = 0;
  size_t capacity_ = 0;
  std::string name_;
};

class GPUData {
 public:
  inline void *ptr() const { return ptr_; }
  inline size_t bytes() const { return bytes_; }
  inline bool empty() const { return ptr_ == nullptr; }
  virtual ~GPUData() { free_memory(nullptr); }
  GPUData() = default;
  GPUData(const std::string &name) { this->name_ = name; }

  void alloc_or_resize_to(size_t nbytes, cudaStream_t stream) {
    if (capacity_ < nbytes) {
      // dprintf("%s Free old %d, malloc new %d bytes.\n", name_.c_str(), capacity_, nbytes);
      free_memory(stream);
      checkRuntime(cudaMalloc(&ptr_, nbytes));
      capacity_ = nbytes;
    }
    bytes_ = nbytes;
  }

  void alloc(size_t nbytes, cudaStream_t stream) { alloc_or_resize_to(nbytes, stream); }

  void resize(size_t nbytes) {
    if (capacity_ < nbytes) {
      Assertf(false, "%s Failed to resize memory to %ld bytes. capacity = %ld", name_.c_str(),
              nbytes, capacity_);
    }
    bytes_ = nbytes;
  }

  void free_memory(cudaStream_t stream) {
    if (ptr_) {

#ifdef __WITH_QNX
      if(stream){
        checkRuntime(cudaFreeAsync(ptr_, (cudaStream_t)stream));
      }else{
        checkRuntime(cudaFree(ptr_));
      }
#else
      checkRuntime(cudaFree(ptr_));
#endif // __WITH_QNX

      ptr_ = nullptr;
      capacity_ = 0;
      bytes_ = 0;
    }
  }

 private:
  void *ptr_ = nullptr;
  size_t bytes_ = 0;
  size_t capacity_ = 0;
  std::string name_;
};

template <typename T>
class GPUMemory {
 public:
  T *ptr() const { return data_ ? (T *)data_->ptr() : nullptr; }
  size_t size() const { return size_; }
  size_t bytes() const { return data_ ? data_->bytes() : 0; }
  bool empty() const { return data_ == nullptr || data_->empty(); }
  bool unset() const { return data_ == nullptr; }
  // GPUMemory() { data_.reset(new GPUData()); }
  virtual ~GPUMemory() { data_.reset(); }
  void set_gpudata(std::shared_ptr<GPUData> data) { this->data_ = data; }

  void alloc_or_resize_to(size_t size, cudaStream_t stream) {
    if (data_) {
      size_ = size;
      data_->alloc_or_resize_to(size * sizeof(T), stream);
    } else {
      Asserts(false, "Failed to alloc or resize memory that because data is nullptr.");
    }
  }

  void alloc(size_t size, cudaStream_t stream) { alloc_or_resize_to(size, stream); }

  void resize(size_t size) {
    if (data_) {
      size_ = size;
      data_->resize(size * sizeof(T));
    } else {
      Asserts(false, "Failed to resize memory that because data is nullptr.");
    }
  }

 private:
  std::shared_ptr<GPUData> data_;
  size_t size_ = 0;
};

class GPUDataManager {
 public:
  std::shared_ptr<GPUData> query_or_alloc(const std::string &tensor_id) {
    std::shared_ptr<GPUData> &output = data_dict_[tensor_id];
    if (output == nullptr) {
      output.reset(new GPUData(tensor_id));
    }
    return output;
  }

 private:
  std::unordered_map<std::string, std::shared_ptr<GPUData>> data_dict_;
};

};  // namespace spconv

#endif  // #ifndef __SPCONV_MEMORY_HPP__