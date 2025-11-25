/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
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
       free_memory(stream);
       check_cuda_api(cudaMallocHost(&ptr_, nbytes));
       capacity_ = nbytes;
     }
     bytes_ = nbytes;
   }
 
   void alloc(size_t nbytes, cudaStream_t stream) { alloc_or_resize_to(nbytes, stream); }
 
   void resize(size_t nbytes) {
     if (capacity_ < nbytes) {
       spconv_assertf(false, "%s Failed to resize memory to %ld bytes. capacity = %ld", name_.c_str(),
               nbytes, capacity_);
     }
     bytes_ = nbytes;
   }
 
   void free_memory(cudaStream_t stream) {
     if (ptr_) {
       check_cuda_api(cudaFreeHost(ptr_));
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
       free_memory(stream);
       check_cuda_api(cudaMalloc(&ptr_, nbytes));
       capacity_ = nbytes;
     }
     bytes_ = nbytes;
   }
 
   void alloc(size_t nbytes, cudaStream_t stream) { alloc_or_resize_to(nbytes, stream); }
 
   void resize(size_t nbytes) {
     if (capacity_ < nbytes) {
       spconv_assertf(false, "%s Failed to resize memory to %ld bytes. capacity = %ld", name_.c_str(),
               nbytes, capacity_);
     }
     bytes_ = nbytes;
   }
 
   void free_memory(cudaStream_t stream) {
     if (ptr_) {
 
 #ifdef __WITH_QNX
       if(stream){
         check_cuda_api(cudaFreeAsync(ptr_, (cudaStream_t)stream));
       }else{
         check_cuda_api(cudaFree(ptr_));
       }
 #else
       check_cuda_api(cudaFree(ptr_));
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
   virtual ~GPUMemory() { data_.reset(); }
   void set_gpudata(std::shared_ptr<GPUData> data) { this->data_ = data; }
 
   void alloc_or_resize_to(size_t size, cudaStream_t stream) {
     if (data_) {
       size_ = size;
       data_->alloc_or_resize_to(size * sizeof(T), stream);
     } else {
       spconv_asserts(false, "Failed to alloc or resize memory that because data is nullptr.");
     }
   }
 
   void alloc(size_t size, cudaStream_t stream) { alloc_or_resize_to(size, stream); }
 
   void resize(size_t size) {
     if (data_) {
       size_ = size;
       data_->resize(size * sizeof(T));
     } else {
       spconv_asserts(false, "Failed to resize memory that because data is nullptr.");
     }
   }
 
  private:
   std::shared_ptr<GPUData> data_;
   size_t size_ = 0;
 };
 
 class GPUDataManager {
  public:
   GPUDataManager(size_t max_bytes_for_io_tensor = 0){
     this->max_bytes_for_io_tensor_ = max_bytes_for_io_tensor;
   }
 
   std::shared_ptr<GPUData> query_or_alloc(const std::string &tensor_id, bool is_iotensor = false, cudaStream_t stream = nullptr) {
     if(tensor_id == "__no_reuse_policy_here__"){
       std::shared_ptr<GPUData> output;
       output.reset(new GPUData(tensor_id));
       return output;
     }
     std::shared_ptr<GPUData> &output = data_dict_[tensor_id];
     if (output == nullptr) {
       output.reset(new GPUData(tensor_id));
       if(is_iotensor && max_bytes_for_io_tensor_ > 0){
         output->alloc(max_bytes_for_io_tensor_, stream);
       }
     }
     return output;
   }
 
  private:
   size_t max_bytes_for_io_tensor_ = 0;
   std::unordered_map<std::string, std::shared_ptr<GPUData>> data_dict_;
 };
 
 };  // namespace spconv
 
 #endif  // #ifndef __SPCONV_MEMORY_HPP__