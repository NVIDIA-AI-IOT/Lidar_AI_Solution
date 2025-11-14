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
 
#ifndef __SPCONV_ENGINE_HPP__
#define __SPCONV_ENGINE_HPP__

#include <memory>
#include <string>
#include <vector>
#include <spconv/tensor.hpp>

namespace spconv {

#define Exported __attribute__((visibility("default")))

enum class Precision : int { None = 0, Float16 = 1, Int8 = 2 };
enum class TensorLayout : int { None = 0, NCHW = 1, NCHW32 = 2, NHWzC = 3 };
enum class LoggerLevel : int {Verb = 0, Warn = 1, Error = 2, Quiet = 99};

class ILogger{
public:
  virtual ~ILogger() = default;
  virtual void log(const char* message, LoggerLevel level, const char* file, int line) const = 0;
};

/**
  Storage of data tensor
**/
class SparseDTensor {
 public:
  virtual ~SparseDTensor(){}
  virtual Tensor& features() = 0;
  virtual Tensor& indices()  = 0;

  virtual void set_grid_size(const std::vector<int>& grid_size) = 0;
  virtual std::vector<int> grid_size() const = 0;
  virtual int device() const = 0;
  virtual const char* name() const = 0;
  virtual void set_dds_num_of_points_pointer(uint32_t* pointer) = 0;
  virtual uint32_t* get_dds_num_of_points_pointer() = 0;
};

/**
  Engine types for sparse convolution
**/
class Engine {
 public:
  Exported virtual ~Engine(){}
  /**
    Inference function for sparse convolution

    features_shape: The shape of the input feature matrix, it must be two elements.
    features_dtype: The data type of the input feature matrix, it must be Float16 now.
    features_data:  The data pointer of the input feature matrix
    indices_shape:  The shape of the input indices matrix, it must be two elements[n, 4]
    indices_dtype:  The data type of the input indices matrix, it must be Int32 now.
    indices_data:   The data pointer of the input indices matrix
    batch:          The batch size of the input, it must be 1 now.
    grid_size:      The grid size of the input data, For example: 41,1440,1440 or 1440,1440,41
    stream:         Which stream is expected to enqueue the inference.
  **/
  Exported virtual void forward(void* stream = nullptr) = 0;
  Exported virtual size_t num_input() const = 0;
  Exported virtual SparseDTensor* input(unsigned int index) = 0;
  Exported virtual size_t num_output() const = 0;
  Exported virtual SparseDTensor* output(unsigned int index) = 0;
};

class ITensor{
public:
  virtual ~ITensor(){}
  virtual const char* name() = 0;
};

class INode{
public:
  virtual ~INode(){}
  virtual const char* name() = 0;
  virtual const char* optype() = 0;
  virtual ITensor* input(unsigned int index) = 0;
  virtual ITensor* output(unsigned int index) = 0;

  virtual unsigned int num_output() = 0;
  virtual unsigned int num_input() = 0;
};

class EngineBuilder{
public:
  Exported virtual ~EngineBuilder(){}
  Exported virtual ITensor* push_input(const char* name) = 0;
  Exported virtual INode* push_add(
      const char* name, 
      ITensor* a, 
      ITensor* b,
      float a_dynamic_range,
      float b_dynamic_range,
      const char* output_name,
      Precision precision, Precision output_precision, uint32_t fixed_launch_points) = 0;

  Exported virtual INode* push_relu(
      const char* name, 
      ITensor* x, 
      const char* output_name) = 0;

  Exported virtual INode* push_dense(
      const char* name, ITensor* x,
      const char* format,
      const char* output_name,
      const std::vector<int>& input_spatial_shape,
      const std::vector<int>& output_shape,
      TensorLayout output_layout = TensorLayout::NCHW,
      float input_dynamic_range = 0.0f ,             // Enabled if int8 output is used,
      uint32_t input_bound = 0                     
  ) = 0;

  Exported virtual INode* push_reshape(
      const char* name, ITensor* x, 
      const std::vector<int64_t>& shape,
      const char* output_name) = 0;

  Exported virtual INode* push_transpose(
      const char* name, ITensor* x, 
      const std::vector<int64_t>& dims,
      const char* output_name) = 0;

  Exported virtual INode* push_sparse_conv(
      const char* name, 
      ITensor* x,
      const std::vector<unsigned short>& weight,
      const std::vector<int>& weight_shape,
      const std::vector<float>& weight_dynamic_ranges,
      const std::vector<unsigned short>& bias,
      const std::vector<int>& bias_shape,
      const char* activation,
      const std::vector<int>& kernel_size,
      const std::vector<int>& stride,
      const std::vector<int>& padding,
      const std::vector<int>& dilation,
      float input_dynamic_range,
      bool submanifold,
      uint32_t max_output_points,
      uint32_t fixed_launch_points,
      const char* rulebook,
      Precision precision,
      Precision output_precision,
      const char* output_name, 
      bool inverse) = 0;

  Exported virtual void push_output(ITensor* value) = 0;

  // build engine
  Exported virtual std::shared_ptr<Engine> build(Precision precision, bool sortmask=false, bool enable_blackwell=false, bool with_auxiliary_stream=false, void* stream = nullptr) = 0;
};

/**
 * To build a engine.
*/
Exported std::shared_ptr<EngineBuilder> create_engine_builder();

Exported const char* get_precision_string(Precision precision);
Exported const char* get_tensor_layout_string(TensorLayout layout);
Exported void set_logger_level(LoggerLevel level);
Exported void set_logger(ILogger* logger);
Exported LoggerLevel get_logger_level();
Exported ILogger* get_logger();
Exported const char* logger_level_string(LoggerLevel level);
Exported void logger_output(const char* file, int line, LoggerLevel level, const char* fmt, ...);

};  // namespace spconv

#endif  // #ifndef __SPCONV_ENGINE_HPP__
