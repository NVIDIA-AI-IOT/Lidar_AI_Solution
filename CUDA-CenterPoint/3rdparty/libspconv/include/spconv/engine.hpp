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
 
#ifndef __SPCONV_ENGINE_HPP__
#define __SPCONV_ENGINE_HPP__

#include <memory>
#include <string>
#include <vector>

namespace spconv {

#define Exported __attribute__((visibility("default")))

enum class DType : int { None = 0, Int32 = 1, Float16 = 2 };
enum class Precision : int { None = 0, Float16 = 1, Int8 = 2 };

/**
  Storage of data tensor
**/
class DTensor {
 public:
  virtual std::vector<int64_t> features_shape() const = 0;
  virtual DType features_dtype() const = 0;
  virtual void* features_data() = 0;

  virtual std::vector<int64_t> indices_shape() const = 0;
  virtual DType indices_dtype() const = 0;
  virtual void* indices_data() = 0;

  virtual std::vector<int> grid_size() const = 0;
  virtual int device() const = 0;
};

/**
  Engine types for sparse convolution
**/
class Engine {
 public:
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
  Exported virtual DTensor* forward(const std::vector<int64_t>& features_shape,
                                    DType features_dtype, void* features_data,
                                    const std::vector<int64_t>& indices_shape, DType indices_dtype,
                                    void* indices_data, int batch, std::vector<int> grid_size,
                                    void* stream = nullptr) = 0;

  // If you change the precision of a node after loading the model, you should call this function to
  // reconfigure it
  Exported virtual void reconfigure() = 0;

  // If you want to execute an implicit PTQ calibration, you can enable int8calibration by marking
  // it and collecting the maximum value of the tensor in the next forward.
  Exported virtual void set_int8_calibration(bool enable) = 0;

  // You can modify the precision of a node with this function, but don't forget to call reconfigure
  Exported virtual void set_node_precision_byname(const char* name, Precision compute_precision,
                                                  Precision output_precision) = 0;
  Exported virtual void set_node_precision_byoptype(const char* optype, Precision compute_precision,
                                                    Precision output_precision) = 0;
};

/**
  Create an engine and load the weights from onnx file

  onnx_file: Store the onnx of model structure, please use tool/deploy/export-scn.py to export the
corresponding onnx precision: What precision to use for model inference. For each layer's precision
should be stored in the "precision" attribute of the layer
            - Model inference will ignore the "precision" attribute of each layer what if set to
Float16
**/
Exported std::shared_ptr<Engine> load_engine_from_onnx(const std::string& onnx_file,
                                                       Precision precision = Precision::Float16);

/**
  Enable detailed information output

  enable: You should set this to true if you want to debug the model inference process. default:
  false
*/
Exported void set_verbose(bool enable);
Exported const char* get_precision_string(Precision precision);

};  // namespace spconv

#endif  // #ifndef __SPCONV_ENGINE_HPP__