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
 
#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <memory>
#include <numeric>

#include "onnx-parser.hpp"
#include <spconv/engine.hpp>
#include <spconv/memory.hpp>
#include <spconv/tensor.hpp>
#include <unordered_map>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

using namespace std;
namespace py = pybind11;

class SCNModel {
 public:
  SCNModel(string onnx, string precision) {
    instance_ = spconv::load_engine_from_onnx(
        onnx, precision == "fp16" ? spconv::Precision::Float16 : spconv::Precision::Int8);
    features_.set_gpudata(make_shared<spconv::GPUData>());
    indices_.set_gpudata(make_shared<spconv::GPUData>());
  }

  virtual ~SCNModel() {
    if (output_features_) {
      check_cuda_api(cudaFreeHost(output_features_));
      output_features_ = nullptr;
    }

    if (output_indices_) {
      check_cuda_api(cudaFreeHost(output_indices_));
      output_indices_ = nullptr;
    }
  }

  bool valid() { return instance_ != nullptr; }

  py::tuple forward(const py::array& features, const py::array& indices, std::vector<int> grid_size,
                    int64_t stream_) {
    if (!valid()) throw py::buffer_error("Invalid engine instance, please makesure your construct");

    int num = features.shape(0);
    int ndim = features.shape(1);
    features_.alloc_or_resize_to(num * ndim);
    indices_.alloc_or_resize_to(num * 4);

    cudaStream_t stream = (cudaStream_t)stream_;
    check_cuda_api(cudaMemcpyAsync(features_.ptr(), features.data(0), features_.bytes(),
                                 cudaMemcpyHostToDevice, stream));
    check_cuda_api(cudaMemcpyAsync(indices_.ptr(), indices.data(0), indices_.bytes(),
                                 cudaMemcpyHostToDevice, stream));

    instance_->input(0)->set_data(
      {num, ndim}, spconv::DataType::Float16, features_.ptr(), {num, 4},
      spconv::DataType::Int32, indices_.ptr(), grid_size
    );

    instance_->forward(stream);
    auto result = instance_->output(0);
    auto out_features = result->features();
    auto out_indices = result->indices();
    if (output_features_bytes_ < out_features.bytes()) {
      if (output_features_) check_cuda_api(cudaFreeHost(output_features_));
      output_features_bytes_ = out_features.bytes();
      check_cuda_api(cudaMallocHost(&output_features_, output_features_bytes_));
    }

    if (output_indices_bytes_ < out_indices.bytes()) {
      if (output_indices_) check_cuda_api(cudaFreeHost(output_indices_));
      output_indices_bytes_ = out_indices.bytes();
      check_cuda_api(cudaMallocHost(&output_indices_, output_indices_bytes_));
    }

    check_cuda_api(cudaMemcpyAsync(output_features_, out_features.ptr(), out_features.bytes(),
                                 cudaMemcpyDeviceToHost, stream));
    check_cuda_api(cudaMemcpyAsync(output_indices_, out_indices.ptr(), out_indices.bytes(),
                                 cudaMemcpyDeviceToHost, stream));
    check_cuda_api(cudaStreamSynchronize(stream));

    py::array result_features(py::dtype("float16"), out_features.shape, output_features_);
    py::array result_indices(py::dtype("float16"), out_indices.shape, output_indices_);
    return py::make_tuple(result_features, result_indices);
  }

 private:
  shared_ptr<spconv::Engine> instance_;
  spconv::GPUMemory<unsigned short> features_;
  spconv::GPUMemory<unsigned int> indices_;
  unsigned short* output_features_ = nullptr;
  size_t output_features_bytes_ = 0;
  unsigned int* output_indices_ = nullptr;
  size_t output_indices_bytes_ = 0;
};

PYBIND11_MODULE(pyscn, m) {
  py::class_<SCNModel, shared_ptr<SCNModel>>(m, "SCNModel")
      .def(py::init([](string onnx, string precision) {
             return make_shared<SCNModel>(onnx, precision);
           }),
           py::arg("onnx"), py::arg("precision"))
      .def(
          "forward",
          [](SCNModel& self, const py::array& features, const py::array& indices,
             std::vector<int> grid_size,
             int64_t stream) { return self.forward(features, indices, grid_size, stream); },
          py::arg("features"), py::arg("indices"), py::arg("grid_size"), py::arg("stream"));

  m.def("set_verbose", spconv::set_verbose, py::arg("enable"));
};