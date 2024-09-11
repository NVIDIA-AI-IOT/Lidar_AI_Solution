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
#include <dirent.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <memory>
#include <numeric>
#include <stack>
#include <string>
#include <unordered_map>

#include "onnx-parser.hpp"
#include "spconv/engine.hpp"
#include "spconv/tensor.hpp"
#include "spconv/timer.hpp"
#include "spconv/version.hpp"
#include "voxelization.cuh"
#define strtok_s strtok_r

using namespace std;

struct Task {
  shared_ptr<spconv::Engine> engine;
  spconv::Tensor features;
  spconv::Tensor indices;
  vector<int> grid_size;
  string name;
  string compare_cmd;
  vector<string> outputs;
  IndiceOrder order;
};

Task load_task(const string& name, spconv::Precision precision) {
  Task task;
  task.name = name;
  if (name == "bevfusionXYZ") {
    task.engine = spconv::load_engine_from_onnx("bevfusion/bevfusion.scn.xyz.onnx", precision);
    task.features = spconv::Tensor::load("bevfusion/infer.xyz.voxels");
    task.indices = spconv::Tensor::load("bevfusion/infer.xyz.coors");
    task.grid_size = {1440, 1440, 41};
    task.order = IndiceOrder::XYZ;
    task.outputs = {"bevfusion/output.xyz.dense"};
    task.compare_cmd =
        "python tool/compare.py workspace/bevfusion/infer.xyz.dense workspace/bevfusion/output.xyz.dense --detail";
  } else if (name == "bevfusionZYX") {
    task.engine = spconv::load_engine_from_onnx("bevfusion/bevfusion.scn.zyx.onnx", precision);
    task.features = spconv::Tensor::load("bevfusion/infer.zyx.voxels");
    task.indices = spconv::Tensor::load("bevfusion/infer.zyx.coors");
    task.grid_size = {41, 1440, 1440};
    task.order = IndiceOrder::ZYX;
    task.outputs = {"bevfusion/output.zyx.dense"};
    task.compare_cmd =
        "python tool/compare.py workspace/bevfusion/infer.zyx.dense workspace/bevfusion/output.zyx.dense --detail";
  } else if (name == "centerpointZYX") {
    task.engine = spconv::load_engine_from_onnx("centerpoint/centerpoint.scn.PTQ.onnx", precision);
    task.features = spconv::Tensor::load("centerpoint/in_features.torch.fp16.tensor");
    task.indices = spconv::Tensor::load("centerpoint/in_indices_zyx.torch.int32.tensor");
    task.grid_size = {41, 1440, 1440};
    task.order = IndiceOrder::ZYX;
    task.outputs = {"centerpoint/output.zyx.dense"};
    task.compare_cmd =
        "python tool/compare.py workspace/centerpoint/out_dense.torch.fp16.tensor workspace/centerpoint/output.zyx.dense --detail";
  } else {
    Assertf(false, "Unsupport task name: %s", name.c_str());
  }
  return task;
}

void print_done(const string& cmd) {
  printf("[PASSED 🤗], libspconv version is %s\n", NVSPCONV_VERSION);
  printf(
      "To verify the results, you can execute the following command.\n"
      "  %s\n",
      cmd.c_str());
}

void run_task(const std::string& task_name, spconv::Precision precision, cudaStream_t stream) {
  spconv::set_verbose(true);
  auto task = load_task(task_name, precision);
  task.engine->input(0)->features().reference(task.features.ptr(), task.features.shape, spconv::DataType::Float16);
  task.engine->input(0)->indices().reference(task.indices.ptr(), task.indices.shape, spconv::DataType::Int32);
  task.engine->input(0)->set_grid_size(task.grid_size);
  task.engine->forward(stream);

  for(int i = 0; i < task.engine->num_output(); ++i){
    auto& out_features = task.engine->output(0)->features();
    printf("🙌 Output.shape: %s, Save to: %s\n", spconv::format_shape(out_features.shape), task.outputs[i].c_str());
    out_features.save(task.outputs[i].c_str(), stream);
  }
  task.engine.reset();
  print_done(task.compare_cmd);
}

int main(int argc, char** argv) {
  const char* cmd = "fp16";
  const char* task_name = "centerpointZYX";
  if (argc > 1) cmd = argv[1];
  if (argc > 2) task_name = argv[2];

  cudaStream_t stream = nullptr;
  checkRuntime(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  if (strcmp(cmd, "int8") == 0) run_task(task_name, spconv::Precision::Int8, stream);
  if (strcmp(cmd, "fp16") == 0) run_task(task_name, spconv::Precision::Float16, stream);
  checkRuntime(cudaStreamDestroy(stream));
  return 0;
}