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
    spconv_assertf(false, "Unsupport task name: %s", name.c_str());
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

// void run_task(const std::string& task_name, spconv::Precision precision, cudaStream_t stream) {
//   spconv::set_logger_level(spconv::LoggerLevel::Verb);
//   auto task = load_task(task_name, precision);
//   task.engine->input(0)->features().reference(task.features.ptr(), task.features.shape, spconv::DataType::Float16);
//   task.engine->input(0)->indices().reference(task.indices.ptr(), task.indices.shape, spconv::DataType::Int32);
//   task.engine->input(0)->set_grid_size(task.grid_size);
//   task.engine->forward(stream);

//   for(int i = 0; i < task.engine->num_output(); ++i){
//     auto& out_features = task.engine->output(0)->features();
//     printf("🙌 Output.shape: %s, Save to: %s\n", spconv::format_shape(out_features.shape), task.outputs[i].c_str());
//     out_features.save(task.outputs[i].c_str(), stream);
//   }
//   task.engine.reset();
//   print_done(task.compare_cmd);
// }

void run_task(const std::string& task_name, spconv::Precision precision, cudaStream_t stream) {
  spconv::set_logger_level(spconv::LoggerLevel::Verb);
  const char* precision_string = precision == spconv::Precision::Float16 ? "fp16" : "int8";
  printf("Run task: %s:%s\n", task_name.c_str(), precision_string);
  auto task = load_task(task_name, precision);
  auto features = task.features.clone();
  auto indices  = task.indices.clone();
  features.memset(0, stream);
  indices.memset(0, stream);
  task.engine->input(0)->features().reference(features.ptr(), features.shape, features.dtype(), true);
  task.engine->input(0)->indices().reference(indices.ptr(), indices.shape, indices.dtype(), true);
  task.engine->input(0)->set_grid_size(task.grid_size);

  bool use_cudagraph = false;
  const char* spconv_use_cudagraph = getenv("SPCONV_USE_CUDAGRAPH");
  if(spconv_use_cudagraph != nullptr && strcmp(spconv_use_cudagraph, "1") == 0){
    use_cudagraph = true;
    printf("Enable cudagraph because SPCONV_USE_CUDAGRAPH is set\n");
  }

  const char* dds = getenv("SPCONV_USE_DDS");
  if(use_cudagraph || (dds != nullptr && strcmp(dds, "1") == 0)){
    uint32_t* num_inputs_pointer = nullptr;
    uint32_t real_num_inputs = task.features.size(0);
    check_cuda_api(cudaMalloc(&num_inputs_pointer, sizeof(uint32_t)));
    check_cuda_api(cudaMemcpy(num_inputs_pointer, &real_num_inputs, sizeof(uint32_t) , cudaMemcpyHostToDevice));
    task.engine->input(0)->set_dds_num_of_points_pointer(num_inputs_pointer);
    printf("Set DDS num of points (%d) pointer to %p\n", real_num_inputs, num_inputs_pointer);
  }

  cudaGraph_t spconv_cuda_graph = nullptr;
  cudaGraphExec_t spconv_cuda_graph_instance = nullptr;
  if(use_cudagraph){
    check_cuda_api(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    task.engine->forward(stream);
    check_cuda_api(cudaStreamEndCapture(stream, &spconv_cuda_graph));
    check_cuda_api(cudaGraphInstantiate(&spconv_cuda_graph_instance, spconv_cuda_graph, nullptr, nullptr, 0));
  }

  check_cuda_api(cudaMemcpyAsync(features.ptr(), task.features.ptr(), features.bytes(), cudaMemcpyDeviceToDevice, stream));
  check_cuda_api(cudaMemcpyAsync(indices.ptr(), task.indices.ptr(), indices.bytes(), cudaMemcpyDeviceToDevice, stream));

  auto forward_func = [&](){
    if(use_cudagraph){
      check_cuda_api(cudaGraphLaunch(spconv_cuda_graph_instance, stream));
    }else{
      task.engine->forward(stream);
    }
  };

  const char* profile = getenv("PROFILE");
  bool profiling = profile != nullptr && strcmp(profile, "1") == 0;
  if(profiling){
    spconv::set_logger_level(spconv::LoggerLevel::Error);
    printf("Profiling task: %s:%s, warmup 10 times, iter 100 times\n", task_name.c_str(), precision_string);
    spconv::EventTimer timer;
    for(int i = 0; i < 100; ++i){
      forward_func();
    }
    timer.start(stream);
    for(int i = 0; i < 1000; ++i){
      forward_func();
    } 
    printf("Profiling task: %s:%s, time: %f ms\n", task_name.c_str(), precision_string, timer.stop(nullptr, false) / 1000.0f);
  }else{
    forward_func();
  }

  if(!profiling){
    for(int i = 0; i < task.engine->num_output(); ++i){
      printf("Save output[%d] to %s\n", i, task.outputs[i].c_str());
      task.engine->output(i)->features().save(task.outputs[i].c_str(), stream);
    }
  }
  task.engine.reset();

  if(!profiling){
    print_done(task.compare_cmd);
  }

  if(use_cudagraph){
    check_cuda_api(cudaGraphDestroy(spconv_cuda_graph));
    check_cuda_api(cudaGraphExecDestroy(spconv_cuda_graph_instance));
  }
}

int main(int argc, char** argv) {
  const char* cmd = "fp16";
  const char* task_name = "centerpointZYX";
  if (argc > 1) cmd = argv[1];
  if (argc > 2) task_name = argv[2];

  cudaStream_t stream = nullptr;
  check_cuda_api(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  if (strcmp(cmd, "int8") == 0) run_task(task_name, spconv::Precision::Int8, stream);
  if (strcmp(cmd, "fp16") == 0) run_task(task_name, spconv::Precision::Float16, stream);
  check_cuda_api(cudaStreamDestroy(stream));
  return 0;
}