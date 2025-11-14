/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following argument_descriptions:
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
 #include <sys/stat.h>
 #include <sys/types.h>
 #include <unistd.h>
 
 #include <algorithm>
 #include <memory>
 #include <limits>
 #include <string>
 #include <map>
 #include <sstream>

 #include "spconv/engine.hpp"
 #include "spconv/tensor.hpp"
 #include "spconv/timer.hpp"
 #include "onnx-parser.hpp"
 
 using namespace std;
 
 class CustomLogger : public spconv::ILogger{
 public:
   virtual void log(const char* message, spconv::LoggerLevel level, const char* file, int line) const override{
     printf("[%s:%d]: %s\n", file, line, message);
   }
 };
 
 class ArgumentsMap : public std::map<std::string, std::string>{
 public:
   std::map<std::string, std::tuple<std::string, std::string, std::string>> argument_descriptions;
   std::vector<std::tuple<std::string, std::string, std::string, std::string>> ordered_argument_descriptions;
   std::string program_name;
 
   bool parse(int argc, char** argv, const std::string& program_name, const std::vector<std::tuple<std::string, std::string, std::string, std::string>>& argument_descriptions){
     this->program_name = program_name;
     this->ordered_argument_descriptions = argument_descriptions;
     for(const auto& description : ordered_argument_descriptions){
       this->argument_descriptions[std::get<0>(description)] = std::make_tuple(std::get<1>(description), std::get<2>(description), std::get<3>(description));
     }

     for(int i = 1; i < argc; ++i){
       char* option = argv[i];
       std::string name = option;
       std::string value = "true";
       if(option[0] == '-' && option[1] == '-'){
         char* p = strchr(option, '=');
         if(p != nullptr){
           name = std::string(option + 2, p - (option + 2));
           value = std::string(p + 1);
         }else{
           name = std::string(option + 2);
           value = "true";
         }
       }

       if(this->argument_descriptions.find(name) != this->argument_descriptions.end()){
         (*this)[name] = value;
       }else{
         printf("Unknown argument [%s]\n", name.c_str());
         print_usage();
         return false;
       }
     }
 
     for(const auto& description : ordered_argument_descriptions){
       const std::string& argument_name = std::get<0>(description);
       const std::string& argument_description_type = std::get<1>(description);
       const std::string& argument_description_value = std::get<2>(description);
       const std::string& argument_description_description = std::get<3>(description);
       if(argument_description_type == "optional"){
         if(this->find(argument_name) == this->end()){
           (*this)[argument_name] = argument_description_value;
         }
       }
     }
     return check();
   }
 
   std::string get(const std::string& key, const std::string& default_value = "") const{
     auto it = this->find(key);
     if(it != this->end()){
       return it->second;
     }
     return default_value;
   }
 
   void print_usage(){
     printf("Usage:  %s ... \n", program_name.c_str());
     for(const auto& description : ordered_argument_descriptions){
       const std::string& argument_name = std::get<0>(description);
       const std::string& argument_description_type = std::get<1>(description);
       const std::string& argument_description_value = std::get<2>(description);
       const std::string& argument_description_description = std::get<3>(description);
       if(argument_description_type == "required"){
         printf("   --%s=value\t%s\n", argument_name.c_str(), argument_description_description.c_str());
       }else if(argument_description_type == "optional"){
         printf("   --%s=[default: %s]\t%s\n", argument_name.c_str(), argument_description_value.c_str(), argument_description_description.c_str());
       }
     }
   }
 
   bool check(){
     for(const auto& description : ordered_argument_descriptions){
       const std::string& argument_name = std::get<0>(description);
       const std::string& argument_description_type = std::get<1>(description);
       const std::string& argument_description_value = std::get<2>(description);
       if(this->find(argument_name) == this->end()){
         if(argument_description_type == "required"){
           printf("Missing required argument [--%s]\n", argument_name.c_str());
           print_usage();
           return false;
         }
       }
     }
     return true;
   }
 };

 struct InferenceTask {
    spconv::Tensor features;
    spconv::Tensor indices;
    vector<int> grid_size;
    spconv::Precision main_precision;
    std::string onnx_file;
    bool valid;
    unsigned int fixed_launch_points;
    bool fp16;
    bool int8;
    bool sortmask;
    bool enable_blackwell;
    bool with_auxiliary_stream;
    bool use_cudagraph;
    bool use_dds;
    bool profiling;
    bool verbosity;
    bool search_best_perf;
    float profiling_latency;
};

bool file_exists(const std::string& filename){
    struct stat buffer;
    return stat(filename.c_str(), &buffer) == 0;
}

std::vector<int> parse_grid_size(const std::string& grid_size_string){
    std::vector<int> grid_size;
    std::stringstream ss(grid_size_string);
    std::string token;
    while(getline(ss, token, ',')){
        grid_size.push_back(stoi(token));
    }
    return grid_size;
}

InferenceTask load_task_from_arguments(const ArgumentsMap& args, cudaStream_t stream){
    InferenceTask task;
    task.onnx_file = args.at("onnx").c_str();
    task.features = spconv::Tensor::load(args.at("feature").c_str(), true, stream);
    task.indices = spconv::Tensor::load(args.at("indice").c_str(), true, stream);
    task.grid_size = parse_grid_size(args.at("grid_size").c_str());
    task.fp16 = args.at("fp16") == "true";
    task.int8 = args.at("int8") == "true";
    task.main_precision = task.int8 ? spconv::Precision::Int8 : spconv::Precision::Float16;
    task.sortmask = args.at("sortmask") == "true";
    task.enable_blackwell = args.at("blackwell") == "true";
    task.with_auxiliary_stream = args.at("auxiliary_stream") == "true";
    task.use_cudagraph = args.at("cudagraph") == "true";
    task.use_dds = args.at("dds") == "true";
    task.profiling = args.at("profiling") == "true";
    task.verbosity = args.at("verbose") == "true";
    task.fixed_launch_points = stoi(args.at("fixed_points").c_str());
    task.search_best_perf = args.at("search_best_perf") == "true";
    task.valid = true;

    if(task.search_best_perf && task.profiling){
      printf("Ignore the search_best_perf flag because --profiling is specified.\n");
      task.search_best_perf = false;
    }

    if(task.search_best_perf || task.profiling){
      if(task.verbosity){
        printf("Disable verbosity mode because --search_best_perf or --profiling is specified.\n");
        task.verbosity = false;
      }
    }

    if(!file_exists(task.onnx_file)){
        printf("ONNX file does not exist: %s\n", task.onnx_file.c_str());
        task.valid = false;
    }

    if(task.features.empty()){
        printf("Failed to load features from file: %s\n", args.at("feature").c_str());
    }

    if(task.indices.empty()){
        printf("Failed to load indices from file: %s\n", args.at("indice").c_str());
    }

    if(task.features.empty() || task.indices.empty()){
      printf("Failed to run task because the features or indices are empty.\n");
      task.valid = false;
    }

    printf("=====================================================================\n");
    printf("Load inference task from arguments: %s\n", args.at("onnx").c_str());
    printf("  onnx: %s\n", task.onnx_file.c_str());
    printf("  feature: %s [%s] : %s\n", spconv::format_shape(task.features.shape, false), task.features.empty() ? "empty" : spconv::dtype_string(task.features.dtype()), args.at("feature").c_str());
    printf("  indice: %s [%s] : %s\n", spconv::format_shape(task.indices.shape, false), task.indices.empty() ? "empty" : spconv::dtype_string(task.indices.dtype()), args.at("indice").c_str());
    printf("  grid_size: %s\n", spconv::format_shape(task.grid_size, false));
    printf("  fp16: %s\n", task.fp16 ? "true" : "false");
    printf("  int8: %s\n", task.int8 ? "true" : "false");
    printf("  precision: %s\n", spconv::get_precision_string(task.main_precision));
    printf("  sortmask: %s\n", task.sortmask ? "true" : "false");
    printf("  blackwell: %s\n", task.enable_blackwell ? "true" : "false");
    printf("  auxiliary_stream: %s\n", task.with_auxiliary_stream ? "true" : "false");
    printf("  cudagraph: %s\n", task.use_cudagraph ? "true" : "false");
    printf("  fixed_points: %d\n", task.fixed_launch_points);
    printf("  profiling: %s\n", task.profiling ? "true" : "false");
    printf("  use_dds: %s\n", task.use_dds ? "true" : "false");
    printf("  verbose: %s\n", task.verbosity ? "true" : "false");
    printf("  search_best_perf: %s\n", task.search_best_perf ? "true" : "false");
    printf("=====================================================================\n");
    return task;
};

void run_task(InferenceTask& task, cudaStream_t stream) {
    if(!task.search_best_perf){
      printf("Run inference task: %s\n", task.onnx_file.c_str());
    }

    if(task.verbosity){
      spconv::set_logger_level(spconv::LoggerLevel::Verb);
    }else{
      spconv::set_logger_level(spconv::LoggerLevel::Error);
    }

    auto engine = spconv::load_engine_from_onnx(
      task.onnx_file, 
      task.main_precision, 
      task.sortmask, 
      task.enable_blackwell, 
      task.with_auxiliary_stream,
      task.fixed_launch_points,
      stream);

    if(engine == nullptr){
        printf("Failed to load engine from ONNX file: %s\n", task.onnx_file.c_str());
        return;
    }

    auto features = task.features.clone();
    auto indices  = task.indices.clone();
    features.memset(0, stream);  // not necessary call
    indices.memset(0, stream);   // not necessary call
    engine->input(0)->features().reference(features.ptr(), features.shape, features.dtype(), true);
    engine->input(0)->indices().reference(indices.ptr(), indices.shape, indices.dtype(), true);
    engine->input(0)->set_grid_size(task.grid_size);

    uint32_t* num_inputs_pointer = nullptr;
    if(task.use_cudagraph || task.use_dds){
        uint32_t real_num_inputs = task.features.size(0);
        check_cuda_api(cudaMalloc(&num_inputs_pointer, sizeof(uint32_t)));
        check_cuda_api(cudaMemcpyAsync(num_inputs_pointer, &real_num_inputs, sizeof(uint32_t) , cudaMemcpyHostToDevice, stream));
        engine->input(0)->set_dds_num_of_points_pointer(num_inputs_pointer);

        if(!task.search_best_perf){
          printf("Set DDS num of points (%d) pointer to %p\n", real_num_inputs, num_inputs_pointer);
        }
    }

    cudaGraph_t spconv_cuda_graph = nullptr;
    cudaGraphExec_t spconv_cuda_graph_instance = nullptr;
    if(task.use_cudagraph){
        check_cuda_api(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        engine->forward(stream);
        check_cuda_api(cudaStreamEndCapture(stream, &spconv_cuda_graph));
        check_cuda_api(cudaGraphInstantiate(&spconv_cuda_graph_instance, spconv_cuda_graph, nullptr, nullptr, 0));
    }

    // copy features and indices to the input pointers after cudagraph is captured.
    check_cuda_api(cudaMemcpyAsync(features.ptr(), task.features.ptr(), features.bytes(), cudaMemcpyDeviceToDevice, stream));
    check_cuda_api(cudaMemcpyAsync(indices.ptr(), task.indices.ptr(), indices.bytes(), cudaMemcpyDeviceToDevice, stream));

    auto forward_func = [&](){
        if(task.use_cudagraph){
            check_cuda_api(cudaGraphLaunch(spconv_cuda_graph_instance, stream));
        }else{
            engine->forward(stream);
        }
    };

    if(task.profiling){
        spconv::EventTimer timer;
        for(int i = 0; i < 100; ++i){
            forward_func();
        }
        timer.start(stream);
        for(int i = 0; i < 1000; ++i){
            forward_func();
        } 
        task.profiling_latency = timer.stop(nullptr, false) / 1000.0f;
        if(!task.search_best_perf){
          printf("Profiling task: %s:%s, latency: %f ms\n", 
              task.onnx_file.c_str(), 
              spconv::get_precision_string(task.main_precision), 
              task.profiling_latency);
        }
    }else{
        forward_func();
    }

    if(!task.profiling){
        for(int i = 0; i < engine->num_output(); ++i){
            std::string output_name = "output" + std::to_string(i) + "_" + std::string(engine->output(i)->name()) + ".tensor";
            printf("Save output[%d] to %s\n", i, output_name.c_str());
            engine->output(i)->features().save(output_name.c_str(), stream);
        }
    }
    engine.reset();

    if(!task.profiling){
        printf("Done inference task: %s\n", task.onnx_file.c_str());
    }

    if(task.use_cudagraph){
        check_cuda_api(cudaGraphDestroy(spconv_cuda_graph));
        check_cuda_api(cudaGraphExecDestroy(spconv_cuda_graph_instance));
    }

    if(num_inputs_pointer != nullptr){
      check_cuda_api(cudaFree(num_inputs_pointer));
      num_inputs_pointer = nullptr;
    }
}

unsigned int get_current_device_arch(){
  cudaDeviceProp prop;
  int device = 0;
  check_cuda_api(cudaGetDevice(&device));
  check_cuda_api(cudaGetDeviceProperties(&prop, device));
  return prop.major * 100 + prop.minor;
}

void search_best_perf(InferenceTask& task, cudaStream_t stream){
  printf("Search the best performance configuration for the model: %s\n", task.onnx_file.c_str());
  std::vector<bool> flags = {true, false};
  std::vector<bool> blackwell_flags = {true, false};
  float best_latency = std::numeric_limits<float>::max();
  typedef std::tuple<bool, unsigned int, bool, bool, bool, float> ConfigTuple;
  std::vector<ConfigTuple> sorted_config_list;
  int total_iterations = 32;
  int current_iteration = 0;
  unsigned int num_proposed_fixed_launch_points = task.features.size(0);
  unsigned int num_proposed_fixed_launch_points2 = (unsigned int)(num_proposed_fixed_launch_points * 0.2f) / 256 * 256;
  unsigned int num_proposed_fixed_launch_points3 = (unsigned int)(num_proposed_fixed_launch_points * 0.6f) / 256 * 256;
  unsigned int current_cuda_arch = get_current_device_arch();

  if(current_cuda_arch < 1000){
    blackwell_flags = {false};
    total_iterations /= 2;
    printf("Turn off profiling for blackwell kernels on current device (arch=%d).\n", current_cuda_arch);
  }
  printf("Proposed fixed launch points: %d, %d, %d\n", num_proposed_fixed_launch_points2, num_proposed_fixed_launch_points3, task.fixed_launch_points);

  for(bool cudagraph : flags){
    std::vector<unsigned int> fixed_launch_points_list;
    if(!cudagraph){
      fixed_launch_points_list = {task.fixed_launch_points};  // fixed launch points is only effective when cudagraph is disabled
    }else{
      fixed_launch_points_list = {num_proposed_fixed_launch_points2, num_proposed_fixed_launch_points3, task.fixed_launch_points};
    }

    for(unsigned int fixed_launch_points : fixed_launch_points_list){
      for(bool blackwell : blackwell_flags){
        for(bool sortmask : flags){
          for(bool auxiliary_stream : flags){
            task.enable_blackwell = blackwell;
            task.sortmask = sortmask;
            task.with_auxiliary_stream = auxiliary_stream;
            task.use_cudagraph = cudagraph;
            task.profiling = true;
            task.fixed_launch_points = fixed_launch_points;
            run_task(task, stream);

            if(task.profiling_latency < best_latency){
              best_latency = task.profiling_latency;
            }
            sorted_config_list.push_back({cudagraph, fixed_launch_points, blackwell, sortmask, auxiliary_stream, task.profiling_latency});

            current_iteration++;
            printf("  Iteration %d / %d: Cudagraph: %s, Fixed Launch Points: %d, Blackwell: %s, Sortmask: %s, Auxiliary Stream: %s, Latency: %.3f ms. Best latency: %.3f ms\n", 
                current_iteration, total_iterations,
                cudagraph ? "yes" : "no",
                fixed_launch_points,
                blackwell ? "yes" : "no", 
                sortmask ? "yes" : "no", 
                auxiliary_stream ? "yes" : "no",
                task.profiling_latency,
                best_latency
            );
          }
        }
      }
    }
  }

  std::sort(sorted_config_list.begin(), sorted_config_list.end(), [](
    const ConfigTuple& a, const ConfigTuple& b){
      return std::get<5>(a) < std::get<5>(b);
    }
  );

  printf("\n");
  auto best_config = sorted_config_list.front();
  printf("Best configuration found: \n   Cudagraph: %s, Fixed Launch Points: %d, Blackwell: %s, Sortmask: %s, Auxiliary Stream: %s, Latency: %.3f ms\n", 
    std::get<0>(best_config) ? "yes" : "no", 
    std::get<1>(best_config),
    std::get<2>(best_config) ? "yes" : "no", 
    std::get<3>(best_config) ? "yes" : "no", 
    std::get<4>(best_config) ? "yes" : "no", 
    std::get<5>(best_config)
  );

  printf("\n");

  printf("Sorted configuration list (%d configurations): \n", (int)sorted_config_list.size());
  for(size_t i = 0; i < sorted_config_list.size(); ++i){
    const ConfigTuple& config = sorted_config_list[i];
    printf("  %02d. Cudagraph: %s, Fixed Launch Points: %d, Blackwell: %s, Sortmask: %s, Auxiliary Stream: %s, Latency: %.3f ms\n", 
      (int)(i + 1),
      std::get<0>(config) ? "yes" : "no",
      std::get<1>(config),
      std::get<2>(config) ? "yes" : "no", 
      std::get<3>(config) ? "yes" : "no", 
      std::get<4>(config) ? "yes" : "no", 
      std::get<5>(config)
    );
  }
}

int main(int argc, char** argv) {
    ArgumentsMap args;
    if(!args.parse(argc, argv, "./infer", {
        {std::tuple("onnx", "required", "", "\t\t\tA ONNX file path that is exported by the specific script compatible with spconv.")},
        {std::tuple("feature", "required", "", "\t\tA feature tensor file path.")},
        {std::tuple("indice", "required", "", "\t\tAn indice tensor file path.")},
        {std::tuple("grid_size", "required", "", "\t\tThe input sparse grid size, such as 41,1440,1440 or 1440,1440,41.")},
        {std::tuple("fp16", "optional", "true", "\tThe main precision to use for the inference. If both fp16 and int8 are set to 1, the int8 will be used.")},
        {std::tuple("int8", "optional", "false", "\tThe main precision to use for the inference. If the layers of the model has no dynamic ranges, this flag will be ignored.")},
        {std::tuple("sortmask", "optional", "false", "\tEnable sortmask to skip more insignificance computations.")},
        {std::tuple("blackwell", "optional", "false", "\tEnable blackwell kernels for better performance.")},
        {std::tuple("auxiliary_stream", "optional", "false", "Enable auxiliary stream to run the inference in a separate stream for better performance.")},
        {std::tuple("cudagraph", "optional", "false", "\tEnable cudagraph to capture the inference graph for better performance.")},
        {std::tuple("fixed_points", "optional", "10000", "The fixed launch points for all kernels. Only effective when cudagraph is enabled.")},
        {std::tuple("profiling", "optional", "false", "\tEnable profiling to measure the inference latency.")},
        {std::tuple("verbose", "optional", "false", "\tEnable verbosity to print the inference details.")},
        {std::tuple("dds", "optional", "false", "\tEnable data dependency shape feature to the inference.")},
        {std::tuple("search_best_perf", "optional", "false", "Search the best performance configuration for the inference. This flag will be ignored if profiling is enabled.")},
    })){
        return 0;
    };

    cudaStream_t stream = nullptr;
    check_cuda_api(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    auto task = load_task_from_arguments(args, stream);
    if(!task.valid){
      printf("Skip the invalid inference task.\n");
      return 0;
    }

    if(task.search_best_perf && !task.profiling){
      search_best_perf(task, stream);
    }else{
      run_task(task, stream);
    }
    check_cuda_api(cudaStreamDestroy(stream));
    return 0;
}