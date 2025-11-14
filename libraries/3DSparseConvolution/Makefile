# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# For example:
# CUDA_HOME=/usr/local/cuda SPCONV_CUDA_VERSION=12.8 make fp16 -j

optimize  := -O3
cc        := g++
stdcpp    := c++17
arch      := $(shell arch)
pwd       := $(abspath .)
cppstrict := #-Wall -Werror -Wextra -Werror=all-warnings
custrict  := #-Werror=all-warnings
nvcc      := $(CUDA_HOME)/bin/nvcc
cuda_arch := -gencode arch=compute_80,code=sm_80 \
			 -gencode arch=compute_86,code=sm_86

# For example: SPCONV_CUDA_VERSION=11.4  SPCONV_CUDA_VERSION=12.8
# If your application uses cuda-11.x, you can set SPCONV_CUDA_VERSION=11.4
# If your application uses cuda-12.x, you can set SPCONV_CUDA_VERSION=12.8
spconv_cuda_version := cuda$(SPCONV_CUDA_VERSION)
spconv_library_path := libspconv/lib/$(arch)_$(spconv_cuda_version)
$(info Use spconv_library_path: $(spconv_library_path))

ifeq ($(arch), aarch64)
cuda_arch += -gencode arch=compute_87,code=sm_87
endif

include_paths     := -Isrc -Ilibspconv/include -I$(CUDA_HOME)/include
cpp_compile_flags := -std=$(stdcpp) $(strict) -g $(optimize) -fPIC -fopenmp -pthread $(include_paths)
cu_compile_flags  := -std=$(stdcpp) $(custrict) -g $(optimize) $(cuda_arch) -Xcompiler "$(cpp_compile_flags)"
link_flags        := -L$(spconv_library_path) -L$(CUDA_HOME)/lib64 -lcudart -lstdc++ -ldl -lspconv -lprotobuf -pthread \
					 -fopenmp -Wl,-rpath='$$ORIGIN' -Wl,-rpath=$(pwd)/$(spconv_library_path)

ifeq ($(MAKECMDGOALS), pyscn)
python_include := $(shell python3 -c "import sysconfig;print(sysconfig.get_path('include'))")
python_soname  := $(shell python3 -c "import sysconfig;import re;print(re.sub('lib|.so|.a', '', sysconfig.get_config_var('LIBRARY')))")
python_libpath := $(shell python3 -c "import sysconfig;print(sysconfig.get_config_var('LIBDIR'))")
pybind_include := $(shell cd ../.. && pwd)/dependencies/pybind11/include
$(info Use Python Include: $(python_include))
$(info Use Python SO Name: $(python_soname))
$(info Use Python Library: $(python_libpath))
$(info Use PyBind11: $(pybind_include))
cpp_compile_flags += -I$(python_include) -I$(pybind_include)
cu_compile_flags  += -I$(python_include) -I$(pybind_include)
link_flags        += -L$(python_libpath) -l$(python_soname)
endif

all   : pro pyscn
pro   : workspace/pro
pyscn : tool/pyscn.so
infer : workspace/infer
	@cd workspace && ./infer --onnx=bevfusion/bevfusion.scn.xyz.onnx --feature=bevfusion/infer.xyz.voxels --indice=bevfusion/infer.xyz.coors --grid_size=1440,1440,41 --cudagraph --sortmask --auxiliary_stream
	@python tool/compare.py workspace/bevfusion/infer.xyz.dense workspace/output0_40.tensor --detail

best_perf : workspace/infer
	@cd workspace && ./infer --onnx=bevfusion/bevfusion.scn.xyz.onnx --feature=bevfusion/infer.xyz.voxels --indice=bevfusion/infer.xyz.coors --grid_size=1440,1440,41 --search_best_perf

int8      : pro
	@cd workspace && ./pro int8

fp16      : pro
	@cd workspace && ./pro fp16

memint8      : pro
	@cd workspace && ./pro memint8

memfp16      : pro
	@cd workspace && ./pro memfp16

tool/pyscn.so : objs/pyscn.o objs/onnx-parser.o objs/onnx/onnx-ml.pb.o objs/onnx/onnx-operators-ml.pb.o
	@echo Link $@
	@$(cc) -shared -o $@ $^ $(link_flags)
	@echo You can run \"python tool/pytest.py\" to test

workspace/pro : objs/main.o objs/onnx-parser.o objs/voxelization.uo objs/onnx/onnx-ml.pb.o objs/onnx/onnx-operators-ml.pb.o
	@echo Link $@
	@$(cc) -o $@ $^ $(link_flags)

workspace/infer : objs/infer.o objs/onnx-parser.o objs/voxelization.uo objs/onnx/onnx-ml.pb.o objs/onnx/onnx-operators-ml.pb.o
	@echo Link $@
	@$(cc) -o $@ $^ $(link_flags)

objs/%.o : src/%.cpp
	@echo Compile CXX $<
	@mkdir -p $(dir $@)
	@$(cc) -c $< -o $@ $(cpp_compile_flags)

objs/%.uo : src/%.cu
	@echo Compile CUDA $<
	@mkdir -p $(dir $@)
	@$(nvcc) -c $< -o $@ $(cu_compile_flags)

clean:
	@rm -rf objs workspace/pro tool/pyscn.so

.PHONY: pyscn pro clean

export LD_LIBRARY_PATH:=$(spconv_library_path):$(CUDA_HOME)/lib64:$(LD_LIBRARY_PATH)