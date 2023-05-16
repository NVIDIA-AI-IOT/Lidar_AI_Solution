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

optimize  := -O3
cc        := g++
stdcpp    := c++11
arch      := $(shell arch)
pwd       := $(abspath .)
cppstrict := -Wall -Werror -Wextra -Werror=all-warnings
custrict  := -Werror=all-warnings
nvcc      := $(CUDA_HOME)/bin/nvcc
cuda_arch := -gencode arch=compute_80,code=sm_80 \
			 -gencode arch=compute_86,code=sm_86

ifeq ($(arch), aarch64)
cuda_arch += -gencode arch=compute_87,code=sm_87
endif

include_paths     := -Isrc -Ilibspconv/include -I$(CUDA_HOME)/include
cpp_compile_flags := -std=$(stdcpp) $(strict) -g $(optimize) -fPIC -fopenmp -pthread $(include_paths)
cu_compile_flags  := -std=$(stdcpp) $(custrict) -g $(optimize) $(cuda_arch) -Xcompiler "$(cpp_compile_flags)"
link_flags        := -Llibspconv/lib/$(arch) -L$(CUDA_HOME)/lib64 -lcudart -lstdc++ -ldl -lspconv -pthread \
					 -fopenmp -Wl,-rpath='$$ORIGIN' -Wl,-rpath=$(pwd)/libspconv/lib/$(arch)

ifeq ($(MAKECMDGOALS), pyscn)
python_include := $(shell python -c "import sysconfig;print(sysconfig.get_path('include'))")
python_soname  := $(shell python -c "import sysconfig;import re;print(re.sub('lib|.so|.a', '', sysconfig.get_config_var('LIBRARY')))")
python_libpath := $(shell python -c "import sysconfig;print(sysconfig.get_config_var('LIBDIR'))")
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

int8      : pro
	@cd workspace && ./pro int8

fp16      : pro
	@cd workspace && ./pro fp16

memint8      : pro
	@cd workspace && ./pro memint8

memfp16      : pro
	@cd workspace && ./pro memfp16

tool/pyscn.so : objs/pyscn.o
	@echo Link $@
	@$(cc) -shared -o $@ $^ $(link_flags)
	@echo You can run \"python tool/pytest.py\" to test

workspace/pro : objs/main.o objs/voxelization.uo
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