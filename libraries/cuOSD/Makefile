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

CUDA_HOME :=/usr/local/cuda
cc        := g++
stdcpp    := c++17
nvcc      := $(CUDA_HOME)/bin/nvcc -ccbin=$(cc)

build_pango_backend    := no
build_truetype_backend := yes

libcuosd_objs := objs/src/cuosd.cpp.o          \
		objs/src/cuosd_kernel.cu.o             \
		objs/src/textbackend/backend.cpp.o     \
		objs/src/textbackend/backend.cpp.o     \
		objs/src/textbackend/pango-cairo.cpp.o \
		objs/src/textbackend/stb.cpp.o

# required cuda library
link_cuda      := cuda cudart
link_sys       := stdc++ dl

ifeq ($(build_pango_backend), yes)
link_sys += pango-1.0 cairo pangocairo-1.0 glib-2.0 gobject-2.0
cppdefine += -DENABLE_TEXT_BACKEND_PANGO
endif

ifeq ($(build_truetype_backend), yes)
cppdefine += -DENABLE_TEXT_BACKEND_STB
endif

link_librarys  := $(link_cuda) $(link_sys)

# include path
include_paths := src              \
	test			              \
	tool			              \
    $(CUDA_HOME)/include		  

ifeq ($(build_pango_backend), yes)
include_paths += 				  \
	/usr/include/			      \
	/mnt/deepstream/3rdparty/pango/1.38.1   \
	/mnt/deepstream/3rdparty/pango/1.38.1/pre-generated-headers/x86_64    \
	/mnt/deepstream/3rdparty/cairo/1.16.0/src							 \
	/mnt/deepstream/3rdparty/cairo/1.16.0/pre-generated-headers/x86_64	 \
	/usr/include/glib-2.0												 \
	/usr/lib/aarch64-linux-gnu/glib-2.0/include
endif

# library path
library_paths := $(CUDA_HOME)/lib64 /usr/lib/aarch64-linux-gnu/ $(shell pwd)

# set LD_LIBRARY_PATH
empty :=
library_path_export := $(subst $(empty) $(empty),:,$(library_paths))

# process include & library path
run_paths     := $(foreach item,$(library_paths),-Wl,-rpath=$(item))
include_paths := $(foreach item,$(include_paths),-I$(item))
library_paths := $(foreach item,$(library_paths),-L$(item))
link_librarys := $(foreach item,$(link_librarys),-l$(item))

# To modify cuda_arch based on device SM, add -gencode=arch=compute_87,code=sm_87 to cu_compile_flags
# https://developer.nvidia.com/zh-cn/cuda-gpus#compute
cpp_compile_flags := -std=$(stdcpp) -g -O0 -fPIC -fopenmp -pthread $(cppdefine) -Wall -Werror -Wextra -Werror=missing-declarations
cu_compile_flags  := -std=$(stdcpp) -g -O0 -Xcompiler "$(cpp_compile_flags)"
link_flags        := -pthread -fopenmp -Wl,-rpath='$$ORIGIN'

# CUDA code generation flags
GENCODE_SM53 := -gencode arch=compute_53,code=sm_53
GENCODE_SM62 := -gencode arch=compute_62,code=sm_62
GENCODE_SM72 := -gencode arch=compute_72,code=sm_72
GENCODE_SM75 := -gencode arch=compute_75,code=sm_75
GENCODE_SM86 := -gencode arch=compute_86,code=sm_86
GENCODE_SM87 := -gencode arch=compute_87,code=sm_87
GENCODE_SM_PTX := -gencode arch=compute_72,code=compute_72
GENCODE_FLAGS := $(GENCODE_SM53) $(GENCODE_SM62) $(GENCODE_SM72) $(GENCODE_SM75) $(GENCODE_SM86) $(GENCODE_SM87) $(GENCODE_SM_PTX)

cpp_compile_flags += $(include_paths)
cu_compile_flags  += $(include_paths) $(GENCODE_FLAGS)
link_flags        += $(library_paths) $(link_librarys) $(run_paths)

ifeq ($(MAKECMDGOALS), pycuosd)
python_include := $(shell python -c "import sysconfig;print(sysconfig.get_path('include'))")
python_soname  := $(shell python -c "import sysconfig;import re;print(re.sub('lib|.so|.a', '', sysconfig.get_config_var('LIBRARY')))")
python_libpath := $(shell python -c "import sysconfig;print(sysconfig.get_config_var('LIBDIR'))")
pybind_include := $(shell cd ../.. && pwd)/dependencies/pybind11/include
$(info Use Python Include: $(python_include))
$(info Use Python SO Name: $(python_soname))
$(info Use Python Library: $(python_libpath))
$(info Use PyBind11: $(pybind_include))
endif

ifneq ($(MAKECMDGOALS), clean)
libcuosd_mks := $(libcuosd_objs:.cpp.o=.cpp.mk)
libcuosd_mks := $(libcuosd_mks:.cu.o=.cu.mk)
-include $(libcuosd_mks)
endif

all      : cuosd
run      : cuosd
	@./cuosd ellipse

simple      : cuosd
	@./cuosd simple

test      : cuosd
	@./cuosd test

perf      : cuosd
	@./cuosd comp --text

pycuosd   : test/cuosd/pycuosd.so

test/cuosd/pycuosd.so : objs/test/pycuosd.cpp.o libcuosd.so
	@echo Link test/pycuosd.so
	@$(cc) objs/test/pycuosd.cpp.o -fPIC -shared -o $@ $(link_flags) -L$(python_libpath) -l$(python_soname) -L. -lcuosd

objs/test/pycuosd.cpp.o : test/pycuosd.cpp
	@echo Compile Python CXX $<
	@mkdir -p $(dir $@)
	@$(cc) -c $< -o $@ -I$(pybind_include) -I$(python_include) $(cpp_compile_flags)

cuosd : libcuosd.so objs/test/main.cpp.o objs/test/gpu_image.cu.o
	@echo Link $@
	@$(cc) objs/test/main.cpp.o objs/test/gpu_image.cu.o -o $@ $(link_flags) -L. -lcuosd

libcuosd.so : $(libcuosd_objs)
	@echo Link $@
	@$(cc) -fPIC -shared $^ -o $@ $(link_flags)

libcuosd.a : $(libcuosd_objs)
	@echo Link $@
	ar rcs $@ $^

objs/%.cpp.o : %.cpp
	@echo Compile CXX $<
	@mkdir -p $(dir $@)
	@$(cc) -c $< -o $@ $(cpp_compile_flags)

objs/%.cu.o : %.cu
	@echo Compile CUDA $<
	@mkdir -p $(dir $@)
	@$(nvcc) -c $< -o $@ $(cu_compile_flags)

objs/%.cpp.mk : %.cpp
	@echo Compile depends C++ $<
	@mkdir -p $(dir $@)
	@$(cc) -M $< -MF $@ -MT $(@:.cpp.mk=.cpp.o) $(cpp_compile_flags)

objs/%.cu.mk : %.cu
	@echo Compile depends CUDA $<
	@mkdir -p $(dir $@)
	@$(nvcc) -M $< -MF $@ -MT $(@:.cu.mk=.cu.o) $(cu_compile_flags)

clean :
	@rm -rf objs cuosd libcuosd.so libcuosd.a test/cuosd/pycuosd.so

.PHONY : clean run cuosd pycuosd

export LD_LIBRARY_PATH:=$(library_path_export)
