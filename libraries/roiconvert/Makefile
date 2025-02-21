CUDA_VER?=
ifeq ($(CUDA_VER),)
  $(error "CUDA_VER is not set")
endif

cc        := g++
cuda_home := /usr/local/cuda-$(CUDA_VER)
cuda_arch := -gencode=arch=compute_80,code=sm_80
nvcc      := $(cuda_home)/bin/nvcc -ccbin=$(cc)

include_paths := -Isrc                \
    -I$(cuda_home)/include		  	  \
	-Iroi_conversion				  \
	-I../../dependencies/pybind11/include			      \
	-I/opt/conda3/include/python3.11/ \
	-I/usr/include/python3.8/ \
	-I/usr/include/python3.10

library_paths := -L$(cuda_home)/lib64 \
	-L/usr/lib/x86_64-linux-gnu/ \
	-L./ \
	-L/opt/conda3/lib/
	

cpp_compile_flags := -std=c++17 -w -g -O3 -fPIC -fopenmp -pthread $(include_paths)
cu_compile_flags  := -std=c++17 -w -g -O3 $(cuda_arch) -Xcompiler "$(cpp_compile_flags)" $(include_paths)
link_flags        := -pthread -fopenmp -Wl,-rpath='$$ORIGIN' $(library_paths) -lcudart -lcuda -Wl,-rpath='$(cuda_home)/lib64'

roiconv.so : objs/roi_conversion/roi_conversion.cu.o objs/src/python.cpp.o
	@echo Link $@
	@mkdir -p $(dir $@)
	@$(cc) -shared $^ -o $@ $(link_flags)

py : roiconv.so
test : roiconv.so
	@python unitests/test_nv12.py

test_all : roiconv.so
	@python unitests/test_all.py

test_resize : roiconv.so
	@python unitests/test_resize.py

test_perf : roiconv.so
	@python unitests/test_perf.py

objs/%.cpp.o : ./%.cpp
	@echo Compile CXX $<
	@mkdir -p $(dir $@)
	@$(cc) -c $< -o $@ $(cpp_compile_flags)

objs/%.cu.o : ./%.cu
	@echo Compile CUDA $<
	@mkdir -p $(dir $@)
	@$(nvcc) -c $< -o $@ $(cu_compile_flags)

clean :
	@rm -rf objs roiconv roiconv.so

.PHONY : clean run roiconv py

export PYTHONPATH=.
