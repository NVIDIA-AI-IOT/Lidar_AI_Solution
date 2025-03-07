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
#include <unordered_map>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "cuosd.h"
#include "memory.hpp"

#include <unordered_map>
#include <mutex>
#include <memory>

using namespace std;
namespace py = pybind11;

static mutex g_lock_;
static unordered_map<cuOSDContext_t, shared_ptr<cuOSDContext>> g_contexts_;

static cuOSDColor pytocolor(py::tuple color){
  if(color.size() > 4 || color.size() == 0) throw py::value_error("Invalid color size.");

  cuOSDColor ret;
  memset(&ret, 0, sizeof(ret));
  ret.a = 255;

  unsigned char* pr = (unsigned char*)&ret;
  for(size_t i = 0; i < color.size(); ++i){
    pr[i] = color[i].cast<unsigned char>();
  }
  return ret;
}

class cuOSD {
 public:
  cuOSD(){
    context_ = cuosd_context_create();

    std::unique_lock<mutex> l(g_lock_);
    g_contexts_.insert(make_pair(context_, shared_ptr<cuOSDContext>(context_, cuosd_context_destroy)));
  }

  virtual ~cuOSD(){
    
    if(context_){
      std::unique_lock<mutex> l(g_lock_);

      g_contexts_.erase(context_);
      context_ = nullptr;
    }
  }

  static void exit_cleanup(){
    std::unique_lock<mutex> l(g_lock_);
    g_contexts_.clear();
  }

  void set_backend(cuOSDTextBackend text_backend){
    cuosd_set_text_backend(context_, text_backend);
  }

  py::tuple measure_text(const char* utf8_text, int font_size, const char* font){
    int width, height, yoffset;
    cuosd_measure_text(context_, utf8_text, font_size, font, &width, &height, &yoffset);
    return py::make_tuple(width, height, yoffset);
  }

  void text(const char* utf8_text,int font_size, const char* font, int x, int y, py::tuple border_color, py::tuple bg_color){
    cuosd_draw_text(context_, utf8_text, font_size, font, x, y, pytocolor(border_color), pytocolor(bg_color));
  }

  void clock(cuOSDClockFormat format, long time, int font_size, const char* font, int x, int y, py::tuple border_color, py::tuple bg_color){
    cuosd_draw_clock(context_, format, time, font_size, font, x, y, pytocolor(border_color), pytocolor(bg_color));
  }

  void line(int x0, int y0, int x1, int y1, int thickness, py::tuple color, bool interpolation = true){
    cuosd_draw_line(context_, x0, y0, x1, y1, thickness, pytocolor(color), interpolation);
  }

  void arrow(int x0, int y0, int x1, int y1, int arrow_size, int thickness, py::tuple color, bool interpolation = false){
    cuosd_draw_arrow(context_, x0, y0, x1, y1, arrow_size, thickness, pytocolor(color), interpolation);
  }

  void point(int cx, int cy, int radius, py::tuple color){
    cuosd_draw_point(context_, cx, cy, radius, pytocolor(color));
  }

  void circle(int cx, int cy, int radius, int thickness, py::tuple border_color, py::tuple bg_color){
    cuosd_draw_circle(context_, cx, cy, radius, thickness, pytocolor(border_color), pytocolor(bg_color));
  }

  void rectangle(int left, int top, int right, int bottom, int thickness, py::tuple border_color, py::tuple bg_color){
    cuosd_draw_rectangle(context_, left, top, right, bottom, thickness, pytocolor(border_color), pytocolor(bg_color));
  }

  void boxblur(int left, int top, int right, int bottom, int kernel_size){
    cuosd_draw_boxblur(context_, left, top, right, bottom, kernel_size);
  }

  void rotationbox(int cx, int cy, int width, int height, float yaw, int thickness, py::tuple border_color, bool interpolation, py::tuple bg_color){
    cuosd_draw_rotationbox(context_, cx, cy, width, height, yaw, thickness, pytocolor(border_color), interpolation, pytocolor(bg_color));
  }

  void ellipse(int cx, int cy, int width, int height, float yaw, int thickness, py::tuple border_color, py::tuple bg_color){
    cuosd_draw_ellipse(context_, cx, cy, width, height, yaw, thickness, pytocolor(border_color), pytocolor(bg_color));
  }

  void segmentmask(int left, int top, int right, int bottom, int thickness, float* d_seg, int seg_width, int seg_height, float seg_threshold, py::tuple border_color, py::tuple seg_color){
    cuosd_draw_segmentmask(context_, left, top, right, bottom, thickness, d_seg, seg_width, seg_height, seg_threshold, pytocolor(border_color), pytocolor(seg_color));
  }

  void polyline(int* h_pts, int* d_pts, int n_pts, int thickness, bool is_closed, py::tuple border_color, bool interpolation, py::tuple fill_color){
    cuosd_draw_polyline(context_, h_pts, d_pts, n_pts, thickness, is_closed, pytocolor(border_color), interpolation, pytocolor(fill_color));
  }

  void rgba_source(void* d_src, int cx, int cy, int w, int h){
    cuosd_draw_rgba_source(context_, d_src, cx, cy, w, h);
  }

  void nv12_source(void* d_src0, void* d_src1, int cx, int cy, int w, int h, py::tuple mask_color, bool block_linear= false){
    cuosd_draw_nv12_source(context_, d_src0, d_src1, cx, cy, w, h, pytocolor(mask_color), block_linear);
  }

  void launch(uint64_t data0, uint64_t data1, int width, int stride, int height, cuOSDImageFormat format, uint64_t stream, bool gpu_memory){
    
    if(gpu_memory){
      cuosd_launch(context_, (void*)data0, (void*)data1, width, stride, height, format, (void*)stream);
      return;
    }

    if(format == cuOSDImageFormat::BlockLinearNV12){
      throw py::value_error("Invalid Image format");
    }

    if(format == cuOSDImageFormat::RGB){
      void* device_memory = nullptr;
      checkRuntime(cudaMalloc(&device_memory, width * height * 3));
      checkRuntime(cudaMemcpy2DAsync(
        device_memory, width * 3,
        (void*)data0, stride, width * 3, height, cudaMemcpyHostToDevice, (cudaStream_t)stream
      ));
      cuosd_launch(context_, device_memory, nullptr, width, width * 3, height, format, (void*)stream);
      checkRuntime(cudaMemcpy2DAsync(
        (void*)data0, stride, 
        device_memory, width * 3, width * 3, height, cudaMemcpyDeviceToHost, (cudaStream_t)stream
      ));
      checkRuntime(cudaStreamSynchronize((cudaStream_t)stream));
      checkRuntime(cudaFree(device_memory));
    }else if(format == cuOSDImageFormat::RGBA){
      void* device_memory = nullptr;
      checkRuntime(cudaMalloc(&device_memory, width * height * 4));
      checkRuntime(cudaMemcpy2DAsync(
        device_memory, width * 4,
        (void*)data0, stride, width * 4, height, cudaMemcpyHostToDevice, (cudaStream_t)stream
      ));
      cuosd_launch(context_, device_memory, nullptr, width, width * 4, height, format, (void*)stream);
      checkRuntime(cudaMemcpy2DAsync(
        (void*)data0, stride,
        device_memory, width * 4, width * 4, height, cudaMemcpyDeviceToHost, (cudaStream_t)stream
      ));
      checkRuntime(cudaStreamSynchronize((cudaStream_t)stream));
      checkRuntime(cudaFree(device_memory));
    }else if(format == cuOSDImageFormat::PitchLinearNV12){
      void* luma_memory   = nullptr;
      void* chroma_memory = nullptr;
      checkRuntime(cudaMalloc(&luma_memory,   width * height));
      checkRuntime(cudaMalloc(&chroma_memory, width * height / 2));
      checkRuntime(cudaMemcpy2DAsync(
        luma_memory, width,
        (void*)data0, stride, width, height, cudaMemcpyHostToDevice, (cudaStream_t)stream
      ));
      checkRuntime(cudaMemcpy2DAsync(
        chroma_memory, width,
        (void*)data0, stride, width, height / 2, cudaMemcpyHostToDevice, (cudaStream_t)stream
      ));
      cuosd_launch(context_, luma_memory, chroma_memory, width, width, height, format, (void*)stream);
      checkRuntime(cudaMemcpy2DAsync(
        (void*)data0, stride,
        luma_memory, width, width, height, cudaMemcpyDeviceToHost, (cudaStream_t)stream
      ));
      checkRuntime(cudaMemcpy2DAsync(
        (void*)data0, stride,
        chroma_memory, width, width, height / 2, cudaMemcpyDeviceToHost, (cudaStream_t)stream
      ));
      checkRuntime(cudaStreamSynchronize((cudaStream_t)stream));
      checkRuntime(cudaFree(luma_memory));
      checkRuntime(cudaFree(chroma_memory));
    }
  }

  void apply(uint64_t data0, uint64_t data1, int width, int stride, int height, cuOSDImageFormat format, uint64_t stream, bool gpu_memory, bool launchit){
    
    cuosd_apply(context_, (void*)data0, (void*)data1, width, stride, height, format, (cudaStream_t)stream, false);
    if(!launchit){
      return;
    }
    launch(data0, data1, width, stride, height, format, stream, gpu_memory);
    cuosd_clear(context_);
  }

 private:
  cuOSDContext_t context_ = nullptr;
};

PYBIND11_MODULE(pycuosd, m) {
  py::enum_<cuOSDTextBackend>(m, "Backend")
		.value("PangoCairo",   cuOSDTextBackend::PangoCairo)
		.value("StbTrueType",  cuOSDTextBackend::StbTrueType);

  py::enum_<cuOSDClockFormat>(m, "ClockFormat")
		.value("YYMMDD_HHMMSS", cuOSDClockFormat::YYMMDD_HHMMSS)
		.value("YYMMDD",  cuOSDClockFormat::YYMMDD)
		.value("HHMMSS",  cuOSDClockFormat::HHMMSS);

  py::enum_<cuOSDImageFormat>(m, "ImageFormat")
		.value("RGB", cuOSDImageFormat::RGB)
		.value("RGBA", cuOSDImageFormat::RGBA)
		.value("BlockLinearNV12",  cuOSDImageFormat::BlockLinearNV12)
		.value("PitchLinearNV12",  cuOSDImageFormat::PitchLinearNV12);

  py::class_<cuOSD, shared_ptr<cuOSD>>(m, "Context")
      .def(py::init([]() {
             return make_shared<cuOSD>();
      }))
      .def("set_backend", &cuOSD::set_backend)
      .def("measure_text", &cuOSD::measure_text)
      .def("text", &cuOSD::text)
      .def("clock", &cuOSD::clock)
      .def("line", &cuOSD::line)
      .def("arrow", &cuOSD::arrow)
      .def("point", &cuOSD::point)
      .def("circle", &cuOSD::circle)
      .def("rectangle", &cuOSD::rectangle)
      .def("boxblur", &cuOSD::boxblur)
      .def("rotationbox", &cuOSD::rotationbox)
      .def("segmentmask", &cuOSD::segmentmask)
      .def("polyline", &cuOSD::polyline)
      .def("rgba_source", &cuOSD::rgba_source)
      .def("nv12_source", &cuOSD::nv12_source)
      .def("apply", &cuOSD::apply)
      .def("launch", &cuOSD::launch);

  // to avoid the cuda exit error.
  m.def("exit_cleanup", &cuOSD::exit_cleanup);
};