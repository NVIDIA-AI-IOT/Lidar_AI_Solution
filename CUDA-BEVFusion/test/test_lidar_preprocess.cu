// Copyright (c) 2026 WATonomous. All rights reserved.
// SPDX-License-Identifier: MIT
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "bevfusion/lidar-preprocess.hpp"

static void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
static void check(cudaError_t error) { require(error == cudaSuccess, cudaGetErrorString(error)); }
template <class F>
static void invalid(F f) {
  bool rejected = false;
  try {
    f();
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "Malformed input was accepted");
}

int main() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) {
    std::cout << "SKIP: CUDA device unavailable\n";
    return 77;
  }
  try {
    using namespace bevfusion::lidar;
    invalid([] { create_preprocessor(0, 3); });
    auto preprocessor = create_preprocessor(8, 3);
    PointCloudLayout layout;
    layout.width = 4;
    layout.height = 2;
    layout.point_step = 29;  // Unaligned fields and organized row padding.
    layout.row_step = 128;
    layout.offsets[0] = 1;
    layout.offsets[1] = 9;
    layout.offsets[2] = 13;
    layout.offsets[3] = 21;
    std::vector<std::array<float, 4>> points = {{1.234f, 2, 3, 4},   {NAN, 2, 3, 4},  {5, 6, 7, 8},
                                                {1, 2, 3, INFINITY}, {9, 10, 11, 12}, {13, 14, 15, 16},
                                                {17, 18, 19, 20},    {1, 2, NAN, 4}};
    std::vector<unsigned char> host(layout.row_step * layout.height, 0xa5);
    void* device = nullptr;
    cudaStream_t stream = nullptr;
    check(cudaMalloc(&device, host.size()));
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    for (bool bigendian : {false, true}) {
      layout.is_bigendian = bigendian;
      for (size_t i = 0; i < points.size(); ++i) {
        for (int j = 0; j < 4; ++j) {
          unsigned char bytes[4];
          std::memcpy(bytes, &points[i][j], 4);
          auto* target = host.data() + (i / layout.width) * layout.row_step + (i % layout.width) * layout.point_step +
                         layout.offsets[j];
          for (int k = 0; k < 4; ++k) target[k] = bytes[bigendian ? 3 - k : k];
        }
      }
      check(cudaMemcpyAsync(device, host.data(), host.size(), cudaMemcpyHostToDevice, stream));
      auto prepared = preprocessor->forward(device, layout, stream);
      require(prepared.count == 3, "Wrong valid/truncated count");
      std::vector<__half> actual(prepared.count * 5);
      check(cudaMemcpyAsync(actual.data(), prepared.points, actual.size() * sizeof(__half), cudaMemcpyDeviceToHost,
                            stream));
      check(cudaStreamSynchronize(stream));
      const int expected_indices[] = {0, 2, 4};
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 5; ++j) {
          const __half expected = __float2half_rn(j == 4 ? 0.f : points[expected_indices[i]][j]);
          require(std::memcmp(&actual[i * 5 + j], &expected, sizeof(__half)) == 0,
                  "GPU preprocessing differs from stable CPU reference");
        }
      }
    }
    auto bad = layout;
    bad.height = 3;
    invalid([&] { preprocessor->forward(device, bad, stream); });
    bad = layout;
    bad.offsets[3] = layout.point_step - 1;
    invalid([&] { preprocessor->forward(device, bad, stream); });
    bad = layout;
    bad.row_step = 1;
    invalid([&] { preprocessor->forward(device, bad, stream); });
    PointCloudLayout empty;
    require(preprocessor->forward(nullptr, empty, stream).count == 0, "Empty cloud must be skipped");
    layout.is_bigendian = false;
    for (size_t i = 0; i < points.size(); ++i) {
      float value = NAN;
      std::memcpy(host.data() + (i / layout.width) * layout.row_step + (i % layout.width) * layout.point_step +
                      layout.offsets[0],
                  &value, 4);
    }
    check(cudaMemcpyAsync(device, host.data(), host.size(), cudaMemcpyHostToDevice, stream));
    require(preprocessor->forward(device, layout, stream).count == 0, "All-invalid cloud must be skipped");
    preprocessor.reset();
    check(cudaFree(device));
    check(cudaStreamDestroy(stream));
    std::cout << "PASS: stable filtering, FP16 equivalence, truncation, padding, endianness, invalid input\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
