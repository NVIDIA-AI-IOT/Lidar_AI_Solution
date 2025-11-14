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

#ifndef ONNX_PARSER_HPP
#define ONNX_PARSER_HPP

#include <spconv/engine.hpp>

namespace spconv{

// Create an engine and load the weights from onnx file
std::shared_ptr<Engine> load_engine_from_onnx(
    const std::string& onnx_file,                         // the path to the onnx file that is exported by the specific script compatible with spconv, such as tools/deploy/export-scn.py.
    Precision inference_precision = Precision::Float16,   // the precision to use for model inference. It has high priority than the precision of each layer.
    bool sortmask = false,                        // only for ampere kernels.
    bool enable_blackwell = false,                // enable blackwell kernels for better performance. requires SM >= 100.
    bool with_auxiliary_stream = false,           // enable auxiliary stream to run the inference in a separate stream for better performance. better for blackwell kernels.
    unsigned int fixed_launch_points = 10000,     // only for cudagraph.
    void* stream = nullptr                        // the stream to use for the inference. Please avoid using the default stream.
);

}; // namespace spconv

#endif // ONNX_PARSER_HPP