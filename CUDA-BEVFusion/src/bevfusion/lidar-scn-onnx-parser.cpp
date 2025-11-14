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

#include "lidar-scn-onnx-parser.hpp"
#include "onnx/onnx-ml.pb.h"
#include "onnx/onnx-operators-ml.pb.h"
#include <fstream>
#include <numeric>
#include <unordered_map>
#include <algorithm>

namespace spconv{

#define LOGV(...)   spconv::logger_output(__FILE__, __LINE__, spconv::LoggerLevel::Verb, __VA_ARGS__)
#define LOGERR(...) spconv::logger_output(__FILE__, __LINE__, spconv::LoggerLevel::Error, __VA_ARGS__)

struct ParameterFP16Data{
    std::vector<unsigned short> data;
    std::vector<int> shape;
};

static onnx::TensorProto get_initializer(const onnx::GraphProto& graph, const std::string& name) {
    for (int i = 0; i < graph.initializer_size(); ++i) {
        auto& init = graph.initializer(i);
        if (init.name() == name) 
            return init;
    }
    LOGERR("Can not find initializer '%s' in ONNX.", name.c_str());
    return onnx::TensorProto();
};

static ParameterFP16Data get_initializer_data(const onnx::GraphProto& graph, const std::string& name) {
    auto proto = get_initializer(graph, name);
    if(proto.data_type() != onnx::TensorProto_DataType_FLOAT16){
        LOGERR("Can not support non float data type[%d] for initializer data.", proto.data_type());
        return ParameterFP16Data();
    }

    ParameterFP16Data output;
    output.shape.resize(proto.dims().size());
    std::transform(proto.dims().begin(), proto.dims().end(), output.shape.begin(), [](int64_t x){return (int)x;});

    size_t volumn = std::accumulate(output.shape.begin(), output.shape.end(), 1ul, std::multiplies<int64_t>());
    if(volumn * sizeof(unsigned short) != proto.raw_data().size()){
        LOGERR("Invalid parameter data size. %ld != %ld", volumn * sizeof(unsigned short), proto.raw_data().size());
        return ParameterFP16Data();
    }
    unsigned short* pdata = (unsigned short*)proto.raw_data().data();
    output.data = std::vector<unsigned short>(pdata, pdata + volumn);
    return output;
};

static bool has_attribute(const onnx::NodeProto& node, const std::string& name) {
    for (int i = 0; i < node.attribute_size(); ++i) {
        auto& attr = node.attribute(i);
        if (attr.name() == name) 
            return true;
    }
    return false;
};

static onnx::AttributeProto get_attribute(const onnx::NodeProto& node, const std::string& name) {
    for (int i = 0; i < node.attribute_size(); ++i) {
        auto& attr = node.attribute(i);
        if (attr.name() == name) 
            return attr;
    }
    LOGV("Can not find the attribute '%s' in the node '%s', the default value will be used.",
        name.c_str(), node.name().c_str());
    return onnx::AttributeProto();
};

static std::vector<int> get_attribute_as_intarray(const onnx::NodeProto& node, const std::string& name) {
    auto ints = get_attribute(node, name).ints();
    std::vector<int> output(ints.size());
    for (size_t i = 0; i < ints.size(); ++i) 
        output[i] = ints[i];
    return output;
};

std::shared_ptr<Engine> load_engine_from_onnx(const std::string& onnx_file, Precision precision, bool sortmask, bool enable_blackwell, bool with_auxiliary_stream, unsigned int fixed_launch_points, void* stream){

    LOGV("Load onnx from: %s", onnx_file.c_str());
    bool mark_all_output = false;
    onnx::ModelProto model;
    std::fstream fin(onnx_file, std::ios::binary | std::ios::in);
    if (!model.ParseFromIstream(&fin)) {
        LOGV("Parse onnx failed: %s", onnx_file.c_str());
        return nullptr;
    }

    auto builder = spconv::create_engine_builder();
    auto graph = model.graph();
    
    std::unordered_map<std::string, spconv::ITensor*> tensor_map_by_name;
    for (int i = 0; i < graph.input_size(); ++i) {
        auto name = graph.input(i).name();
        tensor_map_by_name[name] = builder->push_input(name.c_str());
    }

    std::vector<spconv::ITensor*> collect_outputs;
    for (int i = 0; i < model.graph().node_size(); ++i) {
        auto& node = model.graph().node(i);
        unsigned int local_fixed_launch_points = fixed_launch_points;
        if(has_attribute(node, "fixed_launch_points")){
            local_fixed_launch_points = get_attribute(node, "fixed_launch_points").i();
            LOGV("using fixed_launch_points from attribute: %d for node: %s", local_fixed_launch_points, node.name().c_str());
        }
        
        if (node.op_type() == "SparseConvolution") {
            auto x = tensor_map_by_name[node.input(0)];
            auto weight = get_initializer_data(graph, node.input(1));
            auto bias   = get_initializer_data(graph, node.input(2));
            auto weight_dynamic_ranges_proto = get_attribute(node, "weight_dynamic_ranges");
            auto weight_dynamic_ranges = 
                std::vector<float>(weight_dynamic_ranges_proto.floats().begin(), weight_dynamic_ranges_proto.floats().end());

            unsigned int output_bound = 150000;
            if(has_attribute(node, "output_bound")){
                output_bound = get_attribute(node, "output_bound").i();
            }else{
                LOGV("using default output_bound: %d", output_bound);
            }

            std::string rulebook = "";
            if(has_attribute(node, "rulebook")){
                rulebook = get_attribute(node, "rulebook").s();
            }

            auto n = builder->push_sparse_conv(
                node.name().c_str(), x, 
                weight.data, weight.shape,
                weight_dynamic_ranges,
                bias.data, bias.shape,
                get_attribute(node, "activation").s().c_str(),
                get_attribute_as_intarray(node, "kernel_size"),
                get_attribute_as_intarray(node, "stride"),
                get_attribute_as_intarray(node, "padding"),
                get_attribute_as_intarray(node, "dilation"),
                get_attribute(node, "input_dynamic_range").f(),
                get_attribute(node, "subm").i(),
                output_bound,
                local_fixed_launch_points,
                rulebook.c_str(),
                get_attribute(node, "precision").s() == "int8" ? Precision::Int8 : Precision::Float16,
                get_attribute(node, "output_precision").s() == "int8" ? Precision::Int8 : Precision::Float16,
                node.output(0).c_str(),
                get_attribute(node, "inverse").i()
            );

            if(mark_all_output){
                collect_outputs.push_back(n->output(0));
            }
            tensor_map_by_name[node.output(0)] = n->output(0);
        } else if (node.op_type() == "Add" || node.op_type() == "QuantAdd") {
            auto a = tensor_map_by_name[node.input(0)];
            auto b = tensor_map_by_name[node.input(1)];

            auto n = builder->push_add(
                node.name().c_str(),
                a, b, 
                get_attribute(node, "input0_dynamic_range").f(),
                get_attribute(node, "input1_dynamic_range").f(),
                node.output(0).c_str(), 
                get_attribute(node, "precision").s() == "int8" ? Precision::Int8 : Precision::Float16,
                get_attribute(node, "output_precision").s() == "int8" ? Precision::Int8 : Precision::Float16,
                fixed_launch_points
            );
            tensor_map_by_name[node.output(0)] = n->output(0);
        } else if (node.op_type() == "Relu") {
            auto x = tensor_map_by_name[node.input(0)];
            auto n = builder->push_relu(node.name().c_str(), x, node.output(0).c_str());
            tensor_map_by_name[node.output(0)] = n->output(0);
        } else if (node.op_type() == "ScatterDense") {
            auto x = tensor_map_by_name[node.input(0)];
            auto input_spatial_shape = get_attribute_as_intarray(node, "input_spatial_shape");
            auto output_shape = get_attribute_as_intarray(node, "output_shape");
            auto format = get_attribute(node, "format").s();
            auto input_dynamic_range = get_attribute(node, "input_dynamic_range").f();
            auto output_layout_string = get_attribute(node, "output_layout").s();
            auto layout = output_layout_string == "NCHW32" ? spconv::TensorLayout::NCHW32 : (output_layout_string == "NHWzC" ? spconv::TensorLayout::NHWzC : spconv::TensorLayout::NCHW);
            auto n = builder->push_dense(node.name().c_str(), x, format.c_str(), node.output(0).c_str(), input_spatial_shape, output_shape, layout, input_dynamic_range, fixed_launch_points);
            tensor_map_by_name[node.output(0)] = n->output(0);
        } else if (node.op_type() == "Reshape") {
            auto x = tensor_map_by_name[node.input(0)];
            auto dims = get_attribute(node, "dims");
            std::vector<int64_t> shape(dims.ints().begin(), dims.ints().end());
            auto n = builder->push_reshape(node.name().c_str(), x, shape, node.output(0).c_str());
            tensor_map_by_name[node.output(0)] = n->output(0);
        } else if (node.op_type() == "Transpose") {
            auto x = tensor_map_by_name[node.input(0)];
            auto dims = get_attribute(node, "dims");
            std::vector<int64_t> shape(dims.ints().begin(), dims.ints().end());
            auto n = builder->push_transpose(node.name().c_str(), x, shape, node.output(0).c_str());
            tensor_map_by_name[node.output(0)] = n->output(0);
        } else {
            printf("Unsupport operator [%s]\b", node.op_type().c_str());
            return nullptr;
        }
    }

    for (int i = 0; i < graph.output_size(); ++i) {
        auto name = graph.output(i).name();
        collect_outputs.push_back(tensor_map_by_name[name]);
    }

    for (int i = 0; i < collect_outputs.size(); ++i) {
        builder->push_output(collect_outputs[i]);
    }
    return builder->build(precision, sortmask, enable_blackwell, with_auxiliary_stream, stream);
}
};