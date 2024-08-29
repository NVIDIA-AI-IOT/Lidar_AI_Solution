#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

project_folder=$(realpath $(dirname ${BASH_SOURCE[-1]}))
cd $project_folder

protoc=/usr/bin/protoc
mkdir -p pbout
$protoc onnx-ml.proto --cpp_out=pbout
$protoc onnx-operators-ml.proto --cpp_out=pbout

mv pbout/onnx-ml.pb.cc onnx-ml.pb.cpp
mv pbout/onnx-operators-ml.pb.cc onnx-operators-ml.pb.cpp
mv pbout/*.h ./

rm -rf pbout