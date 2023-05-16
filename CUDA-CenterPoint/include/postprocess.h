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
 
#ifndef POSTPROCESS_H_
#define POSTPROCESS_H_

#include <vector>
#include "common.h"
#include "kernel.h"

/*
box_encodings: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading or *[cos, sin], ...]
anchors: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
*/
struct Bndbox {
    float x;
    float y;
    float z;
    float w;
    float l;
    float h;
    float vx;
    float vy;
    float rt;
    int id;
    float score;
    Bndbox(){};
    Bndbox(float x_, float y_, float z_, float w_, float l_, float h_, float vx_, float vy_, float rt_, int id_, float score_)
        : x(x_), y(y_), z(z_), w(w_), l(l_), h(h_), vx(vx_), vy(vy_), rt(rt_), id(id_), score(score_) {}
};

class PostProcessCuda {
  private:
    Params params_;
    float* d_post_center_range_ = nullptr;
    float* d_voxel_size_ = nullptr;
    float* d_pc_range_ = nullptr;

  public:
    PostProcessCuda();
    ~PostProcessCuda();

    int doPostDecodeCuda(
      int N,
      int H,
      int W,
      int C_reg,
      int C_height,
      int C_dim,
      int C_rot,
      int C_vel,
      int C_hm,
      const half *reg,
      const half *height,
      const half *dim,
      const half *rot,
      const half *vel,
      const half *hm,
      unsigned int *detection_num,
      float *detections, cudaStream_t stream);

    int doPostNMSCuda(
      unsigned int boxes_num,
      float *boxes_sorted,
      uint64_t* mask, cudaStream_t stream);

    int doPermuteCuda(
        unsigned int boxes_num, 
        const float *boxes_sorted, 
        float * permute_boxes, cudaStream_t stream);
};

#endif