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
 
#include "kernel.h"

#define HALF_PI  (3.141592653 * 0.5)

__global__ void predictKernel(
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
                            float *detections,
                            const float *post_center_range,
                            float out_size_factor,
                            const float *voxel_size,
                            const float *pc_range,
                            float score_threshold)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if(x >= H * W) return;
  
    int HW = H * W;
    int h = x / W;
    int w = x % W;

    for (int n = 0; n < N; n++) {

        int label = 0;
        float score = 1 / (1 + exp(-__half2float(hm[n * 0 * HW + 0 * HW + h * W + w])));

        for (int i = 1; i < C_hm; i++) {
            float sco = 1 / (1 + exp(-__half2float(hm[n * C_hm * HW + i * HW + h * W + w])));
            if (sco > score) {
                label = i;
                score = sco;
            }
        }

        if(score < score_threshold) continue;

        auto xs = __half2float(reg[n * C_reg * HW + h * W + w]) + w;
        auto ys = __half2float(reg[n * C_reg * HW + HW + h * W + w]) + h;

        xs = xs * out_size_factor * voxel_size[0] + pc_range[0];
        ys = ys * out_size_factor * voxel_size[1] + pc_range[1];

        auto zs = __half2float(height[n * C_height * HW + h * W + w]);

        if(xs < post_center_range[0] || xs > post_center_range[3]) continue;
        if(ys < post_center_range[1] || ys > post_center_range[4]) continue;
        if(zs < post_center_range[2] || zs > post_center_range[5]) continue;

        unsigned int curDet = 0;
        curDet = atomicAdd(detection_num, 1);

        if(curDet >= MAX_DET_NUM){
            *detection_num = MAX_DET_NUM;
            continue;
        }

        float3 dim_;
        dim_.x = exp(__half2float(dim[n * C_dim * HW + 0 * HW + h * W + w]));
        dim_.y = exp(__half2float(dim[n * C_dim * HW + 1 * HW + h * W + w]));
        dim_.z = exp(__half2float(dim[n * C_dim * HW + 2 * HW + h * W + w]));

        auto vx = __half2float(vel[n * C_vel * HW + 0 * HW + h * W + w]);
        auto vy = __half2float(vel[n * C_vel * HW + 1 * HW + h * W + w]);
        auto rs = atan2(__half2float(rot[n * C_rot * HW + h * W + w]), __half2float(rot[n * C_rot * HW + HW + h * W + w]));

        *(float3 *)(&detections[n * MAX_DET_NUM * DET_CHANNEL + DET_CHANNEL * curDet + 0]) = make_float3(xs, ys, zs);
        *(float3 *)(&detections[n * MAX_DET_NUM * DET_CHANNEL + DET_CHANNEL * curDet + 3]) = dim_;
        detections[n * MAX_DET_NUM * DET_CHANNEL + DET_CHANNEL * curDet + 6] = vx;
        detections[n * MAX_DET_NUM * DET_CHANNEL + DET_CHANNEL * curDet + 7] = vy;
        *(float3 *)(&detections[n * MAX_DET_NUM * DET_CHANNEL + DET_CHANNEL * curDet + 8]) = make_float3(rs, label, score);
    }
}

__device__ inline float cross(const float2 p1, const float2 p2, const float2 p0) {
    return (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y);
}

__device__ inline int check_box2d(float const *const box, const float2 p) {
    const float MARGIN = 1e-2;
    float center_x = box[0];
    float center_y = box[1];
    float angle_cos = cos(-box[6]);
    float angle_sin = sin(-box[6]);
    float rot_x = (p.x - center_x) * angle_cos + (p.y - center_y) * (-angle_sin);
    float rot_y = (p.x - center_x) * angle_sin + (p.y - center_y) * angle_cos;

    return (fabs(rot_x) < box[3] / 2 + MARGIN && fabs(rot_y) < box[4] / 2 + MARGIN);
}

__device__ inline bool intersection(const float2 p1, const float2 p0, const float2 q1, const float2 q0, float2 &ans) {

    if (( fmin(p0.x, p1.x) <= fmax(q0.x, q1.x) &&
          fmin(q0.x, q1.x) <= fmax(p0.x, p1.x) &&
          fmin(p0.y, p1.y) <= fmax(q0.y, q1.y) &&
          fmin(q0.y, q1.y) <= fmax(p0.y, p1.y) ) == 0)
        return false;


    float s1 = cross(q0, p1, p0);
    float s2 = cross(p1, q1, p0);
    float s3 = cross(p0, q1, q0);
    float s4 = cross(q1, p1, q0);

    if (!(s1 * s2 > 0 && s3 * s4 > 0))
        return false;

    float s5 = cross(q1, p1, p0);
    if (fabs(s5 - s1) > 1e-8) {
        ans.x = (s5 * q0.x - s1 * q1.x) / (s5 - s1);
        ans.y = (s5 * q0.y - s1 * q1.y) / (s5 - s1);

    } else {
        float a0 = p0.y - p1.y, b0 = p1.x - p0.x, c0 = p0.x * p1.y - p1.x * p0.y;
        float a1 = q0.y - q1.y, b1 = q1.x - q0.x, c1 = q0.x * q1.y - q1.x * q0.y;
        float D = a0 * b1 - a1 * b0;

        ans.x = (b0 * c1 - b1 * c0) / D;
        ans.y = (a1 * c0 - a0 * c1) / D;
    }

    return true;
}

__device__ inline void rotate_around_center(const float2 &center, const float angle_cos, const float angle_sin, float2 &p) {
    float new_x = (p.x - center.x) * angle_cos + (p.y - center.y) * (-angle_sin) + center.x;
    float new_y = (p.x - center.x) * angle_sin + (p.y - center.y) * angle_cos + center.y;
    p = float2 {new_x, new_y};
    return;
}

__device__ inline bool devIoU(float const *const box_a, float const *const box_b, const float nms_thresh) {
    float a_angle = box_a[6], b_angle = box_b[6];
    float a_dx_half = box_a[3] / 2, b_dx_half = box_b[3] / 2, a_dy_half = box_a[4] / 2, b_dy_half = box_b[4] / 2;
    float a_x1 = box_a[0] - a_dx_half, a_y1 = box_a[1] - a_dy_half;
    float a_x2 = box_a[0] + a_dx_half, a_y2 = box_a[1] + a_dy_half;
    float b_x1 = box_b[0] - b_dx_half, b_y1 = box_b[1] - b_dy_half;
    float b_x2 = box_b[0] + b_dx_half, b_y2 = box_b[1] + b_dy_half;
    float2 box_a_corners[5];
    float2 box_b_corners[5];

    float2 center_a = float2 {box_a[0], box_a[1]};
    float2 center_b = float2 {box_b[0], box_b[1]};

    float2 cross_points[16];
    float2 poly_center =  {0, 0};
    int cnt = 0;
    bool flag = false;

    box_a_corners[0] = float2 {a_x1, a_y1};
    box_a_corners[1] = float2 {a_x2, a_y1};
    box_a_corners[2] = float2 {a_x2, a_y2};
    box_a_corners[3] = float2 {a_x1, a_y2};

    box_b_corners[0] = float2 {b_x1, b_y1};
    box_b_corners[1] = float2 {b_x2, b_y1};
    box_b_corners[2] = float2 {b_x2, b_y2};
    box_b_corners[3] = float2 {b_x1, b_y2};

    float a_angle_cos = cos(a_angle), a_angle_sin = sin(a_angle);
    float b_angle_cos = cos(b_angle), b_angle_sin = sin(b_angle);

    for (int k = 0; k < 4; k++) {
        rotate_around_center(center_a, a_angle_cos, a_angle_sin, box_a_corners[k]);
        rotate_around_center(center_b, b_angle_cos, b_angle_sin, box_b_corners[k]);
    }

    box_a_corners[4] = box_a_corners[0];
    box_b_corners[4] = box_b_corners[0];

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            flag = intersection(box_a_corners[i + 1], box_a_corners[i],
                                box_b_corners[j + 1], box_b_corners[j],
                                cross_points[cnt]);
            if (flag) {
                poly_center = {poly_center.x + cross_points[cnt].x, poly_center.y + cross_points[cnt].y};
                cnt++;
            }
        }
    }

    for (int k = 0; k < 4; k++) {
        if (check_box2d(box_a, box_b_corners[k])) {
            poly_center = {poly_center.x + box_b_corners[k].x, poly_center.y + box_b_corners[k].y};
            cross_points[cnt] = box_b_corners[k];
            cnt++;
        }
        if (check_box2d(box_b, box_a_corners[k])) {
            poly_center = {poly_center.x + box_a_corners[k].x, poly_center.y + box_a_corners[k].y};
            cross_points[cnt] = box_a_corners[k];
            cnt++;
        }
    }

    poly_center.x /= cnt;
    poly_center.y /= cnt;

    float2 temp;
    for (int j = 0; j < cnt - 1; j++) {
        for (int i = 0; i < cnt - j - 1; i++) {
            if (atan2(cross_points[i].y - poly_center.y, cross_points[i].x - poly_center.x) >
                atan2(cross_points[i+1].y - poly_center.y, cross_points[i+1].x - poly_center.x)
                ) {
                temp = cross_points[i];
                cross_points[i] = cross_points[i + 1];
                cross_points[i + 1] = temp;
            }
        }
    }

    float area = 0;
    for (int k = 0; k < cnt - 1; k++) {
        float2 a = {cross_points[k].x - cross_points[0].x,
                    cross_points[k].y - cross_points[0].y};
        float2 b = {cross_points[k + 1].x - cross_points[0].x,
                    cross_points[k + 1].y - cross_points[0].y};
        area += (a.x * b.y - a.y * b.x);
    }

    float s_overlap = fabs(area) / 2.0;;
    float sa = box_a[3] * box_a[4];
    float sb = box_b[3] * box_b[4];
    float iou = s_overlap / fmaxf(sa + sb - s_overlap, 1e-8);

    return iou >= nms_thresh;
}

__global__ void nms_cuda(const int n_boxes, const float iou_threshold, const float *dev_boxes, uint64_t *dev_mask) {
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;
  const int tid = threadIdx.x;

  if (row_start > col_start) return;

  const int row_size = fminf(n_boxes - row_start * NMS_THREADS_PER_BLOCK, NMS_THREADS_PER_BLOCK);
  const int col_size = fminf(n_boxes - col_start * NMS_THREADS_PER_BLOCK, NMS_THREADS_PER_BLOCK);

  __shared__ float block_boxes[NMS_THREADS_PER_BLOCK * 7];

  if (tid < col_size) {
    block_boxes[tid * 7 + 0] = dev_boxes[(NMS_THREADS_PER_BLOCK * col_start + tid) * DET_CHANNEL + 0];
    block_boxes[tid * 7 + 1] = dev_boxes[(NMS_THREADS_PER_BLOCK * col_start + tid) * DET_CHANNEL + 1];
    block_boxes[tid * 7 + 2] = dev_boxes[(NMS_THREADS_PER_BLOCK * col_start + tid) * DET_CHANNEL + 2];
    block_boxes[tid * 7 + 3] = dev_boxes[(NMS_THREADS_PER_BLOCK * col_start + tid) * DET_CHANNEL + 3];
    block_boxes[tid * 7 + 4] = dev_boxes[(NMS_THREADS_PER_BLOCK * col_start + tid) * DET_CHANNEL + 4];
    block_boxes[tid * 7 + 5] = dev_boxes[(NMS_THREADS_PER_BLOCK * col_start + tid) * DET_CHANNEL + 5];
    block_boxes[tid * 7 + 6] = dev_boxes[(NMS_THREADS_PER_BLOCK * col_start + tid) * DET_CHANNEL + 6];
  }
  __syncthreads();

  if (tid < row_size) {
    const int cur_box_idx = NMS_THREADS_PER_BLOCK * row_start + tid;
    const float *cur_box = dev_boxes + cur_box_idx * DET_CHANNEL;
    int i = 0;
    uint64_t t = 0;
    int start = 0;
    if (row_start == col_start) {
      start = tid + 1;
    }
    for (i = start; i < col_size; i++) {
      if (devIoU(cur_box, block_boxes + i * 7, iou_threshold)) {
        t |= 1ULL << i;
      }
    }
    dev_mask[cur_box_idx * gridDim.y + col_start] = t;
  }
}

int postprocess_launch(
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
                    float *detections,
                    const float *post_center_range,
                    float out_size_factor,
                    const float *voxel_size,
                    const float *pc_range,
                    float score_threshold,
                    cudaStream_t stream)
{
    dim3 threads(32);
    dim3 blocks(H * W + threads.x - 1/ threads.x);

    predictKernel<<<blocks, threads, 0, stream>>>(N, H, W, C_reg, C_height, C_dim, C_rot, C_vel, C_hm,
                                                  reg, height, dim, rot, vel, hm, detection_num, detections,
                                                  post_center_range, out_size_factor, voxel_size, pc_range,
                                                  score_threshold);
    cudaStreamSynchronize(stream);
    auto err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        return -1;
    }

    return 0;
}

int nms_launch(unsigned int boxes_num,
               float *boxes_sorted,
               float nms_iou_threshold,
               uint64_t* mask,
               cudaStream_t stream)
{
    int col_blocks = DIVUP(boxes_num, NMS_THREADS_PER_BLOCK);

    dim3 blocks(col_blocks, col_blocks);
    dim3 threads(NMS_THREADS_PER_BLOCK);

    nms_cuda<<<blocks, threads, 0, stream>>>(boxes_num, nms_iou_threshold, boxes_sorted, mask);
    cudaStreamSynchronize(stream);

    auto err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        return -1;
    }

    return 0;
}

__global__ void permute_cuda(const float *boxes_sorted, 
                            float * permute_boxes_sorted, int boxes_num){

    const int tid = threadIdx.x;  
    if(tid < boxes_num){
        permute_boxes_sorted[tid * DET_CHANNEL + 0] = boxes_sorted[tid * DET_CHANNEL + 0];
        permute_boxes_sorted[tid * DET_CHANNEL + 1] = boxes_sorted[tid * DET_CHANNEL + 1];
        permute_boxes_sorted[tid * DET_CHANNEL + 2] = boxes_sorted[tid * DET_CHANNEL + 2];
        permute_boxes_sorted[tid * DET_CHANNEL + 3] = boxes_sorted[tid * DET_CHANNEL + 4];
        permute_boxes_sorted[tid * DET_CHANNEL + 4] = boxes_sorted[tid * DET_CHANNEL + 3];
        permute_boxes_sorted[tid * DET_CHANNEL + 5] = boxes_sorted[tid * DET_CHANNEL + 5];
        permute_boxes_sorted[tid * DET_CHANNEL + 6] = -boxes_sorted[tid * DET_CHANNEL + 8] - HALF_PI;
    }
}

int permute_launch(unsigned int boxes_num, 
                   const float *boxes_sorted,  
                   float * permute_boxes_sorted, 
                   cudaStream_t stream)
{
    dim3 blocks(1);
    dim3 threads(boxes_num);
    permute_cuda<<<blocks, threads, 0, stream>>>(boxes_sorted, permute_boxes_sorted, boxes_num);
    cudaStreamSynchronize(stream);

    auto err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        return -1;
    }

    return 0;
}