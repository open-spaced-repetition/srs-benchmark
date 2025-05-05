#pragma once

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "rwkv7_cuda_utils.h"

#define FULL_MASK 0xffffffff

template <typename F>
__global__ void rwkv7_wkv_forward_time_parallel_base_kernel(
    const int B,
    const int T,
    const int H,
    const int COARSE,
    const F* __restrict__ k_BTHK,
    const F* __restrict__ v_BTHK,
    const float* __restrict__ w_BTHK,
    const F* __restrict__ a_BTHK,
    const F* __restrict__ k_deformed_BTHK,
    const bool* __restrict__ skip_BT,
    const int M,
    float* __restrict__ partial_mul_BMHKK,
    float* __restrict__ partial_add_BMHKK
    ) {
    const int K = 32;
    int b = blockIdx.x;
    int h = blockIdx.y;
    int chunk_i = blockIdx.z;
    const int start_t = chunk_i * COARSE;
    const int end_t = std::min(T - 1, -1 + (chunk_i + 1) * COARSE);

    // Swapped for better memory coalescing. We want x to refer to rows and y to refer to columns and one entire row = 1 warp
    int x = threadIdx.y;
    int y = threadIdx.x;
    
    // Initialize as the identity matrix
    float mul_xy = 0.0; 
    if (x == y) mul_xy = 1.0;

    // Initialize as zeros
    float add_xy = 0.0;

    int64_t global_y = get_index3(b, start_t, h, y, T, H, K);
    int64_t global_x = get_index3(b, start_t, h, x, T, H, K);
    for (int t = start_t; t <= end_t; t++) {
        bool skip = skip_BT[get_index1(b, t, T)];
        if (!skip) {
            float k_y = to_float<F>(k_BTHK[global_y]);
            float v_x = to_float<F>(v_BTHK[global_x]);
            float w_y = w_BTHK[global_y];
            float a_y = to_float<F>(a_BTHK[global_y]);
            float k_deformed_y = to_float<F>(k_deformed_BTHK[global_y]);

            auto mul_decay_remove = [&](float &A_xy) -> void {
                float A_xy_decayed = A_xy * w_y;
                float A_k_dot = A_xy * k_deformed_y;
                for (int offset = 16; offset > 0; offset /= 2) {
                    A_k_dot += __shfl_down_sync(FULL_MASK, A_k_dot, offset);
                }
                A_k_dot = __shfl_sync(FULL_MASK, A_k_dot, 0);
                A_xy = A_xy_decayed - A_k_dot * a_y * k_deformed_y;
            };
            mul_decay_remove(mul_xy);
            mul_decay_remove(add_xy);
            add_xy += v_x * k_y;
        }
        global_x += H * K;
        global_y += H * K;
    }
    partial_mul_BMHKK[get_index4(b, chunk_i, h, x, y, M, H, K, K)] = mul_xy;
    partial_add_BMHKK[get_index4(b, chunk_i, h, x, y, M, H, K, K)] = add_xy;
}

template <int CHUNK_LEN, typename F>
__global__ void rwkv7_wkv_forward_time_parallel_final_kernel(
    const int B,
    const int T,
    const int H,
    const int COARSE,
    const F* __restrict__ r_BTHK,
    const F* __restrict__ k_BTHK,
    const F* __restrict__ v_BTHK,
    const float* __restrict__ w_BTHK,
    const F* __restrict__ a_BTHK,
    const F* __restrict__ k_deformed_BTHK,
    const bool* __restrict__ skip_BT,
    F* __restrict__ out_BTHK,
    const int M,
    const float* __restrict__ partial_add_BMHKK,
    const int L,
    float* __restrict__ state_checkpoints_BLHKK
) {
    const int K = 32;
    int b = blockIdx.x;
    int h = blockIdx.y;
    int chunk_i = blockIdx.z;
    const int start_t = chunk_i * COARSE;
    const int end_t = std::min(T - 1, -1 + (chunk_i + 1) * COARSE);
    int x = threadIdx.y;
    int y = threadIdx.x;

    float state_xy = 0.0;
    if (chunk_i > 0) {
        state_xy = partial_add_BMHKK[get_index4(b, chunk_i - 1, h, x, y, M, H, K, K)];
    }
    int64_t global_y = get_index3(b, start_t, h, y, T, H, K);
    int64_t global_x = get_index3(b, start_t, h, x, T, H, K);
    for (int t = start_t; t <= end_t; t++) {
        if (t % CHUNK_LEN == 0) {
            state_checkpoints_BLHKK[get_index4(b, t / CHUNK_LEN, h, x, y, L, H, K, K)] = state_xy;
        }
        // load in the relevant values
        float r_y = to_float<F>(r_BTHK[global_y]);
        float k_y = to_float<F>(k_BTHK[global_y]);
        float v_x = to_float<F>(v_BTHK[global_x]);
        float w_y = w_BTHK[global_y];
        float a_y = to_float<F>(a_BTHK[global_y]);
        float k_deformed_y = to_float<F>(k_deformed_BTHK[global_y]);
        bool skip = skip_BT[get_index1(b, t, T)];
        float in_state_xy = state_xy;

        // compute decayed state value at (x, y)
        float state_xy_decayed = state_xy * w_y;
        float state_k_dot = state_xy * k_deformed_y;
        // compute S@k. We do this in parallel at the row (warp) level
        // Parallel reduction: https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/
        for (int offset = 16; offset > 0; offset /= 2) {
            state_k_dot += __shfl_down_sync(FULL_MASK, state_k_dot, offset);
        }
        state_k_dot = __shfl_sync(FULL_MASK, state_k_dot, 0);
        state_xy = state_xy_decayed - state_k_dot * a_y * k_deformed_y;
        state_xy += v_x * k_y;
        // Compute S@r and store the result in out
        float state_r_dot = state_xy * r_y;
        for (int offset = 16; offset > 0; offset /= 2) {
            state_r_dot += __shfl_down_sync(FULL_MASK, state_r_dot, offset);
        }
        if (y == 0) {
            out_BTHK[global_x] = to_F<F>(state_r_dot);
        }
        if (skip) {
            state_xy = in_state_xy;
        }
        global_x += H * K;
        global_y += H * K;
    }
}