#pragma once

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "rwkv7_cuda_utils.h"

#define FULL_MASK 0xffffffff

template <typename F>
__global__ void rwkv7_wkv_backward_time_parallel_base_kernel(
    const int B,
    const int T,
    const int H,
    const int COARSE,
    const F* __restrict__ grad_BTHK,
    const F* __restrict__ r_BTHK,
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
    const int start_t = std::min(T - 1, -1 + (chunk_i + 1) * COARSE);
    const int end_t = chunk_i * COARSE;

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

    for (int t = start_t; t >= end_t; t--) {
        bool skip = skip_BT[get_index1(b, t, T)];
        float grad_x = to_float<F>(grad_BTHK[global_x]);
        float r_y = to_float<F>(r_BTHK[global_y]);
        float w_y = w_BTHK[get_index3(b, t, h, y, T, H, K)];
        float a_y = to_float<F>(a_BTHK[get_index3(b, t, h, y, T, H, K)]);
        float k_deformed_y = to_float<F>(k_deformed_BTHK[get_index3(b, t, h, y, T, H, K)]);

        auto mul_decay_remove_transpose = [&](float &A_xy) -> void {
            float A_xy_decayed = A_xy * w_y;
            float A_k_dot = A_xy * a_y * k_deformed_y;
            for (int offset = 16; offset > 0; offset /= 2) {
                A_k_dot += __shfl_down_sync(FULL_MASK, A_k_dot, offset);
            }
            A_k_dot = __shfl_sync(FULL_MASK, A_k_dot, 0);
            A_xy = A_xy_decayed - A_k_dot * k_deformed_y;
        };
        if (!skip) {
            mul_decay_remove_transpose(mul_xy);
            add_xy += grad_x * r_y;
            mul_decay_remove_transpose(add_xy);
        } else {
            float in_grad = grad_x * r_y;
            mul_decay_remove_transpose(in_grad);
            add_xy += in_grad;
        }

        global_x -= H * K;
        global_y -= H * K;
    }
    partial_mul_BMHKK[get_index4(b, chunk_i, h, x, y, M, H, K, K)] = mul_xy;
    partial_add_BMHKK[get_index4(b, chunk_i, h, x, y, M, H, K, K)] = add_xy;
}

template <int CHUNK_LEN, typename F>
__global__ void rwkv7_wkv_backward_time_parallel_final_kernel(
    const int B,
    const int T,
    const int H,
    const int COARSE,
    const F* __restrict__ grad_BTHK,
    const F* __restrict__ r_BTHK,
    const F* __restrict__ k_BTHK,
    const F* __restrict__ v_BTHK,
    const float* __restrict__ w_BTHK,
    const F* __restrict__ a_BTHK,
    const F* __restrict__ k_deformed_BTHK,
    const bool* __restrict__ skip_BT,
    const int M,
    const float* __restrict__ partial_add_BMHKK,
    const int L,
    const float* __restrict__ state_checkpoints_BLHKK,
    F* __restrict__ r_grad_BTHK,
    F* __restrict__ k_grad_BTHK,
    F* __restrict__ v_grad_BTHK,
    float* __restrict__ w_grad_BTHK,
    F* __restrict__ a_grad_BTHK,
    F* __restrict__ k_deformed_grad_BTHK
) {
    float state_xy_chunk[CHUNK_LEN];
    float state_prev_xy_chunk[CHUNK_LEN];
    const int K = 32;
    __shared__ float KK_state[32 * (32 + 1)];
    __shared__ float KK_state_prev[32 * (32 + 1)];
    __shared__ float KK_grad_decay_remove[32 * 32];
    __shared__ float KK_dS[32 * (32 + 1)];
    __shared__ float KK_dS_prev[32 * 32];
    __shared__ float KK_grad_decay[32 * (32 + 1)];
    __shared__ float K_k_deformed[32];
    __shared__ float K_a[32];
    int b = blockIdx.x;
    int h = blockIdx.y;
    int coarse_i = blockIdx.z;
    const int start_t = std::min(T - 1, -1 + (coarse_i + 1) * COARSE);
    const int end_t = coarse_i * COARSE;

    int x = threadIdx.y;
    int y = threadIdx.x;

    if (end_t == 0) {
        if (x == 0) {
            a_grad_BTHK[get_index3(b, 0, h, y, T, H, K)] = to_F<F>(0.0);
            k_deformed_grad_BTHK[get_index3(b, 0, h, y, T, H, K)] = to_F<F>(0.0);
        }
    }

    int checkpoint_start = start_t / CHUNK_LEN;
    int checkpoint_end = end_t / CHUNK_LEN;
    float dS_xy = 0.0;
    float dS_xy_prev = 0.0;
    float dS_xy_contrib = 0.0;
    float w_y_prev = 0.0;
    float a_y_prev = 0.0;
    float k_deformed_y_prev = 0.0;
    if (start_t < T - 1) {
        dS_xy_contrib = partial_add_BMHKK[get_index4(b, coarse_i + 1, h, x, y, M, H, K, K)];
    }
    if (x == 0) {
        K_k_deformed[y] = k_deformed_y_prev;
        K_a[y] = a_y_prev;
    }

    for (int l = checkpoint_start; l >= checkpoint_end; l--) {
        float state_xy = state_checkpoints_BLHKK[get_index4(b, l, h, x, y, L, H, K, K)];
        for (int c = 0; c < CHUNK_LEN; c++) {
            int t = l * CHUNK_LEN + c;
            if (t >= T) break;

            bool skip = skip_BT[get_index1(b, t, T)];
            state_prev_xy_chunk[c] = state_xy;
            float in_state_xy = state_xy;
            int64_t global_y = get_index3(b, t, h, y, T, H, K);
            int64_t global_x = get_index3(b, t, h, x, T, H, K);
            float k_y = to_float<F>(k_BTHK[global_y]);
            float v_x = to_float<F>(v_BTHK[global_x]);
            float w_y = w_BTHK[global_y];
            float a_y = to_float<F>(a_BTHK[global_y]);
            float k_deformed_y = to_float<F>(k_deformed_BTHK[global_y]);

            // compute decayed state value at (x, y)
            float state_xy_decayed = state_xy * w_y;
            float state_k_dot = state_xy * k_deformed_y;
            for (int offset = 16; offset > 0; offset /= 2) {
                state_k_dot += __shfl_down_sync(FULL_MASK, state_k_dot, offset);
            }

            state_k_dot = __shfl_sync(FULL_MASK, state_k_dot, 0);
            state_xy = state_xy_decayed - state_k_dot * a_y * k_deformed_y;
            state_xy += v_x * k_y;
            state_xy_chunk[c] = state_xy;
            if (skip) {
                state_xy = in_state_xy;
            }
        }

        for (int t = std::min(T - 1, (l + 1) * CHUNK_LEN - 1); t >= l * CHUNK_LEN; t--) {
            int c = t - l * CHUNK_LEN;
            float state_xy = state_xy_chunk[c];
            KK_state[get_index1(x, y, K+1)] = state_xy;
            KK_state_prev[get_index1(x, y, K+1)] = state_prev_xy_chunk[c];

            int64_t global_x = get_index3(b, t, h, x, T, H, K);
            int64_t global_y = get_index3(b, t, h, y, T, H, K);
            float r_y = to_float<F>(r_BTHK[global_y]);
            float k_y = to_float<F>(k_BTHK[global_y]);
            float v_y = to_float<F>(v_BTHK[global_y]);
            float w_y = w_BTHK[global_y];
            float a_y = to_float<F>(a_BTHK[global_y]);
            float k_deformed_x = to_float<F>(k_deformed_BTHK[global_x]);
            float k_deformed_y = to_float<F>(k_deformed_BTHK[global_y]);
            float grad_x = to_float<F>(grad_BTHK[global_x]);
            float grad_y = to_float<F>(grad_BTHK[global_y]);
            bool skip = skip_BT[get_index1(b, t, T)];

            float dS_xy = grad_x * r_y;
            if (!skip) {
                dS_xy += dS_xy_contrib;
                dS_xy_contrib = 0.0;
            }
            float dS_xy_decay = dS_xy * w_y;
            float dS_xy_remove = dS_xy * a_y * k_deformed_y;
            KK_dS[get_index1(x, y, K + 1)] = dS_xy;
            if (x == 0) {
                K_k_deformed[y] = k_deformed_y;
                K_a[y] = a_y;
            }

            __syncthreads(); // for KK_state, KK_dS

            float grad_decay_remove_xy = 0.0;
            for (int k = 0; k < K; k++) {
                grad_decay_remove_xy += KK_state_prev[get_index1(k, x, K+1)] * KK_dS[get_index1(k, y, K+1)];
            }
            if (x == y) {
                w_grad_BTHK[get_index3(b, t, h, x, T, H, K)] = grad_decay_remove_xy;
            }
            KK_grad_decay[get_index1(x, y, K+1)] = grad_decay_remove_xy;

            float state_mT_xy = KK_state[get_index1(y, x, K + 1)];
            float state_grad_dot = state_mT_xy * grad_y;
            float v_grad_x = dS_xy * k_y;
            float k_grad_x = KK_dS[get_index1(y, x, K + 1)] * v_y;

            // But we can still use 3x4 = 12 warps to do this on the tensor cores instead.
            for (int offset = 16; offset > 0; offset /= 2) {
                v_grad_x += __shfl_down_sync(FULL_MASK, v_grad_x, offset);
                k_grad_x += __shfl_down_sync(FULL_MASK, k_grad_x, offset);
                state_grad_dot += __shfl_down_sync(FULL_MASK, state_grad_dot, offset);
                dS_xy_remove += __shfl_down_sync(FULL_MASK, dS_xy_remove, offset);
            }
            if (y == 0) {
                v_grad_BTHK[get_index3(b, t, h, x, T, H, K)] = to_F<F>(v_grad_x);
                k_grad_BTHK[get_index3(b, t, h, x, T, H, K)] = to_F<F>(k_grad_x);
                r_grad_BTHK[get_index3(b, t, h, x, T, H, K)] = to_F<F>(state_grad_dot);
            }
            __syncthreads(); // For matrices prep for tensor ops
            float KK_grad_decay_yx = KK_grad_decay[get_index1(y, x, K+1)];
            float a_grad_x = -KK_grad_decay_yx * K_k_deformed[y];
            float k_deformed_t1 = -grad_decay_remove_xy * K_a[y] * K_k_deformed[y];
            float k_deformed_t2 = -K_a[x] * KK_grad_decay_yx * K_k_deformed[y];
            // TODO potential tensor core optimization
            for (int offset = 16; offset > 0; offset /= 2) {
                a_grad_x += __shfl_down_sync(FULL_MASK, a_grad_x, offset);
                k_deformed_t1 += __shfl_down_sync(FULL_MASK, k_deformed_t1, offset);
                k_deformed_t2 += __shfl_down_sync(FULL_MASK, k_deformed_t2, offset);
            }
            
            if (y == 0) {
                a_grad_BTHK[get_index3(b, t, h, x, T, H, K)] = to_F<F>(a_grad_x * K_k_deformed[x]);
                k_deformed_grad_BTHK[get_index3(b, t, h, x, T, H, K)] = to_F<F>(k_deformed_t1 + k_deformed_t2);
            }

            w_y_prev = w_y;
            a_y_prev = a_y;
            k_deformed_y_prev = k_deformed_y;
            dS_xy_remove = __shfl_sync(FULL_MASK, dS_xy_remove, 0);
            dS_xy_contrib += dS_xy_decay - dS_xy_remove * k_deformed_y;
            __syncthreads();
        }
    }
}