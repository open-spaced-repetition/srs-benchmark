#include <iostream>
#include "rwkv7_cuda_utils.h"
#include "parallel_scan.h"

// for recurrences of the form f(i) = A_i * f(i-1) + B_i where A and B are matrices
// this function modifies the inputs in_mul and in_add in place.
template <bool FORWARD>
__global__ void rwkv7_scan_kernel(
    const int B,
    const int H,
    const int coarse,
    const int N,
    float* __restrict__ in_mul_BNHKK,
    float* __restrict__ in_add_BNHKK,
    const int M,
    float* __restrict__ partial_mul_BMHKK,
    float* __restrict__ partial_add_BMHKK
) {
    const int K = 32;
    __shared__ float mul_A[K * K];
    __shared__ float mul_B[K * K];
    __shared__ float add[K * K];
    int b = blockIdx.x;
    int h = blockIdx.y;
    int chunk_i = blockIdx.z;
    int x = threadIdx.y;
    int y = threadIdx.x;

    float next_mul_xy = 0.0;
    float next_add_xy = 0.0;
    auto iter = [&](int t) -> void {
        int64_t global_index = get_index4(b, t, h, x, y, N, H, K, K);
        mul_B[get_index1(x, y, K)] = in_mul_BNHKK[global_index];
        float t_add_xy = in_add_BNHKK[global_index];

        __syncthreads();
        next_mul_xy = 0.0;
        next_add_xy = 0;
        for (int k = 0; k < K; k++) {
            next_mul_xy += mul_A[get_index1(x, k, K)] * mul_B[get_index1(k, y, K)];
            next_add_xy += add[get_index1(x, k, K)] * mul_B[get_index1(k, y, K)];
        }
        next_add_xy += t_add_xy;
        __syncthreads();

        // overwrite the current mul and add global values
        in_mul_BNHKK[global_index] = next_mul_xy;
        in_add_BNHKK[global_index] = next_add_xy;

        // prepare for the next iteration
        mul_A[get_index1(x, y, K)] = next_mul_xy;
        add[get_index1(x, y, K)] = next_add_xy;
    };

    int start_t, end_t;
    if constexpr (FORWARD) {
        start_t = chunk_i * coarse;
        end_t = min(N-1, -1 + (chunk_i + 1) * coarse);
    } else {
        start_t = min(N-1, -1 + (chunk_i + 1) * coarse);
        end_t = chunk_i * coarse;
    }
    next_mul_xy = in_mul_BNHKK[get_index4(b, start_t, h, x, y, N, H, K, K)];
    next_add_xy = in_add_BNHKK[get_index4(b, start_t, h, x, y, N, H, K, K)];
    mul_A[get_index1(x, y, K)] = next_mul_xy;
    add[get_index1(x, y, K)] = next_add_xy;
    __syncthreads();
    if constexpr (FORWARD) {
        for (int t = start_t + 1; t <= end_t; t++) {
            iter(t);
        }
    } else {
        for (int t = start_t - 1; t >= end_t; t--) {
            iter(t);
        }
    }

    int64_t partial_index = get_index4(b, chunk_i, h, x, y, M, H, K, K);
    partial_mul_BMHKK[partial_index] = next_mul_xy;
    partial_add_BMHKK[partial_index] = next_add_xy;
}

template <bool FORWARD>
__global__ void rwkv7_add_kernel(
    const int B,
    const int H,
    const int partial_factor,
    const int M,
    // float* __restrict__ partial_mul_BMHKK,
    const float* __restrict__ partial_add_BMHKK,
    const int N,
    const float* __restrict__ out_mul_BNHKK, // This doesn't need to be modified
    float* __restrict__ out_add_BNHKK
) { 
    const int K = 32;
    __shared__ float partial_add[K * K];
    __shared__ float mul[K * K];
    int b = blockIdx.x;
    int h = blockIdx.y;
    int t = blockIdx.z;
    int x = threadIdx.y;
    int y = threadIdx.x;

    int partial_i;
    if constexpr (FORWARD) {
        partial_i = -1 + t / partial_factor;
        if (partial_i < 0) return;
    } else {
        partial_i = 1 + t / partial_factor;
        if (partial_i >= M) return;
    }
    partial_add[get_index1(x, y, K)] = partial_add_BMHKK[get_index4(b, partial_i, h, x, y, M, H, K, K)];
    mul[get_index1(x, y, K)] = out_mul_BNHKK[get_index4(b, t, h, x, y, N, H, K, K)];
    __syncthreads();

    float tot = 0.0;
    for (int k = 0; k < K; k++) {
        tot += partial_add[get_index1(x, k, K)] * mul[get_index1(k, y, K)];
    }
    out_add_BNHKK[get_index4(b, t, h, x, y, N, H, K, K)] += tot;
}

template <bool FORWARD>
void rwkv7_scan(
    const int B,
    const int N,
    const int H,
    float* in_mul_BNHKK,
    float* in_add_BNHKK
) {
    const int K = 32;
    const int COARSE = 5;
    const int M = (N + COARSE - 1) / COARSE;

    float *buffer;
    cudaMalloc(&buffer, sizeof(float) * 2 * B * M * H * K * K);
    float *partial_mul_BMHKK, *partial_add_BMHKK;
    partial_mul_BMHKK = buffer;
    partial_add_BMHKK = buffer + (int64_t) B * M * H * K * K;
    dim3 scan_grid_dim(B, H, M);
    dim3 scan_block_dim(32, 32);
    rwkv7_scan_kernel<FORWARD><<<scan_grid_dim, scan_block_dim>>>(B, H, COARSE, N, in_mul_BNHKK, in_add_BNHKK, M, partial_mul_BMHKK, partial_add_BMHKK);
    if (M > 1) {
        rwkv7_scan<FORWARD>(B, M, H, partial_mul_BMHKK, partial_add_BMHKK);
        // One block per element
        dim3 add_grid_dim(B, H, N);
        dim3 add_block_dim(32, 32);
        rwkv7_add_kernel<FORWARD><<<add_grid_dim, add_block_dim>>>(B, H, COARSE, M, partial_add_BMHKK, N, in_mul_BNHKK, in_add_BNHKK);
    }
    cudaFree(buffer);
}

void rwkv7_scan_forward(
    const int B,
    const int N,
    const int H,
    float* in_mul_BNHKK,
    float* in_add_BNHKK
) {
    return rwkv7_scan<true>(B, N, H, in_mul_BNHKK, in_add_BNHKK);
}

void rwkv7_scan_backward(
    const int B,
    const int N,
    const int H,
    float* in_mul_BNHKK,
    float* in_add_BNHKK
) {
    return rwkv7_scan<false>(B, N, H, in_mul_BNHKK, in_add_BNHKK);
}