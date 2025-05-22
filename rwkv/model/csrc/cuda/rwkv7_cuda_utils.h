#pragma once

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define FULL_MASK 0xffffffff

template <typename F>
__host__ __device__ inline float to_float(F x) {
    if constexpr (std::is_same_v<F, __half>) {
        return __half2float(x);
    } else if constexpr (std::is_same_v<F, __nv_bfloat16>) {
        return __bfloat162float(x);
    } else {
        return x;
    }
}
template <typename F>
__host__ __device__ inline F to_F(float x) {
    if constexpr (std::is_same_v<F, __half>) {
        return __float2half(x);
    } else if constexpr (std::is_same_v<F, __nv_bfloat16>) {
        return __float2bfloat16(x);
    } else {
        return x;
    }
}

__device__ inline int64_t get_index1(int b, int t, int T) {
    return (int64_t) b * T + t;
}

__device__ inline int64_t get_index2(int b, int t, int h, int T, int H) {
    return ((int64_t) b * T + t) * H + h;
}

__device__ inline int64_t get_index3(int b, int t, int h, int k, int T, int H, int K) {
    return (((int64_t) b * T + t) * H + h) * K + k;
}

__device__ inline int64_t get_index4(int b, int t, int h, int k, int k2, int T, int H, int K, int K2) {
    return ((((int64_t) b * T + t) * H + h) * K + k) * K2 + k2;
}