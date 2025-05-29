#pragma once

void rwkv7_scan_forward(
    const int B,
    const int N,
    const int H,
    float* in_mul_BNHKK,
    float* in_add_BNHKK);

void rwkv7_scan_backward(
    const int B,
    const int N,
    const int H,
    float* in_mul_BNHKK,
    float* in_add_BNHKK);
