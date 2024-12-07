#ifndef TENSOR_MUL_H
#define TENSOR_MUL_H

#include "tensor_type.h"

bool tensor_matmul_2d(const Tensor* A, const Tensor* B, Tensor* C);  // 2D矩阵乘法 [M, K] × [K, N]
bool tensor_matmul_3d(const Tensor* A, const Tensor* B, Tensor* C);  // 3D张量乘法 [batch, M, K] × [batch, K, N]
bool tensor_matmul_4d(const Tensor* A, const Tensor* B, Tensor* C);  // 4D张量乘法 [b1, b2, M, K] × [b1, b2, K, N]


// 3D张量与2D权重相乘
// input: [batch_size, seq_len, model_dim]
// weight: [model_dim, model_dim]
// output: [batch_size, seq_len, model_dim]
bool tensor_mul_3_2(const Tensor* input, const Tensor* weight, Tensor* output);

// 4D张量乘法,K的最后两个维度要转置
// input1: [batch_size, num_heads, seq_len, head_dim]
// input2: [batch_size, num_heads, seq_len, head_dim]
// output: [batch_size, num_heads, seq_len, seq_len]
bool tensor_mul_4d_transpose(
    const Tensor* input1,
    const Tensor* input2,
    float scale,
    Tensor* output
);

#endif // TENSOR_MUL_H

