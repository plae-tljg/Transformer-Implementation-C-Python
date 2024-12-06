#include "tensor_check.h"
#include <stdbool.h>

// this file seem useless now

// 检查矩阵乘法维度是否匹配, just for matrix multiplication, not matrix-vector multiplication
bool check_matmul_dims(int A_dim, int B_dim, const int* A_shape, const int* B_shape, int* out_shape, int* out_ndims) {
    // 快速检查维度数
    if (A_dim < 2 || B_dim < 2) {
        return false;
    }
    
    // 检查最后两个维度是否匹配
    // A: [..., M, K]  B: [..., K, N]
    if (A_shape[A_dim - 1] != B_shape[B_dim - 2]) {
        return false;
    }

    // 检查剩余维度是否可广播
    int batch_dims_a = A_dim - 2;
    int batch_dims_b = B_dim - 2;
    int max_batch_dims = batch_dims_a > batch_dims_b ? batch_dims_a : batch_dims_b;
    
    // 计算输出形状
    *out_ndims = max_batch_dims + 2;
    
    // 一次遍历同时检查维度匹配并设置输出形状
    for (int i = 0; i < *out_ndims; i++) {
        if (i < max_batch_dims) {
            // 处理批次维度
            int a_dim = i < batch_dims_a ? A_shape[batch_dims_a - 1 - i] : 1;
            int b_dim = i < batch_dims_b ? B_shape[batch_dims_b - 1 - i] : 1;
            if (a_dim != b_dim && a_dim != 1 && b_dim != 1) {
                return false;
            }
            out_shape[i] = a_dim > b_dim ? a_dim : b_dim;
        } else if (i == *out_ndims - 2) {
            // 设置M维度
            out_shape[i] = A_shape[A_dim - 2];
        } else {
            // 设置N维度
            out_shape[i] = B_shape[B_dim - 1];
        }
    }
    
    return true;
}