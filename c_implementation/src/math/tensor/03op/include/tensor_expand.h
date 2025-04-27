#ifndef TENSOR_EXPAND_H
#define TENSOR_EXPAND_H

#include "tensor_type.h"
#include <stdbool.h>

// 广播2D张量到4D
bool tensor_broadcast_2d_to_4d(
    const Tensor* input,      // [M, N]
    int batch_size,          // 目标batch维度
    int num_heads,           // 目标head维度
    Tensor* output           // [batch_size, num_heads, M, N]
);

// 广播操作的通用接口
bool tensor_broadcast(
    const Tensor* input,     // 输入张量
    const int* target_shape, // 目标形状
    int target_dims,         // 目标维度数
    Tensor* output          // 输出张量
);

#endif // TENSOR_EXPAND_H
