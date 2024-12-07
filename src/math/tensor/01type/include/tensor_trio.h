#ifndef TENSOR_TRIO_H
#define TENSOR_TRIO_H

#include "tensor_type.h"
#include <stdbool.h>

bool tensor_create_pad_mask(Tensor* mask, int batch_size, int q_seq_len, Tensor* k, int pad_token_id);

// 三角形掩码类型
typedef enum {
    LOWER_TRIANGULAR = 0,  // 下三角（包括对角线）
    UPPER_TRIANGULAR = 1,  // 上三角（包括对角线）
    STRICTLY_LOWER = 2,    // 严格下三角（不包括对角线）
    STRICTLY_UPPER = 3     // 严格上三角（不包括对角线）
} TriangularType;

// 创建三角形掩码
bool tensor_create_triangular_mask(
    Tensor* tensor,           // 目标张量
    TriangularType type,      // 三角形类型
    float fill_value,         // 三角区域的填充值
    float other_value        // 其他区域的填充值
);

// 创建因果掩码（下三角为1，上三角为0）
bool tensor_create_causal_mask(
    Tensor* tensor           // 目标张量 [seq_length, seq_length]
);

#endif // TENSOR_TRIO_H