#include "tensor_trio.h"
#include <stdio.h>

// 创建padding掩码 - 用于处理序列中的padding token
// mask: [batch_size, num_heads, q_seq_len, k_seq_len]
// k: [batch_size, k_seq_len, model_dim] - 只需要考虑k中的padding
bool tensor_create_pad_mask(Tensor* mask, int batch_size, int q_seq_len, Tensor* k, int pad_token_id) {
    if (!mask || !k) {
        fprintf(stderr, "Invalid tensors for pad mask\n");
        return false;
    }
    
    const int num_heads = mask->shape[1];
    const int k_seq_len = k->shape[1];
    
    // 检查输入tensor的形状是否正确
    if (mask->shape[0] != batch_size || 
        mask->shape[2] != q_seq_len ||
        mask->shape[3] != k_seq_len ||
        k->shape[0] != batch_size) {
        fprintf(stderr, "Invalid tensor shapes for pad mask\n");
        return false;
    }
    
    // 初始化掩码,只考虑k中的padding token
    #pragma omp parallel for collapse(4) if(batch_size * num_heads * q_seq_len * k_seq_len > 1000)
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < num_heads; h++) {
            for (int i = 0; i < q_seq_len; i++) {
                for (int j = 0; j < k_seq_len; j++) {
                    // 如果k中的token是padding token,则设置为0,否则设置为1
                    float mask_value = (k->data[b * k_seq_len + j] == pad_token_id) ? 0.0f : 1.0f;
                    mask->data[((b * num_heads + h) * q_seq_len + i) * k_seq_len + j] = mask_value;
                }
            }
        }
    }
    return true;
}
    
bool tensor_create_triangular_mask(
    Tensor* tensor,
    TriangularType type,
    float fill_value,
    float other_value
) {
    if (!tensor || tensor->num_dims != 2) {
        fprintf(stderr, "Invalid tensor for triangular mask\n");
        return false;
    }

    const int rows = tensor->shape[0];
    const int cols = tensor->shape[1];

    #pragma omp parallel for collapse(2) if(rows * cols > 1000)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            bool is_in_triangle = false;
            switch (type) {
                case LOWER_TRIANGULAR:
                    is_in_triangle = (j <= i);
                    break;
                case UPPER_TRIANGULAR:
                    is_in_triangle = (j >= i);
                    break;
                case STRICTLY_LOWER:
                    is_in_triangle = (j < i);
                    break;
                case STRICTLY_UPPER:
                    is_in_triangle = (j > i);
                    break;
            }
            tensor->data[i * cols + j] = is_in_triangle ? fill_value : other_value;
        }
    }

    return true;
}

bool tensor_create_causal_mask(Tensor* tensor) {
    // 因果掩码就是一个特殊的下三角掩码
    return tensor_create_triangular_mask(
        tensor,
        LOWER_TRIANGULAR,  // 下三角（包括对角线）
        1.0f,             // 三角区域填充1
        0.0f              // 其他区域填充0
    );
}
