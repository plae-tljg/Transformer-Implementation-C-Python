#include "lookup.h"
#include <stdio.h>
#include <string.h>

// 新增的嵌入查找函数，便于将来GPU加速
bool perform_embedding_lookup(
    const Tensor* embedding_matrix,
    const Tensor* tokens,
    Tensor* output,
    int batch_size, 
    int seq_length,
    int embedding_dim
) {
    // 这部分将来可以替换为GPU实现
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_length; s++) {
            int token = (int)tokens->data[b * seq_length + s];
            
            // 检查token是否超出范围
            if (token < 0 || token >= embedding_matrix->shape[0]) {
                fprintf(stderr, "Token id %d exceeds vocabulary size %d\n", 
                    token, embedding_matrix->shape[0]);
                return false;
            }
            
            // 计算源和目标的偏移量
            float* dst = output->data + (b * seq_length + s) * embedding_dim;
            float* src = embedding_matrix->data + token * embedding_dim;
            
            // 复制embedding向量
            memcpy(dst, src, embedding_dim * sizeof(float));
        }
    }
    return true;
}