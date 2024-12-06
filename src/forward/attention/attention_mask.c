#include <stdlib.h>
#include <stdio.h>
#include "attention_mask.h"

// 创建注意力掩码
AttentionMask* attention_mask_create(int seq_length) {
    AttentionMask* mask = (AttentionMask*)malloc(sizeof(AttentionMask));
    if (!mask) {
        fprintf(stderr, "Failed to allocate memory for attention mask\n");
        return NULL;
    }

    mask->seq_length = seq_length;
    
    // 创建二维掩码张量 [seq_length, seq_length]
    int mask_shape[] = {seq_length, seq_length};
    mask->mask = tensor_create(mask_shape, 2);
    if (!mask->mask) {
        fprintf(stderr, "Failed to allocate memory for mask tensor\n");
        free(mask);
        return NULL;
    }

    // 初始化为因果掩码(上三角为0,下三角为1)
    for (int i = 0; i < seq_length; i++) {
        for (int j = 0; j < seq_length; j++) {
            mask->mask->data[i * seq_length + j] = j <= i ? 1.0f : 0.0f;
        }
    }

    return mask;
}

// 释放注意力掩码
void attention_mask_free(AttentionMask* mask) {
    if (!mask) return;
    tensor_free(mask->mask);
    free(mask);
} 

bool apply_attention_mask(
    const Tensor* scores,
    const AttentionMask* mask,
    Tensor* output
) {
    int batch_size = scores->shape[0];
    int num_heads = scores->shape[1];
    int seq_len = scores->shape[2];

    // 复制scores到output
    memcpy(output->data, scores->data, 
           batch_size * num_heads * seq_len * seq_len * sizeof(float));

    // 应用掩码：将被掩码的位置设置为负无穷（实际中用一个很小的数）
    const float MASKING_VALUE = -1e9f;
    
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < num_heads; h++) {
            for (int i = 0; i < seq_len; i++) {
                for (int j = 0; j < seq_len; j++) {
                    int scores_idx = ((b * num_heads + h) * seq_len + i) * seq_len + j;
                    int mask_idx = i * seq_len + j;
                    
                    if (mask->mask->data[mask_idx] == 0.0f) {
                        output->data[scores_idx] = MASKING_VALUE;
                    }
                }
            }
        }
    }
    return true;
}