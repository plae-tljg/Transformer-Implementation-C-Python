#include <stdlib.h>
#include <stdio.h>
#include "attention.h"

// 创建注意力掩码
AttentionMask* attention_mask_create(int seq_length) {
    AttentionMask* mask = (AttentionMask*)malloc(sizeof(AttentionMask));
    if (!mask) {
        fprintf(stderr, "Failed to allocate memory for attention mask\n");
        return NULL;
    }

    mask->seq_length = seq_length;
    mask->mask = (float*)malloc(seq_length * seq_length * sizeof(float));
    if (!mask->mask) {
        fprintf(stderr, "Failed to allocate memory for mask matrix\n");
        free(mask);
        return NULL;
    }

    // 初始化为因果掩码(上三角为0,下三角为1)
    for (int i = 0; i < seq_length; i++) {
        for (int j = 0; j < seq_length; j++) {
            mask->mask[i * seq_length + j] = j <= i ? 1.0f : 0.0f;
        }
    }

    return mask;
}

// 释放注意力掩码
void attention_mask_free(AttentionMask* mask) {
    if (!mask) return;
    free(mask->mask);
    free(mask);
} 