#include "attention.h"
#include <stdlib.h>
#include <float.h>

// 创建注意力掩码
float* create_attention_mask(int seq_length) {
    float* mask = malloc(seq_length * seq_length * sizeof(float));
    if (!mask) return NULL;
    
    // 创建上三角掩码（用于解码器的自注意力）
    for (int i = 0; i < seq_length; i++) {
        for (int j = 0; j < seq_length; j++) {
            mask[i * seq_length + j] = (j <= i) ? 1.0f : 0.0f;
        }
    }
    
    return mask;
}

// 应用注意力掩码
void apply_attention_mask(float* attention_scores, float* mask, int seq_length) {
    if (!attention_scores || !mask || seq_length <= 0) return;
    
    // 将掩码为0的位置设置为负无穷（在softmax后会变为0）
    for (int i = 0; i < seq_length * seq_length; i++) {
        if (mask[i] == 0.0f) {
            attention_scores[i] = -FLT_MAX;
        }
    }
}