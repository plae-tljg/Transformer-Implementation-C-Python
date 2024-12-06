#ifndef ATTENTION_MASK_H
#define ATTENTION_MASK_H

#include "tensor.h"

typedef struct AttentionMask AttentionMask;

// 注意力掩码结构
struct AttentionMask {
    int seq_length;
    Tensor* mask;  // [seq_length, seq_length] 二维张量，用于存储注意力掩码
                   // 1.0表示允许注意力，0.0表示屏蔽注意力
};

AttentionMask* attention_mask_create(int seq_length);
void attention_mask_free(AttentionMask* mask);
bool apply_attention_mask(
    const Tensor* scores,
    const AttentionMask* mask,
    Tensor* output
);
#endif
