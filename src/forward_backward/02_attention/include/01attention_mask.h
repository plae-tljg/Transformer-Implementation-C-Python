#ifndef ATTENTION_MASK_H
#define ATTENTION_MASK_H

#include "tensor_type.h"

typedef struct AttentionMask AttentionMask;

// 注意力掩码结构
struct AttentionMask {
    int seq_length;
    Tensor* mask;  // [seq_length, seq_length] 二维张量，用于存储注意力掩码
                   // 1.0表示允许注意力，0.0表示屏蔽注意力
};

// 创建注意力掩码,用于处理padding token
AttentionMask* pad_mask_create(Tensor* q, Tensor* k, int num_heads, int pad_token_id);

// 创建因果掩码,用于decoder的自注意力
AttentionMask* create_causal_mask(int* shape, int num_dims);

// 创建目标掩码,用于decoder的自注意力,考虑padding
AttentionMask* create_trg_mask(int* shape, int num_dims);

// 释放注意力掩码
void attention_mask_free(AttentionMask* mask);

// 应用注意力掩码到注意力分数上
bool apply_attention_mask(
    const Tensor* scores,  // 输入的注意力分数
    const AttentionMask* mask,  // 注意力掩码
    Tensor* output  // 输出的掩码后的分数
);

#endif
