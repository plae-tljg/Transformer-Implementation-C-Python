#ifndef ATTENTION_H
#define ATTENTION_H

#include "types.h"

// 注意力掩码函数
float* create_attention_mask(int seq_length);
void apply_attention_mask(float* attention_scores, float* mask, int seq_length);

// 自注意力相关函数
SelfAttention* create_self_attention(int head_dim);
void free_self_attention(SelfAttention* self_attn);
float* compute_self_attention_head(SelfAttention* self_attn, float* input, int seq_length);

// 多头注意力函数
float* compute_multi_head_attention(MultiHeadAttention* multi_attn,
                                  float* query,
                                  float* key,
                                  float* value,
                                  int seq_length);

#endif // ATTENTION_H 