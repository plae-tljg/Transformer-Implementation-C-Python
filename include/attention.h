#ifndef ATTENTION_H
#define ATTENTION_H

#include "types.h"

// 多头注意力的主要函数
float* compute_multi_head_attention(MultiHeadAttention* multi_attn,
                                  float* query,
                                  float* key,
                                  float* value,
                                  float* attention_mask,  // 可以为 NULL
                                  int seq_length);

// 创建和释放函数
MultiHeadAttention* create_multi_head_attention(int model_dim, int num_heads);
void free_attention(MultiHeadAttention* mha);

#endif // ATTENTION_H 