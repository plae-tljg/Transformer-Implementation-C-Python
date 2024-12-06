#ifndef SELF_ATTENTION_H
#define SELF_ATTENTION_H

#include "attention_mask.h"
#include "tensor.h"

// not in use, since would use 4D tensor in transformer by multiple heads

typedef struct SelfAttention SelfAttention;

// 自注意力结构 (单头实现)
struct SelfAttention {
    int head_dim;   // 注意力头维度, encoding dimension per head, d_model / num_heads
    
    // 权重 (使用Tensor替代原始指针)
    Tensor* query_weights;  // [head_dim, head_dim], Q
    Tensor* key_weights;    // [head_dim, head_dim], K
    Tensor* value_weights;  // [head_dim, head_dim], V
    
    // 偏置
    Tensor* query_bias;    // [head_dim], Q
    Tensor* key_bias;      // [head_dim], K
    Tensor* value_bias;    // [head_dim], V

    // no requires_grad in this version, since need to be trained
};

// 函数声明
SelfAttention* self_attention_create(int head_dim);
void self_attention_free(SelfAttention* self_attn);

// 前向传播函数 - 处理单个注意力头
void self_attention_forward(
    SelfAttention* self_attn,
    Tensor* input,          // [batch_size, seq_len, head_dim], from transformer embedding layer
    Tensor* output,         // [batch_size, seq_len, head_dim]
    AttentionMask* mask    // [batch_size, seq_len, seq_len]
);

#endif
