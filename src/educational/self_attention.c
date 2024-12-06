#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "self_attention.h"

// not in use, since would use 4D tensor in transformer by multiple heads

// 创建自注意力层
SelfAttention* self_attention_create(int head_dim) {
    SelfAttention* self_attn = (SelfAttention*)malloc(sizeof(SelfAttention));
    if (!self_attn) {
        fprintf(stderr, "Failed to allocate memory for self attention\n");
        return NULL;
    }

    self_attn->head_dim = head_dim;

    // 创建权重张量 [head_dim, head_dim]
    int weight_shape[] = {head_dim, head_dim};
    self_attn->query_weights = tensor_create(weight_shape, 2);
    self_attn->key_weights = tensor_create(weight_shape, 2);
    self_attn->value_weights = tensor_create(weight_shape, 2);

    // 创建偏置张量 [head_dim]
    int bias_shape[] = {head_dim};
    self_attn->query_bias = tensor_create(bias_shape, 1);
    self_attn->key_bias = tensor_create(bias_shape, 1);
    self_attn->value_bias = tensor_create(bias_shape, 1);

    // 检查内存分配
    if (!self_attn->query_weights || !self_attn->key_weights || 
        !self_attn->value_weights || !self_attn->query_bias || 
        !self_attn->key_bias || !self_attn->value_bias) {
        fprintf(stderr, "Failed to allocate memory for attention tensors\n");
        self_attention_free(self_attn);
        return NULL;
    }

    return self_attn;
}

// 释放自注意力层
void self_attention_free(SelfAttention* self_attn) {
    if (!self_attn) return;

    tensor_free(self_attn->query_weights);
    tensor_free(self_attn->key_weights);
    tensor_free(self_attn->value_weights);
    
    tensor_free(self_attn->query_bias);
    tensor_free(self_attn->key_bias);
    tensor_free(self_attn->value_bias);

    free(self_attn);
}

// 自注意力前向传播
void self_attention_forward(
    SelfAttention* self_attn,
    Tensor* input,          // [seq_len, head_dim]
    Tensor* output,         // [seq_len, head_dim]
    AttentionMask* mask    // [seq_len, seq_len]
) {
    int seq_length = input->shape[0];
    int head_dim = self_attn->head_dim;
    
    // 创建Q、K、V张量 [seq_len, head_dim]
    int qkv_shape[] = {seq_length, head_dim};
    Tensor* query = tensor_create(qkv_shape, 2);
    Tensor* key = tensor_create(qkv_shape, 2);
    Tensor* value = tensor_create(qkv_shape, 2);
    
    // 计算Q、K、V (包括线性变换和偏置)
    tensor_linear_transform(input, self_attn->query_weights, self_attn->query_bias, query);
    tensor_linear_transform(input, self_attn->key_weights, self_attn->key_bias, key);
    tensor_linear_transform(input, self_attn->value_weights, self_attn->value_bias, value);
    
    // 计算注意力分数 [seq_len, seq_len]
    int scores_shape[] = {seq_length, seq_length};
    Tensor* attention_scores = tensor_create(scores_shape, 2);
    
    // QK^T
    tensor_matmul(query, key, attention_scores, true);  // transpose_b=true
    
    // 缩放
    float scale = sqrtf(head_dim);
    tensor_scale(attention_scores, 1.0f / scale);
    
    // 应用掩码（如果提供）
    if (mask) {
        tensor_apply_mask(attention_scores, mask->mask);
    }
    
    // Softmax
    tensor_softmax(attention_scores);
    
    // 计算输出 [seq_len, head_dim]
    tensor_matmul(attention_scores, value, output, false);
    
    // 清理临时张量
    tensor_free(query);
    tensor_free(key);
    tensor_free(value);
    tensor_free(attention_scores);
}
