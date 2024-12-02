#ifndef ATTENTION_H
#define ATTENTION_H

#include <stdbool.h>
typedef struct SelfAttention SelfAttention;
typedef struct MultiHeadAttention MultiHeadAttention;
typedef struct AttentionMask AttentionMask;

// 注意力掩码结构
struct AttentionMask {
    int seq_length;
    float* mask;  // [seq_length, seq_length]
};

// 自注意力结构
struct SelfAttention {
    int head_dim;   // 注意力头维度, d_model / num_heads
    int num_heads;
    
    // 权重
    float* query_weights;
    float* key_weights;
    float* value_weights;
    float* output_weights;
    
    // 偏置
    float* query_bias;
    float* key_bias;
    float* value_bias;
    float* output_bias;
    
    // // 梯度
    // float* query_gradients;
    // float* key_gradients;
    // float* value_gradients;
    // float* output_gradients;
    
    // // 偏置梯度
    // float* query_bias_gradients;
    // float* key_bias_gradients;
    // float* value_bias_gradients;
    // float* output_bias_gradients;
    
    bool requires_grad;
};

// 多头注意力结构
struct MultiHeadAttention {
    int num_heads;
    int model_dim;
    int head_dim;
    
    // 注意力头
    SelfAttention** attention_heads;
    
    // 输出层
    float* output_weights;
    float* output_bias;
    
    // // 梯度
    // float* output_weights_gradients;
    // float* output_bias_gradients;
    // float* attention_weights_gradients;
    
    // 缓存用于反向传播
    float* input_cache;
    float* attention_scores_cache;
    float* attention_output_cache;
    
    bool requires_grad;
};

// 添加掩码相关函数声明
AttentionMask* attention_mask_create(int seq_length);
void attention_mask_free(AttentionMask* mask);

// 添加以下函数声明
void multihead_attention_free(MultiHeadAttention* mha);
SelfAttention* self_attention_create(int num_heads, int head_dim, bool requires_grad);
void self_attention_free(SelfAttention* self_attn);
void self_attention_forward(SelfAttention* self_attn, float* input, int seq_length, float* output, AttentionMask* mask);

// 添加多头注意力相关函数声明
MultiHeadAttention* multihead_attention_create(int num_heads, int model_dim, bool requires_grad);
void multihead_attention_free(MultiHeadAttention* mha);
void multihead_attention_forward(MultiHeadAttention* mha, float* input, int seq_length, float* output, AttentionMask* mask);

#endif