#ifndef MULTIATTENTION_H
#define MULTIATTENTION_H

#include "tensor_type.h"
#include "01attention_mask.h"

typedef struct MultiHeadAttention MultiHeadAttention;

struct MultiHeadAttention {
    int num_heads;
    int model_dim;  // 模型维度, d_model, not splited by num_heads
    int head_dim;   // 注意力头维度, d_model / num_heads
    
    // QKV投影权重和偏置
    Tensor* W_q;    // [model_dim, model_dim]
    Tensor* W_k;    // [model_dim, model_dim]
    Tensor* W_v;    // [model_dim, model_dim]
    Tensor* b_q;    // [model_dim]
    Tensor* b_k;    // [model_dim]
    Tensor* b_v;    // [model_dim]
    
    // 输出投影
    Tensor* W_o;    // [model_dim, model_dim], 用于将多头注意力结果合并成一个向量
    Tensor* b_o;    // [model_dim], 用于将多头注意力结果合并成一个向量
};

MultiHeadAttention* multihead_attention_create(int num_heads, int model_dim);
void multihead_attention_free(MultiHeadAttention* mha);
bool multihead_attention_forward(
    MultiHeadAttention* mha,
    Tensor* input,        // [batch_size, seq_len, model_dim]
    Tensor* output,       // [batch_size, seq_len, model_dim]
    AttentionMask* mask // [batch_size, num_heads, seq_len, seq_len] 或 NULL
);

bool cross_attention_forward(
    MultiHeadAttention* mha,
    Tensor* input_q,        // [batch_size, seq_len, model_dim]
    Tensor* input_k,        // [batch_size, seq_len, model_dim]
    Tensor* input_v,        // [batch_size, seq_len, model_dim]
    Tensor* output,       // [batch_size, seq_len, model_dim]
    AttentionMask* mask         // [batch_size, num_heads, seq_len, seq_len]
);

bool project_qkv(
    const Tensor* input_q,      // [batch_size, seq_len, model_dim]
    const Tensor* input_k,      // [batch_size, seq_len, model_dim]
    const Tensor* input_v,      // [batch_size, seq_len, model_dim]
    const Tensor* weight_q,   // [model_dim, model_dim]  
    const Tensor* weight_k,   // [model_dim, model_dim]
    const Tensor* weight_v,   // [model_dim, model_dim]
    const Tensor* weight_combine, // [model_dim, model_dim]
    const Tensor* bias_q,     // [model_dim]
    const Tensor* bias_k,     // [model_dim] 
    const Tensor* bias_v,     // [model_dim]
    const Tensor* bias_combine, // [model_dim]
    const AttentionMask* mask,
    Tensor* output           // [batch_size, num_heads, seq_len, head_dim]
);

#endif // MULTIATTENTION_H