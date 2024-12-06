#ifndef TENSOR_MUL_HEAD_H
#define TENSOR_MUL_HEAD_H

#include "tensor.h"
#include "attention_mask.h"

// 3D张量与2D权重相乘
// input: [batch_size, seq_len, model_dim]
// weight: [model_dim, model_dim]
// output: [batch_size, seq_len, model_dim]
bool tensor_mul_3_2(
    const Tensor* input,
    const Tensor* weight,
    Tensor* output
);

// 将3D张量重塑为4D张量
// input: [batch_size, seq_len, model_dim]
// output: [batch_size, num_heads, seq_len, head_dim]
bool tensor_reshape_3d_to_4d(
    const Tensor* input,
    int num_heads,
    Tensor* output
);

// 将4D张量重塑为3D张量
// input: [batch_size, num_heads, seq_len, head_dim]
// output: [batch_size, seq_len, model_dim]
bool tensor_reshape_4d_to_3d(
    const Tensor* input,
    Tensor* output
);

// 4D张量乘法,K的最后两个维度要转置
// input1: [batch_size, num_heads, seq_len, head_dim]
// input2: [batch_size, num_heads, seq_len, head_dim]
// output: [batch_size, num_heads, seq_len, seq_len]
bool tensor_mul_4d_transpose(
    const Tensor* input1,
    const Tensor* input2,
    float scale,
    Tensor* output
);

// 完整的多头注意力投影计算
// input: [batch_size, seq_len, model_dim]
// weight_q/k/v: [model_dim, head_dim]
// bias_q/k/v: [head_dim]
// weight_o: [model_dim, model_dim]
// bias_o: [model_dim]
// output: [batch_size, seq_len, model_dim]
bool project_qkv(
    const Tensor* input,      
    const Tensor* weight_q,   
    const Tensor* weight_k,   
    const Tensor* weight_v,   
    const Tensor* weight_o,   
    const Tensor* bias_q,     
    const Tensor* bias_k,     
    const Tensor* bias_v,     
    const Tensor* bias_o,     
    const AttentionMask* mask,
    int num_heads,
    Tensor* output           
);

#endif
