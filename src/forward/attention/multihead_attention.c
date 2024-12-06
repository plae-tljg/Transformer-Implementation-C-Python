#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "multihead_attention.h"

MultiHeadAttention* multihead_attention_create(int num_heads, int model_dim) {
    MultiHeadAttention* mha = (MultiHeadAttention*)malloc(sizeof(MultiHeadAttention));
    if (!mha) return NULL;

    mha->num_heads = num_heads;
    mha->model_dim = model_dim;
    mha->head_dim = model_dim / num_heads;

    // 初始化QKV投影权重和偏置
    // 注意：权重维度现在是 [model_dim, head_dim]
    int qkv_weight_shape[] = {model_dim, mha->head_dim};
    int qkv_bias_shape[] = {mha->head_dim};
    
    mha->W_q = tensor_create(qkv_weight_shape, 2);
    mha->W_k = tensor_create(qkv_weight_shape, 2);
    mha->W_v = tensor_create(qkv_weight_shape, 2);
    mha->b_q = tensor_create(qkv_bias_shape, 1);
    mha->b_k = tensor_create(qkv_bias_shape, 1);
    mha->b_v = tensor_create(qkv_bias_shape, 1);
    
    // 初始化输出投影
    mha->W_o = tensor_create(qkv_weight_shape, 2);
    mha->b_o = tensor_create(qkv_bias_shape, 1);


    return mha;
}

void multihead_attention_forward(
    MultiHeadAttention* mha,
    Tensor* input,        // [batch_size, seq_len, model_dim]
    Tensor* output,       // [batch_size, seq_len, model_dim]
    AttentionMask* mask         // [batch_size, num_heads, seq_len, seq_len]
) {
    int batch_size = input->shape[0];
    int seq_len = input->shape[1];
    int model_dim = input->shape[2];
    
    // 执行多头注意力计算
    bool success = project_qkv(
        input,          // 输入
        mha->W_q,      // Q权重
        mha->W_k,      // K权重
        mha->W_v,      // V权重
        mha->W_o,      // 输出权重
        mha->b_q,      // Q偏置
        mha->b_k,      // K偏置
        mha->b_v,      // V偏置
        mha->b_o,      // 输出偏置
        (AttentionMask*)mask,  // 注意力掩码
        mha->num_heads,// 注意力头数
        output         // 输出
    );

    if (!success) {
        fprintf(stderr, "多头注意力计算失败\n");
        return;
    }
}

void multihead_attention_free(MultiHeadAttention* mha) {
    if (!mha) return;
    
    tensor_free(mha->W_q);
    tensor_free(mha->W_k);
    tensor_free(mha->W_v);
    tensor_free(mha->b_q);
    tensor_free(mha->b_k);
    tensor_free(mha->b_v);
    tensor_free(mha->W_o);
    tensor_free(mha->b_o);
    
    free(mha);
}