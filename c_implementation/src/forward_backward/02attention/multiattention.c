#include "02multiattention.h"
#include "model_config.h"
#include "tensor_mul.h"
#include "tensor_add.h"
#include "tensor_reshape.h"
#include "softmax.h"
#include <stdio.h>

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

bool multihead_attention_forward(
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
        input,          // 输入
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
        output         // 输出
    );

    if (!success) {
        fprintf(stderr, "多头注意力计算失败\n");
        return;
    }
}

bool cross_attention_forward(
    MultiHeadAttention* mha,
    Tensor* input_q,        // [batch_size, seq_len, model_dim]
    Tensor* input_k,        // [batch_size, seq_len, model_dim]
    Tensor* input_v,        // [batch_size, seq_len, model_dim]
    Tensor* output,       // [batch_size, seq_len, model_dim]
    AttentionMask* mask         // [batch_size, num_heads, seq_len, seq_len]
) {
    int batch_size = input_q->shape[0];
    int seq_len = input_q->shape[1];
    int model_dim = input_q->shape[2];
    
    // 执行多头注意力计算
    bool success = project_qkv(
        input_q,          // 输入
        input_k,          // 输入
        input_v,          // 输入
        mha->W_q,      // Q权重
        mha->W_k,      // K权重
        mha->W_v,      // V权重
        mha->W_o,      // 输出权重
        mha->b_q,      // Q偏置
        mha->b_k,      // K偏置
        mha->b_v,      // V偏置
        mha->b_o,      // 输出偏置
        (AttentionMask*)mask,  // 注意力掩码
        output         // 输出
    );

    if (!success) {
        fprintf(stderr, "多头注意力计算失败\n");
        return;
    }
}


// 辅助函数：将3D输入与2D权重相乘并重塑为4D输出
// input: [batch_size, seq_len, model_dim]
// weight: [model_dim, model_dim]
// bias: [model_dim]
// output: [batch_size, num_heads, seq_len, head_dim]
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
    const AttentionMask* mask,    // [batch_size, num_heads, seq_len, seq_len]
    Tensor* output           // [batch_size, num_heads, seq_len, head_dim]
) {
    int batch_size = input_q->shape[0];
    int seq_len = input_q->shape[1];
    int model_dim = input_q->shape[2];
    int num_heads = g_model_config.num_heads;
    int head_dim = model_dim / num_heads;

    // 1. 首先进行批量矩阵乘法: [batch_size, seq_len, model_dim] @ [model_dim, model_dim]
    int temp_shape[] = {batch_size, seq_len, model_dim};
    Tensor* temp_q = tensor_create(temp_shape, 3);
    Tensor* temp_k = tensor_create(temp_shape, 3);
    Tensor* temp_v = tensor_create(temp_shape, 3);

    // 执行3D矩阵乘法
    if (!tensor_mul_3_2(input_q, weight_q, temp_q) ||
        !tensor_mul_3_2(input_k, weight_k, temp_k) ||
        !tensor_mul_3_2(input_v, weight_v, temp_v)) {
        tensor_free(temp_q);
        tensor_free(temp_k);
        tensor_free(temp_v);
        return false;
    }

    // 添加偏置
    if (!tensor_add_bias_3d(temp_q, bias_q, temp_q) ||
        !tensor_add_bias_3d(temp_k, bias_k, temp_k) ||
        !tensor_add_bias_3d(temp_v, bias_v, temp_v)) {
        tensor_free(temp_q);
        tensor_free(temp_k);
        tensor_free(temp_v);
        return false;
    }
    
    // 2. 重塑结果为4D: [batch_size, num_heads, seq_len, head_dim]
    int new_shape[] = {batch_size, num_heads, seq_len, head_dim};
    
    Tensor* reshaped_q = tensor_create(new_shape, 4);
    Tensor* reshaped_k = tensor_create(new_shape, 4);
    Tensor* reshaped_v = tensor_create(new_shape, 4);
    
    if (!tensor_reshape_3d_to_4d(temp_q, num_heads, reshaped_q) ||
        !tensor_reshape_3d_to_4d(temp_k, num_heads, reshaped_k) ||
        !tensor_reshape_3d_to_4d(temp_v, num_heads, reshaped_v)) {
        return false;
    }
    // 计算注意力分数: Q * K^T / sqrt(head_dim)
    float scale = 1.0f / sqrt(head_dim);
    
    // 调用4D张量乘法函数,K的最后两个维度需要转置
    if (!tensor_mul_4d_transpose(reshaped_q, reshaped_k, scale, output)) {
        return false;
    }

    // 应用注意力掩码
    if (mask) {
        if (!apply_attention_mask(output, mask, output)) {
            return false;
        }
    }

    // 应用softmax
    if (!attention_scores_softmax(output, output)) {
        return false;
    }
    // 将softmax后的注意力分数与V相乘
    // output: [batch_size, num_heads, seq_len, seq_len]
    // reshaped_v: [batch_size, num_heads, seq_len, head_dim]
    // 结果仍存在output中: [batch_size, num_heads, seq_len, head_dim]
    if (!tensor_matmul_4d(output, reshaped_v, output)) {
        return false;
    }

    // 4. 将4D张量重塑为3D张量
    if (!tensor_reshape_4d_to_3d(output, output)) {
        return false;
    }

    // 计算最终的多头注意力输出: [batch_size, seq_len, model_dim]
    if (!tensor_mul_3_2(output, weight_combine, output)) {
        return false;
    }

    // 添加输出偏置
    if (!tensor_add_bias_3d(output, bias_combine, output)) {
        return false;
    }

    // 5. 释放临时张量
    tensor_free(reshaped_q);
    tensor_free(reshaped_k); 
    tensor_free(reshaped_v);
    
    tensor_free(temp_q);
    tensor_free(temp_k);
    tensor_free(temp_v);
    return true;
}