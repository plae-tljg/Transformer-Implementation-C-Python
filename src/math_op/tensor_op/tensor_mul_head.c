#include "tensor_mul_head.h"
#include "attention_mask.h"
#include "softmax.h"
#include "model_config.h"

// 将3D输入与2D权重相乘并重塑为4D输出, on Q,K,V running separately
// input: [batch_size, seq_len, model_dim]
// weight: [model_dim, model_dim],
// output: [batch_size, seq_len, model_dim]
bool tensor_mul_3_2(
    const Tensor* input,
    const Tensor* weight,
    Tensor* output
){
    int batch_size = input->shape[0];
    int seq_len = input->shape[1];  // length of the sequence
    int model_dim = input->shape[2]; // model dimension
    

    // 进行批量矩阵乘法: [batch_size, seq_len, model_dim] @ [model_dim, model_dim] for last 2 dimensions
    // optimize later for gpu and blas
    for (int b = 0; b < batch_size; b++) {
        int input_offset = b * seq_len * model_dim;
        // 执行矩阵乘法 [seq_len, model_dim] x [model_dim, model_dim]
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < model_dim; j++) {
                float sum = 0.0f;
                for (int k = 0; k < model_dim; k++) {
                    // input[b,i,k] * weight[k,j] to output[b,i,j]
                    sum += input->data[(b * seq_len * model_dim) + (i * model_dim) + k] * 
                           weight->data[k * model_dim + j];
                }
                output->data[(b * seq_len * model_dim) + (i * model_dim) + j] = sum;
            }
        }
    }
    return true;
}

// 将偏置添加到3D张量
// input: [batch_size, seq_len, model_dim]
// bias: [model_dim]
// output: [batch_size, seq_len, model_dim]
bool tensor_add_bias_3d(
    const Tensor* input,
    const Tensor* bias,
    Tensor* output
) {

    int batch_size = input->shape[0];
    int seq_len = input->shape[1];
    int model_dim = input->shape[2];

    // 将输入复制到输出
    memcpy(output->data, input->data, batch_size * seq_len * model_dim * sizeof(float));

    // 为每个位置添加偏置
    // optimize later for gpu and blas
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_len; s++) {
            for (int d = 0; d < model_dim; d++) {
                // output[b,s,d] += bias[d]
                int idx = (b * seq_len * model_dim) + (s * model_dim) + d;
                output->data[idx] += bias->data[d];
            }
        }
    }

    return true;
}

// 将3D张量重塑为4D张量
// input: [batch_size, seq_len, model_dim]
// output: [batch_size, num_heads, seq_len, head_dim]
bool tensor_reshape_3d_to_4d(
    const Tensor* input,
    int num_heads,
    Tensor* output
) {
    int batch_size = input->shape[0];
    int seq_len = input->shape[1];
    int model_dim = input->shape[2];
    int head_dim = model_dim / num_heads;

    // 检查维度是否匹配
    if (model_dim % num_heads != 0) {
        return false;
    }

    // 检查输出张量形状是否正确
    if (output->shape[0] != batch_size ||
        output->shape[1] != num_heads ||
        output->shape[2] != seq_len ||
        output->shape[3] != head_dim) {
        return false;
    }

    // 重新排列数据
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_len; s++) {
            for (int h = 0; h < num_heads; h++) {
                for (int d = 0; d < head_dim; d++) {
                    // 源索引: [b, s, h * head_dim + d]
                    int src_idx = (b * seq_len * model_dim) + 
                                (s * model_dim) + 
                                (h * head_dim + d);
                    
                    // 目标索引: [b, h, s, d]
                    int dst_idx = (b * num_heads * seq_len * head_dim) +
                                (h * seq_len * head_dim) +
                                (s * head_dim) + d;
                    
                    output->data[dst_idx] = input->data[src_idx];
                }
            }
        }
    }

    return true;
}



// 将4D张量重塑为3D张量, reshape with transpose
// input: [batch_size, num_heads, seq_len, head_dim] 
// output: [batch_size, seq_len, model_dim]
bool tensor_reshape_4d_to_3d(
    const Tensor* input,
    Tensor* output
) {
    // 这个函数用于将注意力计算后的4D张量重塑回3D张量
    // 输入: [batch_size, num_heads, seq_len, head_dim]
    // 输出: [batch_size, seq_len, model_dim]
    int batch_size = input->shape[0];
    int num_heads = input->shape[1];
    int seq_len = input->shape[2];
    int head_dim = input->shape[3];
    int model_dim = num_heads * head_dim;

    // 重新排列数据
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_len; s++) {
            for (int h = 0; h < num_heads; h++) {
                for (int d = 0; d < head_dim; d++) {
                    // 源索引: [b, h, s, d]
                    int src_idx = (b * num_heads * seq_len * head_dim) +
                                (h * seq_len * head_dim) +
                                (s * head_dim) + d;
                    
                    // 目标索引: [b, s, h * head_dim + d]
                    int dst_idx = (b * seq_len * model_dim) +
                                (s * model_dim) +
                                (h * head_dim + d);
                    
                    output->data[dst_idx] = input->data[src_idx];
                }
            }
        }
    }

    return true;
}

// 4D张量乘法,K的最后两个维度要转置
// input1: [batch_size, num_heads, seq_len, head_dim]
// input2: [batch_size, num_heads, seq_len, head_dim], transposed last two dimensions and then multiply
// output: [batch_size, num_heads, seq_len, seq_len]
bool tensor_mul_4d_transpose(
    const Tensor* input1,
    const Tensor* input2, 
    float scale,
    Tensor* output
) {
    int batch_size = input1->shape[0];
    int num_heads = input1->shape[1];
    int seq_len = input1->shape[2];
    int head_dim = input1->shape[3];

    // 对每个batch和head计算注意力分数
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < num_heads; h++) {
            for (int i = 0; i < seq_len; i++) {
                for (int j = 0; j < seq_len; j++) {
                    float score = 0.0f;
                    for (int k = 0; k < head_dim; k++) {
                        // input1[b,h,i,k] * input2[b,h,j,k]
                        int idx1 = ((b * num_heads + h) * seq_len + i) * head_dim + k;
                        int idx2 = ((b * num_heads + h) * seq_len + j) * head_dim + k;
                        score += input1->data[idx1] * input2->data[idx2];
                    }
                    // 将结果存储在输出中
                    int out_idx = ((b * num_heads + h) * seq_len + i) * seq_len + j;
                    output->data[out_idx] = score * scale;
                }
            }
        }
    }
    return true;
}


// 辅助函数：将3D输入与2D权重相乘并重塑为4D输出
// input: [batch_size, seq_len, model_dim]
// weight: [model_dim, model_dim]
// bias: [model_dim]
// output: [batch_size, num_heads, seq_len, head_dim]
bool project_qkv(
    const Tensor* input,      // [batch_size, seq_len, model_dim]
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
) {
    int batch_size = input->shape[0];
    int seq_len = input->shape[1];
    int model_dim = input->shape[2];
    int num_heads = g_model_config.num_heads;
    int head_dim = model_dim / num_heads;

    // 1. 首先进行批量矩阵乘法: [batch_size, seq_len, model_dim] @ [model_dim, model_dim]
    int temp_shape[] = {batch_size, seq_len, model_dim};
    Tensor* temp_q = tensor_create(temp_shape, 3);
    Tensor* temp_k = tensor_create(temp_shape, 3);
    Tensor* temp_v = tensor_create(temp_shape, 3);

    // 执行3D矩阵乘法
    if (!tensor_mul_3_2(input, weight_q, temp_q) ||
        !tensor_mul_3_2(input, weight_k, temp_k) ||
        !tensor_mul_3_2(input, weight_v, temp_v)) {
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
    
    Tensor* reshaped_q = tensor_reshape(temp_q, new_shape, 4);
    Tensor* reshaped_k = tensor_reshape(temp_k, new_shape, 4);  
    Tensor* reshaped_v = tensor_reshape(temp_v, new_shape, 4);
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
    if (!tensor_mul_4d(output, reshaped_v, output)) {
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


