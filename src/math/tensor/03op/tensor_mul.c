#include "tensor_mul.h"

#include "tensor_mul.h"
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

// 2D矩阵乘法: [M, K] × [K, N] -> [M, N]
bool tensor_matmul_2d(const Tensor* left, const Tensor* right, Tensor* output) {
    // 检查维度数量
    if (left->num_dims != 2 || right->num_dims != 2) {
        fprintf(stderr, "张量必须是2维的\n");
        return false;
    }

    // 获取维度
    const int* left_shape = left->shape;
    const int* right_shape = right->shape;
    const int* out_shape = output->shape;
    const float* left_data = left->data;
    const float* right_data = right->data;
    float* out_data = output->data;

    int rows = left_shape[0];
    int inner_dim = left_shape[1];
    int cols = right_shape[1];

    // 检查维度匹配
    if (right_shape[0] != inner_dim) {
        fprintf(stderr, "矩阵维度不匹配\n");
        return false;
    }

    // 检查输出张量维度
    if (output->num_dims != 2 || 
        out_shape[0] != rows || 
        out_shape[1] != cols) {
        fprintf(stderr, "输出张量维度无效\n");
        return false;
    }

    // 执行矩阵乘法
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            float sum = 0.0f;
            for (int k = 0; k < inner_dim; k++) {
                sum += left_data[row * inner_dim + k] * right_data[k * cols + col];
            }
            out_data[row * cols + col] = sum;
        }
    }

    return true;
}

// 3D张量乘法: [batch_size, M, K] × [batch_size, K, N] -> [batch_size, M, N]
bool tensor_matmul_3d(const Tensor* left, const Tensor* right, Tensor* output) {
    // 检查维度数量
    if (left->num_dims != 3 || right->num_dims != 3) {
        fprintf(stderr, "Tensors must be 3-dimensional\n");
        return false;
    }

    // 获取维度
    const int* left_shape = left->shape;
    const int* right_shape = right->shape;
    const int* out_shape = output->shape;
    const float* left_data = left->data;
    const float* right_data = right->data;
    float* out_data = output->data;

    int batch_size = left_shape[0];
    int rows = left_shape[1];
    int inner_dim = left_shape[2];
    int cols = right_shape[2];

    // 检查维度匹配
    if (right_shape[0] != batch_size || right_shape[1] != inner_dim) {
        fprintf(stderr, "Incompatible dimensions for 3D tensor multiplication\n");
        return false;
    }

    // 检查输出张量维度
    if (output->num_dims != 3 || 
        out_shape[0] != batch_size || 
        out_shape[1] != rows || 
        out_shape[2] != cols) {
        fprintf(stderr, "Invalid output tensor dimensions\n");
        return false;
    }

    // 执行批量矩阵乘法
    for (int batch = 0; batch < batch_size; batch++) {
        // 计算当前批次的基础偏移量
        int left_offset = batch * rows * inner_dim;
        int right_offset = batch * inner_dim * cols;
        int out_offset = batch * rows * cols;

        // 矩阵乘法
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                float sum = 0.0f;
                for (int k = 0; k < inner_dim; k++) {
                    sum += left_data[left_offset + row * inner_dim + k] * 
                          right_data[right_offset + k * cols + col];
                }
                out_data[out_offset + row * cols + col] = sum;
            }
        }
    }
    return true;
}

// 4D张量乘法: [batch1, batch2, M, K] × [batch1, batch2, K, N] -> [batch1, batch2, M, N]
bool tensor_matmul_4d(const Tensor* left, const Tensor* right, Tensor* output) {
    // 检查维度数量
    if (left->num_dims != 4 || right->num_dims != 4) {
        fprintf(stderr, "Tensors must be 4-dimensional\n");
        return false;
    }

    // 获取维度
    const int* left_shape = left->shape;
    const int* right_shape = right->shape;
    const int* out_shape = output->shape;
    const float* left_data = left->data;
    const float* right_data = right->data;
    float* out_data = output->data;

    int outer_batch = left_shape[0];
    int inner_batch = left_shape[1];
    int rows = left_shape[2];
    int inner_dim = left_shape[3];
    int cols = right_shape[3];

    // 检查维度匹配
    if (right_shape[0] != outer_batch || 
        right_shape[1] != inner_batch || 
        right_shape[2] != inner_dim) {
        fprintf(stderr, "Incompatible dimensions for 4D tensor multiplication\n");
        return false;
    }

    // 检查输出张量维度
    if (output->num_dims != 4 || 
        out_shape[0] != outer_batch || 
        out_shape[1] != inner_batch || 
        out_shape[2] != rows || 
        out_shape[3] != cols) {
        fprintf(stderr, "Invalid output tensor dimensions\n");
        return false;
    }

    // 执行批量矩阵乘法
    for (int batch1 = 0; batch1 < outer_batch; batch1++) {
        for (int batch2 = 0; batch2 < inner_batch; batch2++) {
            // 计算当前批次的基础偏移量
            int left_offset = (batch1 * inner_batch + batch2) * rows * inner_dim;
            int right_offset = (batch1 * inner_batch + batch2) * inner_dim * cols;
            int out_offset = (batch1 * inner_batch + batch2) * rows * cols;

            // 矩阵乘法
            for (int row = 0; row < rows; row++) {
                for (int col = 0; col < cols; col++) {
                    float sum = 0.0f;
                    for (int k = 0; k < inner_dim; k++) {
                        sum += left_data[left_offset + row * inner_dim + k] * 
                              right_data[right_offset + k * cols + col];
                    }
                    out_data[out_offset + row * cols + col] = sum;
                }
            }
        }
    }
    return true;
}

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
