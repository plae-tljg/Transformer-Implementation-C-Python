#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdbool.h>
#include "attention.h"

// 创建多头注意力层
MultiHeadAttention* multihead_attention_create(int num_heads, int model_dim, bool requires_grad) {
    MultiHeadAttention* mha = (MultiHeadAttention*)malloc(sizeof(MultiHeadAttention));
    if (!mha) {
        fprintf(stderr, "Failed to allocate memory for multi-head attention\n");
        return NULL;
    }

    mha->num_heads = num_heads;
    mha->model_dim = model_dim;
    mha->head_dim = model_dim / num_heads;
    mha->requires_grad = requires_grad;

    // 创建注意力头
    mha->attention_heads = (SelfAttention**)malloc(num_heads * sizeof(SelfAttention*));
    if (!mha->attention_heads) {
        fprintf(stderr, "Failed to allocate memory for attention heads\n");
        free(mha);
        return NULL;
    }

    // 初始化每个注意力头
    for (int i = 0; i < num_heads; i++) {
        mha->attention_heads[i] = self_attention_create(mha->head_dim, num_heads, requires_grad);
        if (!mha->attention_heads[i]) {
            // 清理已分配的内存
            for (int j = 0; j < i; j++) {
                self_attention_free(mha->attention_heads[j]);
            }
            free(mha->attention_heads);
            free(mha);
            return NULL;
        }
    }

    // 分配输出层权重和偏置
    mha->output_weights = (float*)malloc(model_dim * model_dim * sizeof(float));
    mha->output_bias = (float*)malloc(model_dim * sizeof(float));

    if (!mha->output_weights || !mha->output_bias) {
        fprintf(stderr, "Failed to allocate memory for output weights/bias\n");
        multihead_attention_free(mha);
        return NULL;
    }

    return mha;
}

// 释放多头注意力层
void multihead_attention_free(MultiHeadAttention* mha) {
    if (!mha) return;

    if (mha->attention_heads) {
        for (int i = 0; i < mha->num_heads; i++) {
            self_attention_free(mha->attention_heads[i]);
        }
        free(mha->attention_heads);
    }

    free(mha->output_weights);
    free(mha->output_bias);
    free(mha);
}

// 多头注意力前向传播
void multihead_attention_forward(
    MultiHeadAttention* mha,
    float* input,           // [seq_len, model_dim]
    int seq_length,
    float* output          // [seq_len, model_dim]
) {
    int head_dim = mha->head_dim;
    int model_dim = mha->model_dim;
    int num_heads = mha->num_heads;

    // 为每个头的输出分配内存
    float* head_outputs = (float*)malloc(seq_length * model_dim * sizeof(float));
    float* temp_head_output = (float*)malloc(seq_length * head_dim * sizeof(float));

    // 对每个注意力头进行计算
    for (int h = 0; h < num_heads; h++) {
        // 提取当前头的输入部分
        float* head_input = (float*)malloc(seq_length * head_dim * sizeof(float));
        for (int i = 0; i < seq_length; i++) {
            for (int j = 0; j < head_dim; j++) {
                head_input[i * head_dim + j] = input[i * model_dim + h * head_dim + j];
            }
        }

        // 计算当前头的注意力
        self_attention_forward(mha->attention_heads[h], head_input, seq_length, temp_head_output);

        // 将输出复制到对应位置
        for (int i = 0; i < seq_length; i++) {
            for (int j = 0; j < head_dim; j++) {
                head_outputs[i * model_dim + h * head_dim + j] = temp_head_output[i * head_dim + j];
            }
        }

        free(head_input);
    }

    // 最终输出层变换
    for (int i = 0; i < seq_length; i++) {
        for (int j = 0; j < model_dim; j++) {
            float sum = mha->output_bias[j];
            for (int k = 0; k < model_dim; k++) {
                sum += head_outputs[i * model_dim + k] * mha->output_weights[j * model_dim + k];
            }
            output[i * model_dim + j] = sum;
        }
    }

    // 释放临时内存
    free(head_outputs);
    free(temp_head_output);
}
