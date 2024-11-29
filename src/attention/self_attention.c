#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "attention.h"
#include <stdbool.h>

// 创建自注意力层
SelfAttention* self_attention_create(int head_dim, int num_heads, bool requires_grad) {
    SelfAttention* self_attn = (SelfAttention*)malloc(sizeof(SelfAttention));
    if (!self_attn) {
        fprintf(stderr, "Failed to allocate memory for self attention\n");
        return NULL;
    }

    self_attn->head_dim = head_dim;
    self_attn->num_heads = num_heads;
    self_attn->requires_grad = requires_grad;

    // 分配权重内存
    int weight_size = head_dim * head_dim;
    self_attn->query_weights = (float*)malloc(weight_size * sizeof(float));
    self_attn->key_weights = (float*)malloc(weight_size * sizeof(float));
    self_attn->value_weights = (float*)malloc(weight_size * sizeof(float));
    self_attn->output_weights = (float*)malloc(weight_size * sizeof(float));

    // 分配偏置内存
    self_attn->query_bias = (float*)malloc(head_dim * sizeof(float));
    self_attn->key_bias = (float*)malloc(head_dim * sizeof(float));
    self_attn->value_bias = (float*)malloc(head_dim * sizeof(float));
    self_attn->output_bias = (float*)malloc(head_dim * sizeof(float));

    // 检查内存分配
    if (!self_attn->query_weights || !self_attn->key_weights || 
        !self_attn->value_weights || !self_attn->output_weights ||
        !self_attn->query_bias || !self_attn->key_bias ||
        !self_attn->value_bias || !self_attn->output_bias) {
        fprintf(stderr, "Failed to allocate memory for attention weights/biases\n");
        self_attention_free(self_attn);
        return NULL;
    }

    return self_attn;
}

// 释放自注意力层
void self_attention_free(SelfAttention* self_attn) {
    if (!self_attn) return;

    free(self_attn->query_weights);
    free(self_attn->key_weights);
    free(self_attn->value_weights);
    free(self_attn->output_weights);
    
    free(self_attn->query_bias);
    free(self_attn->key_bias);
    free(self_attn->value_bias);
    free(self_attn->output_bias);

    free(self_attn);
}

// 自注意力前向传播
void self_attention_forward(
    SelfAttention* self_attn,
    float* input,           // [seq_len, head_dim]
    int seq_length,
    float* output          // [seq_len, head_dim]
) {
    int head_dim = self_attn->head_dim;
    
    // 临时缓冲区
    float* query = (float*)malloc(seq_length * head_dim * sizeof(float));
    float* key = (float*)malloc(seq_length * head_dim * sizeof(float));
    float* value = (float*)malloc(seq_length * head_dim * sizeof(float));
    float* attention_scores = (float*)malloc(seq_length * seq_length * sizeof(float));

    // 计算Q、K、V
    for (int i = 0; i < seq_length; i++) {
        // Q = input * Wq + bq
        for (int j = 0; j < head_dim; j++) {
            float sum = self_attn->query_bias[j];
            for (int k = 0; k < head_dim; k++) {
                sum += input[i * head_dim + k] * self_attn->query_weights[j * head_dim + k];
            }
            query[i * head_dim + j] = sum;
        }

        // K = input * Wk + bk
        for (int j = 0; j < head_dim; j++) {
            float sum = self_attn->key_bias[j];
            for (int k = 0; k < head_dim; k++) {
                sum += input[i * head_dim + k] * self_attn->key_weights[j * head_dim + k];
            }
            key[i * head_dim + j] = sum;
        }

        // V = input * Wv + bv
        for (int j = 0; j < head_dim; j++) {
            float sum = self_attn->value_bias[j];
            for (int k = 0; k < head_dim; k++) {
                sum += input[i * head_dim + k] * self_attn->value_weights[j * head_dim + k];
            }
            value[i * head_dim + j] = sum;
        }
    }

    // 计算注意力分数 Q * K^T / sqrt(d_k)
    float scale = sqrtf(head_dim);
    for (int i = 0; i < seq_length; i++) {
        for (int j = 0; j < seq_length; j++) {
            float score = 0.0f;
            for (int k = 0; k < head_dim; k++) {
                score += query[i * head_dim + k] * key[j * head_dim + k];
            }
            attention_scores[i * seq_length + j] = score / scale;
        }
    }

    // Softmax
    for (int i = 0; i < seq_length; i++) {
        float max_val = attention_scores[i * seq_length];
        for (int j = 1; j < seq_length; j++) {
            if (attention_scores[i * seq_length + j] > max_val) {
                max_val = attention_scores[i * seq_length + j];
            }
        }

        float sum = 0.0f;
        for (int j = 0; j < seq_length; j++) {
            attention_scores[i * seq_length + j] = expf(attention_scores[i * seq_length + j] - max_val);
            sum += attention_scores[i * seq_length + j];
        }

        for (int j = 0; j < seq_length; j++) {
            attention_scores[i * seq_length + j] /= sum;
        }
    }

    // 计算输出 Attention(Q,K,V) = softmax(QK^T/sqrt(d_k))V
    for (int i = 0; i < seq_length; i++) {
        for (int j = 0; j < head_dim; j++) {
            float sum = 0.0f;
            for (int k = 0; k < seq_length; k++) {
                sum += attention_scores[i * seq_length + k] * value[k * head_dim + j];
            }
            output[i * head_dim + j] = sum;
        }
    }

    // 释放临时内存
    free(query);
    free(key);
    free(value);
    free(attention_scores);
}
