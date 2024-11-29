#include "attention.h"
#include "utils.h"
#include <stdlib.h>
#include <string.h>

float* compute_self_attention_head(SelfAttention* self_attn, float* input, int seq_length) {
    if (!self_attn || !input || seq_length <= 0) return NULL;

    const int head_dim = self_attn->head_dim;

    // 计算查询、键、值矩阵
    float* Q = matrix_multiply(input, self_attn->query_weights,
                             seq_length, head_dim, head_dim);
    if (!Q) return NULL;

    float* K = matrix_multiply(input, self_attn->key_weights,
                             seq_length, head_dim, head_dim);
    if (!K) {
        free(Q);
        return NULL;
    }

    float* V = matrix_multiply(input, self_attn->value_weights,
                             seq_length, head_dim, head_dim);
    if (!V) {
        free(Q);
        free(K);
        return NULL;
    }

    // 添加偏置
    if (self_attn->query_bias) {
        matrix_add_inplace(Q, self_attn->query_bias, seq_length * head_dim);
    }
    if (self_attn->key_bias) {
        matrix_add_inplace(K, self_attn->key_bias, seq_length * head_dim);
    }
    if (self_attn->value_bias) {
        matrix_add_inplace(V, self_attn->value_bias, seq_length * head_dim);
    }

    // 计算注意力分数
    float* scores = matrix_multiply(Q, K, seq_length, head_dim, seq_length);
    if (!scores) {
        free(Q);
        free(K);
        free(V);
        return NULL;
    }

    // 缩放注意力分数
    float scale = 1.0f / sqrtf((float)head_dim);
    matrix_scale(scores, scale, seq_length * seq_length);

    // 应用 softmax
    softmax(scores, seq_length * seq_length);

    // 计算注意力输出
    float* attention_output = matrix_multiply(scores, V, 
                                            seq_length, seq_length, head_dim);
    if (!attention_output) {
        free(Q);
        free(K);
        free(V);
        free(scores);
        return NULL;
    }

    // 应用输出投影
    float* final_output = matrix_multiply(attention_output, self_attn->output_weights,
                                        seq_length, head_dim, head_dim);

    // 清理中间结果
    free(Q);
    free(K);
    free(V);
    free(scores);
    free(attention_output);

    return final_output;
} 