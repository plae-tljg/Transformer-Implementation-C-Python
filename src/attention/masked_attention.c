#include "attention.h"
#include "utils.h"
#include <stdlib.h>
#include <math.h>

float* compute_masked_attention(MultiHeadAttention* multi_attn, 
                              float* input,
                              float* attention_mask,
                              int seq_length) {
    if (!multi_attn || !input || !attention_mask || seq_length <= 0) return NULL;
    
    int num_heads = multi_attn->num_heads;
    int model_dim = multi_attn->model_dim;
    int head_dim = model_dim / num_heads;
    
    // 为每个头的输出分配内存
    float* head_outputs = malloc(seq_length * model_dim * sizeof(float));
    if (!head_outputs) return NULL;
    
    // 对每个注意力头进行计算
    for (int h = 0; h < num_heads; h++) {
        SelfAttention* head = multi_attn->attention_heads[h];
        
        // 计算Q、K、V矩阵
        float* Q = matrix_multiply(input, head->query_weights, 
                                 seq_length, model_dim, head_dim);
        float* K = matrix_multiply(input, head->key_weights, 
                                 seq_length, model_dim, head_dim);
        float* V = matrix_multiply(input, head->value_weights, 
                                 seq_length, model_dim, head_dim);
        
        if (!Q || !K || !V) {
            free(Q);
            free(K);
            free(V);
            free(head_outputs);
            return NULL;
        }
        
        // 添加偏置
        if (head->query_bias) {
            matrix_add_inplace(Q, head->query_bias, seq_length * head_dim);
        }
        if (head->key_bias) {
            matrix_add_inplace(K, head->key_bias, seq_length * head_dim);
        }
        if (head->value_bias) {
            matrix_add_inplace(V, head->value_bias, seq_length * head_dim);
        }
        
        // 计算注意力分数
        float* attention_scores = matrix_multiply(Q, K, seq_length, head_dim, seq_length);
        if (!attention_scores) {
            free(Q);
            free(K);
            free(V);
            free(head_outputs);
            return NULL;
        }
        
        // 缩放注意力分数
        float scale = 1.0f / sqrt((float)head_dim);
        matrix_scale(attention_scores, scale, seq_length * seq_length);

        // 应用注意力掩码
        apply_attention_mask(attention_scores, attention_mask, seq_length);

        // 应用 softmax
        softmax(attention_scores, seq_length * seq_length);

        // 计算注意力输出
        float* output = matrix_multiply(attention_scores, V, seq_length, seq_length, head_dim);
        if (!output) {
            free(attention_scores);
            free(V);
            free(head_outputs);
            return NULL;
        }

        // 合并到最终输出
        float* head_output = matrix_multiply(output, head->output_weights,
                                           seq_length, head_dim, model_dim);
        if (!head_output) {
            free(attention_scores);
            free(output);
            free(V);
            free(head_outputs);
            return NULL;
        }

        // 添加到组合输出
        matrix_add_inplace(head_outputs, head_output, seq_length * model_dim);

        // 清理临时内存
        free(Q);
        free(K);
        free(V);
        free(attention_scores);
        free(output);
        free(head_output);
    }

    return head_outputs;
} 