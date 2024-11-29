#include "attention.h"
#include "utils.h"
#include <stdlib.h>
#include <string.h>

float* compute_multi_head_attention(MultiHeadAttention* multi_attn,
                                  float* query,
                                  float* key,
                                  float* value,
                                  float* attention_mask,
                                  int seq_length) {
    if (!multi_attn || !query || !key || !value || seq_length <= 0) return NULL;
    
    int num_heads = multi_attn->num_heads;
    int model_dim = multi_attn->model_dim;
    int head_dim = model_dim / num_heads;
    
    // 为每个头的输出分配内存
    float* head_outputs = malloc(seq_length * model_dim * sizeof(float));
    if (!head_outputs) return NULL;
    memset(head_outputs, 0, seq_length * model_dim * sizeof(float));
    
    // 对每个注意力头进行计算
    for (int h = 0; h < num_heads; h++) {
        SelfAttention* head = multi_attn->attention_heads[h];
        
        // 计算Q、K、V矩阵
        float* Q = matrix_multiply(query, head->query_weights, 
                                 seq_length, model_dim, head_dim);
        float* K = matrix_multiply(key, head->key_weights, 
                                 seq_length, model_dim, head_dim);
        float* V = matrix_multiply(value, head->value_weights, 
                                 seq_length, model_dim, head_dim);
        
        if (!Q || !K || !V) {
            free(Q);
            free(K);
            free(V);
            free(head_outputs);
            return NULL;
        }
        
        // 添加偏置
        if (head->query_bias) matrix_add_inplace(Q, head->query_bias, seq_length * head_dim);
        if (head->key_bias) matrix_add_inplace(K, head->key_bias, seq_length * head_dim);
        if (head->value_bias) matrix_add_inplace(V, head->value_bias, seq_length * head_dim);
        
        // 计算注意力分数
        float* attention_scores = matrix_multiply(Q, K, seq_length, head_dim, seq_length);
        if (!attention_scores) {
            free(Q); free(K); free(V); free(head_outputs);
            return NULL;
        }
        
        // 缩放注意力分数
        float scale = 1.0f / sqrt((float)head_dim);
        matrix_scale(attention_scores, scale, seq_length * seq_length);

        // 如果提供了掩码，则应用掩码
        if (attention_mask) {
            apply_attention_mask(attention_scores, attention_mask, seq_length);
        }

        // 应用 softmax
        softmax(attention_scores, seq_length * seq_length);

        // 计算注意力输出
        float* output = matrix_multiply(attention_scores, V, 
                                      seq_length, seq_length, head_dim);
        if (!output) {
            free(Q); free(K); free(V); 
            free(attention_scores); free(head_outputs);
            return NULL;
        }

        // 合并到最终输出
        float* head_output = matrix_multiply(output, head->output_weights,
                                           seq_length, head_dim, model_dim);
        if (!head_output) {
            free(Q); free(K); free(V);
            free(attention_scores); free(output); free(head_outputs);
            return NULL;
        }

        // 添加到组合输出
        matrix_add_inplace(head_outputs, head_output, seq_length * model_dim);

        // 清理临时内存
        free(Q); free(K); free(V);
        free(attention_scores);
        free(output);
        free(head_output);
    }
    
    // 应用输出变换和偏置
    float* final_output = matrix_multiply(head_outputs, multi_attn->output_weights,
                                        seq_length, model_dim, model_dim);
    free(head_outputs);
    
    if (!final_output) return NULL;
    
    if (multi_attn->output_bias) {
        for (int i = 0; i < seq_length; i++) {
            for (int j = 0; j < model_dim; j++) {
                final_output[i * model_dim + j] += multi_attn->output_bias[j];
            }
        }
    }
    
    return final_output;
}

void free_attention(MultiHeadAttention* mha) {
    if (!mha) return;
    
    // 释放注意力头
    if (mha->attention_heads) {
        for (int i = 0; i < mha->num_heads; i++) {
            if (mha->attention_heads[i]) {
                free_self_attention(mha->attention_heads[i]);
            }
        }
        free(mha->attention_heads);
    }
    
    // 释放权重和偏置
    if (mha->output_weights) {
        free(mha->output_weights);
    }
    if (mha->output_bias) {
        free(mha->output_bias);
    }
    
    // 释放梯度
    if (mha->requires_grad) {
        if (mha->output_weights_gradients) {
            free(mha->output_weights_gradients);
        }
        if (mha->output_bias_gradients) {
            free(mha->output_bias_gradients);
        }
        if (mha->attention_weights_gradients) {
            free(mha->attention_weights_gradients);
        }
    }
    
    // 释放缓存
    if (mha->input_cache) {
        free(mha->input_cache);
    }
    if (mha->attention_output_cache) {
        free(mha->attention_output_cache);
    }
    
    free(mha);
}

MultiHeadAttention* create_multi_head_attention(int model_dim, int num_heads) {
    if (model_dim <= 0 || num_heads <= 0 || model_dim % num_heads != 0) return NULL;
    
    MultiHeadAttention* mha = malloc(sizeof(MultiHeadAttention));
    if (!mha) return NULL;
    
    // 初始化基本参数
    mha->num_heads = num_heads;
    mha->model_dim = model_dim;
    mha->head_dim = model_dim / num_heads;
    mha->requires_grad = true;
    
    // 初始化所有指针为 NULL
    mha->attention_heads = NULL;
    mha->output_weights = NULL;
    mha->output_bias = NULL;
    mha->output_weights_gradients = NULL;
    mha->output_bias_gradients = NULL;
    mha->attention_weights_gradients = NULL;
    mha->input_cache = NULL;
    mha->attention_scores_cache = NULL;
    mha->attention_output_cache = NULL;
    
    // 初始化注意力头
    mha->attention_heads = malloc(num_heads * sizeof(SelfAttention*));
    if (!mha->attention_heads) goto cleanup;
    
    for (int i = 0; i < num_heads; i++) {
        mha->attention_heads[i] = create_self_attention(mha->head_dim);
        if (!mha->attention_heads[i]) goto cleanup;
    }
    
    // 分配输出层权重和偏置
    mha->output_weights = malloc(model_dim * model_dim * sizeof(float));
    mha->output_bias = malloc(model_dim * sizeof(float));
    
    if (!mha->output_weights || !mha->output_bias) goto cleanup;
    
    // 如果需要梯度，分配梯度空间
    if (mha->requires_grad) {
        mha->output_weights_gradients = malloc(model_dim * model_dim * sizeof(float));
        mha->output_bias_gradients = malloc(model_dim * sizeof(float));
        mha->attention_weights_gradients = malloc(model_dim * model_dim * sizeof(float));
        
        if (!mha->output_weights_gradients || !mha->output_bias_gradients || 
            !mha->attention_weights_gradients) goto cleanup;
            
        // 初始化梯度为0
        memset(mha->output_weights_gradients, 0, model_dim * model_dim * sizeof(float));
        memset(mha->output_bias_gradients, 0, model_dim * sizeof(float));
        memset(mha->attention_weights_gradients, 0, model_dim * model_dim * sizeof(float));
    }
    
    // 初始化权重和偏置
    for (int i = 0; i < model_dim * model_dim; i++) {
        mha->output_weights[i] = ((float)rand() / RAND_MAX) * 0.02f - 0.01f;
    }
    for (int i = 0; i < model_dim; i++) {
        mha->output_bias[i] = 0.0f;
    }
    
    return mha;

cleanup:
    free_attention(mha);
    return NULL;
} 