#include "training.h"
#include "grad.h"
#include <float.h>

void self_attention_backward(
    SelfAttention* self_attn,
    float* input,           // [seq_len, head_dim]
    float* grad_output,     // [seq_len, head_dim]
    float* attention_mask,  // [seq_len, seq_len]
    int seq_length,
    MultiHeadAttentionGrad* grad
) {
    int head_dim = self_attn->head_dim;
    
    // 分配临时内存
    float* grad_query = (float*)calloc(seq_length * head_dim, sizeof(float));
    float* grad_key = (float*)calloc(seq_length * head_dim, sizeof(float));
    float* grad_value = (float*)calloc(seq_length * head_dim, sizeof(float));
    float* grad_scores = (float*)calloc(seq_length * seq_length, sizeof(float));
    float* softmax_output = (float*)calloc(seq_length * seq_length, sizeof(float));
    
    // 1. 计算注意力分数
    float scale = 1.0f / sqrtf(head_dim);
    for (int i = 0; i < seq_length; i++) {
        for (int j = 0; j < seq_length; j++) {
            float score = 0.0f;
            for (int k = 0; k < head_dim; k++) {
                score += self_attn->query_weights[i * head_dim + k] * 
                         self_attn->key_weights[j * head_dim + k];
            }
            score *= scale;
            if (attention_mask) {
                score += attention_mask[i * seq_length + j];
            }
            softmax_output[i * seq_length + j] = score;
        }
    }
    
    // 2. Softmax反向传播
    for (int i = 0; i < seq_length; i++) {
        // 计算当前行的softmax
        float max_val = -FLT_MAX;
        for (int j = 0; j < seq_length; j++) {
            if (softmax_output[i * seq_length + j] > max_val) {
                max_val = softmax_output[i * seq_length + j];
            }
        }
        
        float sum = 0.0f;
        for (int j = 0; j < seq_length; j++) {
            softmax_output[i * seq_length + j] = expf(softmax_output[i * seq_length + j] - max_val);
            sum += softmax_output[i * seq_length + j];
        }
        
        for (int j = 0; j < seq_length; j++) {
            softmax_output[i * seq_length + j] /= sum;
        }
        
        // 计算softmax梯度
        for (int j = 0; j < seq_length; j++) {
            float grad_sum = 0.0f;
            for (int k = 0; k < head_dim; k++) {
                grad_sum += grad_output[i * head_dim + k] * self_attn->value_weights[j * head_dim + k];
            }
            float softmax_val = softmax_output[i * seq_length + j];
            grad_scores[i * seq_length + j] = softmax_val * (grad_sum - 
                (grad_sum * softmax_val));
        }
    }
    
    // 3. 计算Q、K、V的梯度
    for (int i = 0; i < seq_length; i++) {
        for (int j = 0; j < head_dim; j++) {
            float grad_q = 0.0f;
            float grad_k = 0.0f;
            float grad_v = 0.0f;
            
            for (int k = 0; k < seq_length; k++) {
                // Query梯度
                grad_q += grad_scores[i * seq_length + k] * 
                         self_attn->key_weights[k * head_dim + j] * scale;
                
                // Key梯度
                grad_k += grad_scores[k * seq_length + i] * 
                         self_attn->query_weights[k * head_dim + j] * scale;
                
                // Value梯度
                grad_v += softmax_output[k * seq_length + i] * 
                         grad_output[k * head_dim + j];
            }
            
            grad_query[i * head_dim + j] = grad_q;
            grad_key[i * head_dim + j] = grad_k;
            grad_value[i * head_dim + j] = grad_v;
        }
    }
    
    // 4. 累积梯度到最终的梯度结构中
    for (int i = 0; i < seq_length * head_dim; i++) {
        grad->grad_W_q[i] += grad_query[i];
        grad->grad_W_k[i] += grad_key[i];
        grad->grad_W_v[i] += grad_value[i];
    }
    
    // 释放临时内存
    free(grad_query);
    free(grad_key);
    free(grad_value);
    free(grad_scores);
    free(softmax_output);
}
