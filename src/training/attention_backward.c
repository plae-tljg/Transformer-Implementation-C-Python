#include "training.h"
#include "utils.h"
#include <stdlib.h>
#include <math.h>

void update_self_attention_gradients(SelfAttention* attn, float learning_rate) {
    if (!attn || !attn->requires_grad) return;
    
    // 更新查询权重
    for (int i = 0; i < attn->head_dim * attn->head_dim; i++) {
        attn->query_weights[i] -= learning_rate * attn->query_gradients[i];
        attn->query_gradients[i] = 0.0f;  // 重置梯度
    }
    
    // 更新键权重
    for (int i = 0; i < attn->head_dim * attn->head_dim; i++) {
        attn->key_weights[i] -= learning_rate * attn->key_gradients[i];
        attn->key_gradients[i] = 0.0f;  // 重置梯度
    }
    
    // 更新值权重
    for (int i = 0; i < attn->head_dim * attn->head_dim; i++) {
        attn->value_weights[i] -= learning_rate * attn->value_gradients[i];
        attn->value_gradients[i] = 0.0f;  // 重置梯度
    }
}

void update_attention_gradients(MultiHeadAttention* mha, float learning_rate) {
    if (!mha || !mha->requires_grad) return;
    
    // 更新每个注意力头的梯度
    for (int h = 0; h < mha->num_heads; h++) {
        if (mha->attention_heads[h]) {
            update_self_attention_gradients(mha->attention_heads[h], learning_rate);
        }
    }
}

void accumulate_attention_gradients(MultiHeadAttention* mha, float* grad) {
    if (!mha || !mha->requires_grad || !grad) return;
    
    // 为每个注意力头累积梯度
    for (int h = 0; h < mha->num_heads; h++) {
        SelfAttention* head = mha->attention_heads[h];
        if (!head || !head->requires_grad) continue;
        
        // 计算该头的梯度偏移
        int offset = h * mha->head_dim;
        
        // 累积查询、键、值的梯度
        for (int i = 0; i < mha->head_dim * mha->head_dim; i++) {
            head->query_gradients[i] += grad[offset + i];
            head->key_gradients[i] += grad[offset + mha->head_dim + i];
            head->value_gradients[i] += grad[offset + 2 * mha->head_dim + i];
        }
    }
}

float* attention_backward(MultiHeadAttention* attn, float* grad_output) {
    if (!attn || !grad_output) return NULL;
    
    // TODO: 实现注意力反向传播
    return NULL;
}

float* masked_attention_backward(MultiHeadAttention* attn, float* grad_output) {
    if (!attn || !grad_output) return NULL;
    
    // TODO: 实现带掩码的注意力反向传播
    return NULL;
} 