#include "training.h"
#include "utils.h"
#include <string.h>
#include <stdlib.h>

void feed_forward_backward(FeedForward* ff, float* grad_output, float* input) {
    if (!ff || !grad_output || !input || !ff->requires_grad) return;
    
    const int batch_size = 1; // 暂时假设批量大小为1
    
    // 第二层的反向传播
    float* d_hidden = matrix_multiply_transpose(grad_output, ff->weights2,
                                              batch_size, ff->input_dim, ff->hidden_dim);
    
    // 计算权重的梯度
    float* weight_grad2 = matrix_multiply_transpose2(ff->hidden_states, grad_output,
                                                   batch_size, ff->hidden_dim, ff->input_dim);
    
    // 累积偏置的梯度
    for (int i = 0; i < ff->input_dim; i++) {
        ff->bias_gradients2[i] += grad_output[i];
    }
    
    // ReLU的反向传播
    for (int i = 0; i < batch_size * ff->hidden_dim; i++) {
        d_hidden[i] *= (ff->hidden_states[i] > 0) ? 1.0f : 0.0f;
    }
    
    // 第一层的反向传播
    float* d_input = matrix_multiply_transpose(d_hidden, ff->weights1,
                                             batch_size, ff->hidden_dim, ff->input_dim);
    
    // 计算权重的梯度
    float* weight_grad1 = matrix_multiply_transpose2(input, d_hidden,
                                                   batch_size, ff->input_dim, ff->hidden_dim);
    
    // 累积偏置的梯度
    for (int i = 0; i < ff->hidden_dim; i++) {
        ff->bias_gradients1[i] += d_hidden[i];
    }
    
    // 累积权重梯度
    for (int i = 0; i < ff->input_dim * ff->hidden_dim; i++) {
        ff->weight_gradients1[i] += weight_grad1[i];
    }
    for (int i = 0; i < ff->hidden_dim * ff->input_dim; i++) {
        ff->weight_gradients2[i] += weight_grad2[i];
    }
    
    // 清理临时内存
    free(d_hidden);
    free(d_input);
    free(weight_grad1);
    free(weight_grad2);
}

void update_feed_forward_gradients(FeedForward* ff, float learning_rate) {
    if (!ff || !ff->requires_grad) return;
    
    const int input_dim = ff->input_dim;
    const int hidden_dim = ff->hidden_dim;
    
    // 更新第一层权重和偏置
    for (int i = 0; i < input_dim * hidden_dim; i++) {
        ff->weights1[i] -= learning_rate * ff->weight_gradients1[i];
        ff->weight_gradients1[i] = 0.0f;
    }
    
    for (int i = 0; i < hidden_dim; i++) {
        ff->bias1[i] -= learning_rate * ff->bias_gradients1[i];
        ff->bias_gradients1[i] = 0.0f;
    }
    
    // 更新第二层权重和偏置
    for (int i = 0; i < hidden_dim * input_dim; i++) {
        ff->weights2[i] -= learning_rate * ff->weight_gradients2[i];
        ff->weight_gradients2[i] = 0.0f;
    }
    
    for (int i = 0; i < input_dim; i++) {
        ff->bias2[i] -= learning_rate * ff->bias_gradients2[i];
        ff->bias_gradients2[i] = 0.0f;
    }
}

void accumulate_feed_forward_gradients(FeedForward* ff, float* grad) {
    if (!ff || !ff->requires_grad || !grad) return;
    
    const int input_dim = ff->input_dim;
    const int hidden_dim = ff->hidden_dim;
    
    // 累积第一层的梯度
    for (int i = 0; i < input_dim * hidden_dim; i++) {
        ff->weight_gradients1[i] += grad[i];
    }
    
    // 累积第二层的梯度
    for (int i = 0; i < hidden_dim * input_dim; i++) {
        ff->weight_gradients2[i] += grad[input_dim * hidden_dim + i];
    }
} 