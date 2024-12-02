#include "training.h"
#include "grad.h"
void feed_forward_backward(
    FeedForward* ff,
    float* input,           // [batch_size, input_dim]
    float* grad_output,     // [batch_size, input_dim]
    int batch_size,
    float* grad_input,      // [batch_size, input_dim]
    FeedForwardGrad* grad
) {
    // 1. 保存前向传播的中间值
    float* hidden = (float*)malloc(batch_size * ff->hidden_dim * sizeof(float));
    float* hidden_pre_act = (float*)malloc(batch_size * ff->hidden_dim * sizeof(float));
    float* grad_hidden = (float*)malloc(batch_size * ff->hidden_dim * sizeof(float));
    
    // 2. 反向传播通过第二层
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < ff->hidden_dim; h++) {
            float grad_sum = 0.0f;
            for (int i = 0; i < ff->input_dim; i++) {
                grad_sum += grad_output[b * ff->input_dim + i] * ff->w2[i * ff->hidden_dim + h];
            }
            grad_hidden[b * ff->hidden_dim + h] = grad_sum;
        }
    }
    
    // 3. ReLU的反向传播
    for (int i = 0; i < batch_size * ff->hidden_dim; i++) {
        grad_hidden[i] *= (hidden_pre_act[i] > 0) ? 1.0f : 0.0f;
    }
    
    // 4. 累积参数梯度
    // w1的梯度
    for (int h = 0; h < ff->hidden_dim; h++) {
        for (int i = 0; i < ff->input_dim; i++) {
            float grad_sum = 0.0f;
            for (int b = 0; b < batch_size; b++) {
                grad_sum += grad_hidden[b * ff->hidden_dim + h] * input[b * ff->input_dim + i];
            }
            grad->grad_w1[h * ff->input_dim + i] += grad_sum;
        }
    }

    // b1的梯度
    for (int h = 0; h < ff->hidden_dim; h++) {
        float grad_sum = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            grad_sum += grad_hidden[b * ff->hidden_dim + h];
        }
        grad->grad_b1[h] += grad_sum;
    }

    // w2的梯度
    for (int i = 0; i < ff->input_dim; i++) {
        for (int h = 0; h < ff->hidden_dim; h++) {
            float grad_sum = 0.0f;
            for (int b = 0; b < batch_size; b++) {
                grad_sum += grad_output[b * ff->input_dim + i] * hidden[b * ff->hidden_dim + h];
            }
            grad->grad_w2[i * ff->hidden_dim + h] += grad_sum;
        }
    }

    // b2的梯度
    for (int i = 0; i < ff->input_dim; i++) {
        float grad_sum = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            grad_sum += grad_output[b * ff->input_dim + i];
        }
        grad->grad_b2[i] += grad_sum;
    }

    // 5. 计算输入的梯度
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < ff->input_dim; i++) {
            float grad_sum = 0.0f;
            for (int h = 0; h < ff->hidden_dim; h++) {
                grad_sum += grad_hidden[b * ff->hidden_dim + h] * ff->w1[h * ff->input_dim + i];
            }
            grad_input[b * ff->input_dim + i] = grad_sum;
        }
    }
    
    free(hidden);
    free(hidden_pre_act);
    free(grad_hidden);
} 