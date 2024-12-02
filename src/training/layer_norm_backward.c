#include "training.h"
#include "grad.h"
#include <stdlib.h>

void layer_norm_backward(
    LayerNorm* ln,
    float* input,          // [batch_size, normalized_shape]
    float* grad_output,    // [batch_size, normalized_shape]
    int batch_size,
    float* grad_input,     // [batch_size, normalized_shape]
    LayerNormGrad* grad
) {
    for (int i = 0; i < batch_size; i++) {
        float* current_input = input + i * ln->normalized_shape;
        float* current_grad_output = grad_output + i * ln->normalized_shape;
        float* current_grad_input = grad_input + i * ln->normalized_shape;
        
        // 1. 计算均值和方差
        float mean = 0.0f;
        for (int j = 0; j < ln->normalized_shape; j++) {
            mean += current_input[j];
        }
        mean /= ln->normalized_shape;
        
        float variance = 0.0f;
        for (int j = 0; j < ln->normalized_shape; j++) {
            float diff = current_input[j] - mean;
            variance += diff * diff;
        }
        variance /= ln->normalized_shape;
        float std = sqrtf(variance + ln->epsilon);
        
        // 2. 计算归一化值
        float* normalized = (float*)malloc(ln->normalized_shape * sizeof(float));
        for (int j = 0; j < ln->normalized_shape; j++) {
            normalized[j] = (current_input[j] - mean) / std;
        }
        
        // 3. 计算梯度
        float sum_grad = 0.0f;
        float sum_grad_norm = 0.0f;
        for (int j = 0; j < ln->normalized_shape; j++) {
            sum_grad += current_grad_output[j];
            sum_grad_norm += current_grad_output[j] * normalized[j];
        }
        
        // 计算gamma和beta的梯度
        for (int j = 0; j < ln->normalized_shape; j++) {
            grad->grad_gamma[j] += current_grad_output[j] * normalized[j];
            grad->grad_beta[j] += current_grad_output[j];
        }
        
        // 计算输入的梯度
        float factor = 1.0f / (ln->normalized_shape * std);
        for (int j = 0; j < ln->normalized_shape; j++) {
            float norm_grad = current_grad_output[j] * ln->gamma[j];
            current_grad_input[j] = factor * (
                ln->normalized_shape * norm_grad 
                - sum_grad
                - normalized[j] * sum_grad_norm
            );
        }
        
        free(normalized);
    }
}