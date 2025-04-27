#include "layer_norm_backward.h"

bool layer_norm_backward(
    LayerNorm* ln,
    Tensor* grad_output,    // [batch_size, seq_len, model_dim]
    Tensor* grad_input      // [batch_size, seq_len, model_dim]
) {
    if (!ln || !grad_output || !grad_input) {
        return false;
    }

    int batch_size = grad_output->shape[0];
    int seq_len = grad_output->shape[1];
    int model_dim = grad_output->shape[2];
    float eps = 1e-5f;

    // 创建临时张量
    Tensor* grad_gamma = tensor_create_1d(model_dim);
    Tensor* grad_beta = tensor_create_1d(model_dim);
    if (!grad_gamma || !grad_beta) {
        tensor_free(grad_gamma);
        tensor_free(grad_beta);
        return false;
    }

    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_len; s++) {
            float mean = ln->mean_cache[b * seq_len + s];
            float var = ln->var_cache[b * seq_len + s];
            float std = sqrtf(var + eps);
            float inv_std = 1.0f / std;

            // 计算中间值
            float sum_grad = 0.0f;
            float sum_grad_x = 0.0f;
            
            for (int h = 0; h < model_dim; h++) {
                int idx = (b * seq_len * model_dim) + (s * model_dim) + h;
                float x = ln->input_cache[idx];
                float grad = grad_output->data[idx];
                
                sum_grad += grad;
                sum_grad_x += grad * (x - mean);
            }

            // 计算最终梯度
            for (int h = 0; h < model_dim; h++) {
                int idx = (b * seq_len * model_dim) + (s * model_dim) + h;
                float x = ln->input_cache[idx];
                float grad = grad_output->data[idx];
                
                grad_input->data[idx] = ln->gamma->data[h] * inv_std * (
                    grad - (sum_grad / model_dim) - 
                    ((x - mean) * sum_grad_x) / (model_dim * var)
                );

                // 累积gamma和beta的梯度
                grad_gamma->data[h] += grad * (x - mean) * inv_std;
                grad_beta->data[h] += grad;
            }
        }
    }

    // 更新gamma和beta的梯度
    tensor_add(ln->grad_gamma, grad_gamma, ln->grad_gamma);
    tensor_add(ln->grad_beta, grad_beta, ln->grad_beta);

    tensor_free(grad_gamma);
    tensor_free(grad_beta);
    return true;
}