#include "tensor_std.h"
#include <math.h>
#include <stdlib.h>

bool compute_means_3d(const Tensor* input, Tensor* means) {
    if (!input || !means) return false;
    
    int batch_size = input->shape[0];
    int seq_len = input->shape[1];
    int hidden_dim = input->shape[2];
    
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_len; s++) {
            float sum = 0.0f;
            int offset = (b * seq_len + s) * hidden_dim;
            
            #pragma omp simd reduction(+:sum)
            for (int h = 0; h < hidden_dim; h++) {
                sum += input->data[offset + h];
            }
            
            means->data[b * seq_len + s] = sum / hidden_dim;
        }
    }
    return true;
}

bool compute_variances_3d(const Tensor* input, const Tensor* means, Tensor* variances) {
    if (!input || !means || !variances) return false;
    
    int batch_size = input->shape[0];
    int seq_len = input->shape[1];
    int hidden_dim = input->shape[2];
    
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_len; s++) {
            float sum_sq = 0.0f;
            int offset = (b * seq_len + s) * hidden_dim;
            float mean = means->data[b * seq_len + s];
            
            #pragma omp simd reduction(+:sum_sq)
            for (int h = 0; h < hidden_dim; h++) {
                float diff = input->data[offset + h] - mean;
                sum_sq += diff * diff;
            }
            
            variances->data[b * seq_len + s] = sum_sq / hidden_dim;
        }
    }
    return true;
}

bool normalize_and_scale_3d(const Tensor* input, Tensor* output,
                          const Tensor* means, const Tensor* variances,
                          const Tensor* gamma, const Tensor* beta,
                          float eps) {
    if (!input || !output || !means || !variances || !gamma || !beta) return false;
                          
    int batch_size = input->shape[0];
    int seq_len = input->shape[1];
    int hidden_dim = input->shape[2];
    
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_len; s++) {
            int offset = (b * seq_len + s) * hidden_dim;
            float mean = means->data[b * seq_len + s];
            float std = sqrt(variances->data[b * seq_len + s] + eps);
            
            #pragma omp simd
            for (int h = 0; h < hidden_dim; h++) {
                float normalized = (input->data[offset + h] - mean) / std;
                output->data[offset + h] = gamma->data[h] * normalized + beta->data[h];
            }
        }
    }
    return true;
}

bool layer_norm_forward_3d(const Tensor* input, Tensor* output,
                         const Tensor* gamma, const Tensor* beta,
                         float eps) {
    if (!input || !output || !gamma || !beta) return false;
                         
    // 创建临时张量
    int means_shape[] = {input->shape[0], input->shape[1]};
    Tensor* means = tensor_create(means_shape, 2);
    Tensor* variances = tensor_create(means_shape, 2);
    
    if (!means || !variances) {
        if (means) tensor_free(means);
        if (variances) tensor_free(variances);
        return false;
    }
    
    // 计算均值
    if (!compute_means_3d(input, means)) {
        tensor_free(means);
        tensor_free(variances);
        return false;
    }
    
    // 计算方差
    if (!compute_variances_3d(input, means, variances)) {
        tensor_free(means);
        tensor_free(variances);
        return false;
    }
    
    // 归一化和缩放
    bool success = normalize_and_scale_3d(input, output, means, variances, gamma, beta, eps);
    
    // 释放临时缓冲区
    tensor_free(means);
    tensor_free(variances);
    
    return success;
}