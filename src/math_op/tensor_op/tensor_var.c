#include "tensor_var.h"
#include <math.h>

void compute_means_3d(const float* input, float* means, 
                     int batch_size, int seq_len, int hidden_dim) {
    #pragma omp parallel for collapse(2)  // 为将来的OpenMP优化做准备
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_len; s++) {
            float sum = 0.0f;
            int offset = (b * seq_len + s) * hidden_dim;
            
            // 向量化友好的累加循环
            #pragma omp simd reduction(+:sum)
            for (int h = 0; h < hidden_dim; h++) {
                sum += input[offset + h];
            }
            
            means[b * seq_len + s] = sum / hidden_dim;
        }
    }
}

void compute_variances_3d(const float* input, const float* means, float* variances,
                         int batch_size, int seq_len, int hidden_dim) {
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_len; s++) {
            float sum_sq = 0.0f;
            int offset = (b * seq_len + s) * hidden_dim;
            float mean = means[b * seq_len + s];
            
            #pragma omp simd reduction(+:sum_sq)
            for (int h = 0; h < hidden_dim; h++) {
                float diff = input[offset + h] - mean;
                sum_sq += diff * diff;
            }
            
            variances[b * seq_len + s] = sum_sq / hidden_dim;
        }
    }
}

void normalize_and_scale_3d(const float* input, float* output,
                          const float* means, const float* variances,
                          const float* gamma, const float* beta,
                          int batch_size, int seq_len, int hidden_dim,
                          float eps) {
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_len; s++) {
            int offset = (b * seq_len + s) * hidden_dim;
            float mean = means[b * seq_len + s];
            float std = sqrt(variances[b * seq_len + s] + eps);
            
            #pragma omp simd
            for (int h = 0; h < hidden_dim; h++) {
                float normalized = (input[offset + h] - mean) / std;
                output[offset + h] = gamma[h] * normalized + beta[h];
            }
        }
    }
}

// 整合所有步骤的便捷函数
void layer_norm_forward_3d(const float* input, float* output,
                         const float* gamma, const float* beta,
                         int batch_size, int seq_len, int hidden_dim,
                         float eps) {
    // 分配临时缓冲区
    float* means = (float*)malloc(batch_size * seq_len * sizeof(float));
    float* variances = (float*)malloc(batch_size * seq_len * sizeof(float));
    
    // 计算均值
    compute_means_3d(input, means, batch_size, seq_len, hidden_dim);
    
    // 计算方差
    compute_variances_3d(input, means, variances, batch_size, seq_len, hidden_dim);
    
    // 归一化和缩放
    normalize_and_scale_3d(input, output, means, variances, gamma, beta,
                          batch_size, seq_len, hidden_dim, eps);
    
    // 释放临时缓冲区
    free(means);
    free(variances);
}