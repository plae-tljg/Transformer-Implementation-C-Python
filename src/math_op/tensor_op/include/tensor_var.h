#ifndef TENSOR_VAR_H
#define TENSOR_VAR_H

// 计算3D张量的均值 [batch_size, seq_len, hidden_dim]
void compute_means_3d(const float* input, float* means,
                     int batch_size, int seq_len, int hidden_dim);

// 计算3D张量的方差
void compute_variances_3d(const float* input, const float* means, float* variances,
                         int batch_size, int seq_len, int hidden_dim);

// 执行归一化和缩放操作
void normalize_and_scale_3d(const float* input, float* output,
                          const float* means, const float* variances,
                          const float* gamma, const float* beta,
                          int batch_size, int seq_len, int hidden_dim,
                          float eps);

// 整合的前向计算函数
void layer_norm_forward_3d(const float* input, float* output,
                         const float* gamma, const float* beta,
                         int batch_size, int seq_len, int hidden_dim,
                         float eps);

#endif
