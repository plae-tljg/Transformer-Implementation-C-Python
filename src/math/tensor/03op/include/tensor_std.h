#ifndef TENSOR_VAR_H
#define TENSOR_VAR_H

#include "tensor_type.h"

// 计算3D张量的均值 [batch_size, seq_len, hidden_dim]
bool compute_means_3d(const Tensor* input, Tensor* means);

// 计算3D张量的方差
bool compute_variances_3d(const Tensor* input, const Tensor* means, Tensor* variances);

// 执行归一化和缩放操作
bool normalize_and_scale_3d(const Tensor* input, Tensor* output,
                          const Tensor* means, const Tensor* variances,
                          const Tensor* gamma, const Tensor* beta,
                          float eps);

// 整合的前向计算函数
bool layer_norm_forward_3d(const Tensor* input, Tensor* output,
                         const Tensor* gamma, const Tensor* beta,
                         float eps);

#endif
