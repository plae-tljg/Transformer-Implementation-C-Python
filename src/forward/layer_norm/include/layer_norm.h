#ifndef LAYER_NORM_H
#define LAYER_NORM_H

#include "tensor.h"

typedef struct LayerNorm {
    float eps;          // 用于数值稳定性的小值
    Tensor* gamma;      // 缩放参数
    Tensor* beta;       // 偏移参数
    int normalized_dim; // 需要归一化的维度
} LayerNorm;

// 创建和释放函数
LayerNorm* layer_norm_create(int normalized_shape, float eps);
void layer_norm_free(LayerNorm* ln);

// 前向计算函数
Tensor* layer_norm_forward(LayerNorm* ln, Tensor* input);

#endif
