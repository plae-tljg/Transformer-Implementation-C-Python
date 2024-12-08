#ifndef LINEAR_H
#define LINEAR_H

#include "tensor_type.h"

// 线性层结构
typedef struct {
    Tensor* weight;    // 权重矩阵 [in_features, out_features]
    Tensor* bias;      // 偏置向量 [out_features]
} Linear;

// 创建线性层
Linear* linear_create(int in_features, int out_features);

// 线性层前向传播
// input: [batch_size, seq_len, in_features]
// output: [batch_size, seq_len, out_features]
// 执行: output = input * weight^T + bias
bool linear_forward(Linear* linear, Tensor* input, Tensor* output);

// 释放线性层
void linear_free(Linear* linear);



// 线性层的梯度结构
typedef struct {
    Tensor* grad_weight;    // 权重的梯度
    Tensor* grad_bias;      // 偏置的梯度 (可选)
} LinearGrad;


// 创建线性层梯度结构
LinearGrad* linear_grad_create(int in_features, int out_features);

// 释放线性层梯度结构
void linear_grad_free(LinearGrad* grad);

#endif