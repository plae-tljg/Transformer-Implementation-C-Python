#ifndef LINEAR_BACKWARD_H
#define LINEAR_BACKWARD_H

#include "tensor_type.h"
#include "linear.h"

// 线性层的梯度结构
typedef struct {
    Tensor* grad_weight;    // 权重的梯度
    Tensor* grad_bias;      // 偏置的梯度 (可选)
} LinearGrad;


bool linear_backward(
    const Tensor* weight,         // [in_features, out_features]
    const Tensor* bias,          // [out_features] 可选
    const Tensor* grad_output,   // [*, out_features], 从下一层传来的梯度
    Tensor* grad_input,          // [*, in_features], 传给上一层的梯度
    LinearGrad* grad             // 累积的参数梯度
);

// 创建线性层梯度结构
LinearGrad* linear_grad_create(int in_features, int out_features);

// 释放线性层梯度结构
void linear_grad_free(LinearGrad* grad);

#endif