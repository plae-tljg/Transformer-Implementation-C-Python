#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <stdbool.h>

// 激活函数类型枚举
typedef enum {
    ACTIVATION_RELU,
    ACTIVATION_GELU,
    ACTIVATION_SWISH,
    ACTIVATION_SIGMOID,
    ACTIVATION_TANH
} ActivationType;

// 激活函数结构体
typedef struct {
    ActivationType type;
    bool requires_grad;
} Activation;

// 创建激活函数
Activation* activation_create(ActivationType type, bool requires_grad);

// 释放激活函数
void activation_free(Activation* activation);

// 前向传播
void activation_forward(
    Activation* activation,
    float* input,        // [batch_size, dim]
    int batch_size,
    int dim,
    float* output       // [batch_size, dim]
);

// 单独的激活函数实现
void relu_forward(float* input, int size, float* output);
void gelu_forward(float* input, int size, float* output);
void swish_forward(float* input, int size, float* output);
void sigmoid_forward(float* input, int size, float* output);
void tanh_forward(float* input, int size, float* output);

#endif