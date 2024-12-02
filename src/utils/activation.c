#include <math.h>
#include <stdlib.h>
#include "activation.h"

// 创建激活函数
Activation* activation_create(ActivationType type, bool requires_grad) {
    Activation* activation = (Activation*)malloc(sizeof(Activation));
    if (!activation) return NULL;
    
    activation->type = type;
    activation->requires_grad = requires_grad;
    return activation;
}

// 释放激活函数
void activation_free(Activation* activation) {
    if (activation) {
        free(activation);
    }
}

// 前向传播
void activation_forward(
    Activation* activation,
    float* input,        // [batch_size, dim] 
    int batch_size,
    int dim,
    float* output       // [batch_size, dim]
) {
    int size = batch_size * dim;
    switch (activation->type) {
        case ACTIVATION_RELU:
            relu_forward(input, size, output);
            break;
        case ACTIVATION_GELU:
            gelu_forward(input, size, output);
            break;
        case ACTIVATION_SWISH:
            swish_forward(input, size, output);
            break;
        case ACTIVATION_SIGMOID:
            sigmoid_forward(input, size, output);
            break;
        case ACTIVATION_TANH:
            tanh_forward(input, size, output);
            break;
    }
}

// ReLU激活函数实现
void relu_forward(float* input, int size, float* output) {
    for (int i = 0; i < size; i++) {
        output[i] = input[i] > 0 ? input[i] : 0;
    }
}

// GELU激活函数实现
void gelu_forward(float* input, int size, float* output) {
    for (int i = 0; i < size; i++) {
        output[i] = input[i] * 0.5f * (1.0f + tanhf(0.797884f * input[i] + 0.035677f * powf(input[i], 3)));
    }
}

// Swish激活函数实现
void swish_forward(float* input, int size, float* output) {
    for (int i = 0; i < size; i++) {
        float sigmoid = 1.0f / (1.0f + expf(-input[i]));
        output[i] = input[i] * sigmoid;
    }
}

// Sigmoid激活函数实现
void sigmoid_forward(float* input, int size, float* output) {
    for (int i = 0; i < size; i++) {
        output[i] = 1.0f / (1.0f + expf(-input[i]));
    }
}

// Tanh激活函数实现
void tanh_forward(float* input, int size, float* output) {
    for (int i = 0; i < size; i++) {
        output[i] = tanhf(input[i]);
    }
}
