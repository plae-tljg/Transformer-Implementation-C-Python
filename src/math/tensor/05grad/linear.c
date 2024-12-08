#include "linear.h"
#include "tensor_mul.h"
#include "tensor_add.h"
#include <stdlib.h>
#include <stdio.h>

Linear* linear_create(int in_features, int out_features) {
    if (in_features <= 0 || out_features <= 0) {
        fprintf(stderr, "Invalid dimensions for linear layer\n");
        return NULL;
    }

    // 分配Linear结构体内存
    Linear* linear = (Linear*)malloc(sizeof(Linear));
    if (!linear) {
        fprintf(stderr, "Failed to allocate memory for Linear\n");
        return NULL;
    }

    // 创建权重矩阵 [in_features, out_features]
    int weight_shape[] = {in_features, out_features};
    linear->weight = tensor_create(weight_shape, 2);
    if (!linear->weight) {
        fprintf(stderr, "Failed to create weight tensor\n");
        free(linear);
        return NULL;
    }

    // 创建偏置向量 [out_features]
    int bias_shape[] = {out_features};
    linear->bias = tensor_create(bias_shape, 1);
    if (!linear->bias) {
        fprintf(stderr, "Failed to create bias tensor\n");
        tensor_free(linear->weight);
        free(linear);
        return NULL;
    }

    return linear;
}


LinearGrad* linear_grad_create(int in_features, int out_features) {
    if (in_features <= 0 || out_features <= 0) {
        fprintf(stderr, "Invalid dimensions for linear gradient\n");
        return NULL;
    }

    // 分配LinearGrad结构体内存
    LinearGrad* grad = (LinearGrad*)malloc(sizeof(LinearGrad));
    if (!grad) {
        fprintf(stderr, "Failed to allocate memory for LinearGrad\n");
        return NULL;
    }

    // 创建权重梯度张量 [in_features, out_features]
    int weight_shape[] = {in_features, out_features};
    grad->grad_weight = tensor_create(weight_shape, 2);
    if (!grad->grad_weight) {
        fprintf(stderr, "Failed to create weight gradient tensor\n");
        free(grad);
        return NULL;
    }
    tensor_fill(grad->grad_weight, 0.0f); // 初始化为0

    // 创建偏置梯度张量 [out_features]
    int bias_shape[] = {out_features};
    grad->grad_bias = tensor_create(bias_shape, 1);
    if (!grad->grad_bias) {
        fprintf(stderr, "Failed to create bias gradient tensor\n");
        tensor_free(grad->grad_weight);
        free(grad);
        return NULL;
    }
    tensor_fill(grad->grad_bias, 0.0f); // 初始化为0

    return grad;
}

bool linear_forward(Linear* linear, const Tensor* input, Tensor* output) {
    if (!linear || !input || !output) {
        fprintf(stderr, "NULL pointer passed to linear_forward\n");
        return false;
    }

    // 检查输入维度
    if (input->num_dims < 2) {
        fprintf(stderr, "Input tensor must have at least 2 dimensions\n");
        return false;
    }

    // 检查最后一个维度是否匹配权重的输入特征数
    if (input->shape[input->num_dims - 1] != linear->weight->shape[1]) {
        fprintf(stderr, "Input features dimension mismatch\n");
        return false;
    }

    // 执行矩阵乘法: output = input * weight^T
    if (!tensor_mul_3_2(input, linear->weight, output)) {
        fprintf(stderr, "Matrix multiplication failed in linear_forward\n");
        return false;
    }

    // 添加偏置
    if (!tensor_add_bias_3d(output, linear->bias, output)) {
        fprintf(stderr, "Failed to add bias in linear_forward\n");
        return false;
    }

    return true;
}


void linear_free(Linear* linear) {
    if (linear) {
        if (linear->weight) {
            tensor_free(linear->weight);
        }
        if (linear->bias) {
            tensor_free(linear->bias);
        }
        free(linear);
    }
}


void linear_grad_free(LinearGrad* grad) {
    if (grad) {
        if (grad->grad_weight) {
            tensor_free(grad->grad_weight);
        }
        if (grad->grad_bias) {
            tensor_free(grad->grad_bias);
        }
        free(grad);
    }
}


