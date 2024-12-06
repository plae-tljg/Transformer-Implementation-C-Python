#include "tensor.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// 计算张量总元素数量
static size_t calculate_total_size(const int* shape, int num_dims) {
    size_t total = 1;
    for (int i = 0; i < num_dims; i++) {
        total *= shape[i];
    }
    return total;
}

// 创建空张量
Tensor* tensor_create(int* shape, int num_dims) {
    if (shape == NULL || num_dims <= 0) {
        fprintf(stderr, "Invalid shape or dimensions\n");
        return NULL;
    }

    Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
    if (!tensor) {
        fprintf(stderr, "Failed to allocate tensor structure\n");
        return NULL;
    }

    // 分配并复制shape数组
    tensor->shape = (int*)malloc(num_dims * sizeof(int));
    if (!tensor->shape) {
        fprintf(stderr, "Failed to allocate shape array\n");
        free(tensor);
        return NULL;
    }
    memcpy(tensor->shape, shape, num_dims * sizeof(int));
    tensor->num_dims = num_dims;

    // 计算并分配数据空间
    size_t total_size = calculate_total_size(shape, num_dims);
    tensor->data = (float*)calloc(total_size, sizeof(float));
    if (!tensor->data) {
        fprintf(stderr, "Failed to allocate tensor data\n");
        free(tensor->shape);
        free(tensor);
        return NULL;
    }

    return tensor;
}

// 用已有数据创建张量
Tensor* tensor_create_with_data(int* shape, int num_dims, float* data) {
    Tensor* tensor = tensor_create(shape, num_dims);
    if (!tensor) {
        return NULL;
    }

    size_t total_size = calculate_total_size(shape, num_dims);
    memcpy(tensor->data, data, total_size * sizeof(float));
    return tensor;
}

// 释放张量
void tensor_free(Tensor* tensor) {
    if (tensor) {
        free(tensor->data);
        free(tensor->shape);
        free(tensor);
    }
}