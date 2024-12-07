#include "tensor_type.h"
#include <stdio.h>
#include <stdbool.h>

size_t calculate_total_size(const int* shape, int num_dims) {
    size_t total = 1;
    for (int i = 0; i < num_dims; i++) {
        total *= shape[i];
    }
    return total;
}

bool check_same_shape(const Tensor* A, const Tensor* B) {
    if (A->num_dims != B->num_dims) {
        return false;
    }

    for (int i = 0; i < A->num_dims; i++) {
        if (A->shape[i] != B->shape[i]) {
            return false;
        }
    }
    return true;
}

// 创建空张量
Tensor* tensor_create(int* shape, int num_dims) {
    if (!shape || num_dims <= 0) {
        fprintf(stderr, "Invalid shape or dimensions\n");
        return false;
    }

    Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
    if (!tensor) {
        return false;
    }

    // 分配并复制shape数组
    tensor->shape = (int*)malloc(num_dims * sizeof(int));
    if (!tensor->shape) {
        free(tensor);
        return false;
    }
    memcpy(tensor->shape, shape, num_dims * sizeof(int));
    tensor->num_dims = num_dims;

    // 计算并分配数据空间
    size_t total_size = calculate_total_size(shape, num_dims);
    tensor->data = (float*)calloc(total_size, sizeof(float));
    if (!tensor->data) {
        free(tensor->shape);
        free(tensor);
        return false;
    }

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

// 复制张量
bool tensor_copy(Tensor* dst, const Tensor* src) {
    // 检查输入参数
    if (!dst || !src) {
        return false;
    }


    // 计算总大小
    size_t total_size = calculate_total_size(src->shape, src->num_dims);

    // 复制数据
    memcpy(dst->data, src->data, total_size * sizeof(float));

    return true;
}
