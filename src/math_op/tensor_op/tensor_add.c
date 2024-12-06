#include "tensor_add.h"
#include <stdio.h>
#include <stdbool.h>

// 检查两个张量的形状是否相同
static bool check_same_shape(const Tensor* A, const Tensor* B) {
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

// 计算张量总元素数量
static size_t calculate_total_size(const int* shape, int num_dims) {
    size_t total = 1;
    for (int i = 0; i < num_dims; i++) {
        total *= shape[i];
    }
    return total;
}

// 张量加法操作
bool tensor_add(const Tensor* A, const Tensor* B, Tensor* C) {
    // 检查输入是否有效
    if (!A || !B || !C) {
        fprintf(stderr, "Null tensor pointer(s)\n");
        return false;
    }

    // 检查形状是否匹配
    if (!check_same_shape(A, B) || !check_same_shape(A, C)) {
        fprintf(stderr, "Tensor shapes do not match\n");
        return false;
    }

    // 执行加法运算
    size_t total_size = calculate_total_size(A->shape, A->num_dims);
    for (size_t i = 0; i < total_size; i++) {
        C->data[i] = A->data[i] + B->data[i];
    }

    return true;
}

// 原地张量加法操作（A = A + B）
bool tensor_add_inplace(Tensor* A, const Tensor* B) {
    // 检查输入是否有效
    if (!A || !B) {
        fprintf(stderr, "Null tensor pointer(s)\n");
        return false;
    }

    // 检查形状是否匹配
    if (!check_same_shape(A, B)) {
        fprintf(stderr, "Tensor shapes do not match\n");
        return false;
    }

    // 执行加法运算
    size_t total_size = calculate_total_size(A->shape, A->num_dims);
    for (size_t i = 0; i < total_size; i++) {
        A->data[i] += B->data[i];
    }

    return true;
}
