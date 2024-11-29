#ifndef UTILS_H
#define UTILS_H

#include "types.h"

// 矩阵操作
float* matrix_multiply(float* a, float* b, int m, int k, int n);
float* matrix_multiply_transpose(float* a, float* b, int m, int k, int n);
float* matrix_multiply_transpose2(float* a, float* b, int m, int k, int n);
void matrix_add_inplace(float* a, float* b, int size);
void matrix_scale(float* matrix, float scale, int size);

// 其他工具函数
void apply_dropout(float* x, float dropout_prob, int size);
void apply_layer_norm(float* x, float* gamma, float* beta, int size);

// 激活函数
void softmax(float* x, int size);

#endif // UTILS_H 