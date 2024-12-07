#ifndef TENSOR_TYPE_H
#define TENSOR_TYPE_H

#include <stdbool.h>
#include <stddef.h>

typedef struct Tensor Tensor;

struct Tensor {
    float* data;    // 数据指针
    int* shape;     // 维度数组
    int num_dims;   // 维度数量
};

size_t calculate_total_size(const int* shape, int num_dims);
bool check_same_shape(const Tensor* A, const Tensor* B);
Tensor* tensor_create(int* shape, int num_dims);
void tensor_free(Tensor* tensor);
bool tensor_copy(Tensor* dst, const Tensor* src);
#endif // TENSOR_TYPE_H
