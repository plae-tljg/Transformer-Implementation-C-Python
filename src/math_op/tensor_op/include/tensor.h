#ifndef TENSOR_H
#define TENSOR_H

#include <stdbool.h>

typedef struct Tensor Tensor;

struct Tensor {
    float* data;    // 数据指针
    int* shape;     // 维度数组
    int num_dims;   // 维度数量
};

// 创建和释放函数
Tensor* tensor_create(int* shape, int num_dims);
Tensor* tensor_create_with_data(int* shape, int num_dims, float* data);
void tensor_free(Tensor* tensor);

#endif
