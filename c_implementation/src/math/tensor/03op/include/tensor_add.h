#ifndef TENSOR_ADD_H
#define TENSOR_ADD_H

#include "tensor_type.h"

bool tensor_add(const Tensor* A, const Tensor* B, Tensor* output);
bool tensor_add_bias_3d(const Tensor* input, const Tensor* bias, Tensor* output);

#endif // TENSOR_ADD_H