#ifndef TENSOR_ADD_H
#define TENSOR_ADD_H

#include "tensor.h"

bool tensor_add(const Tensor* A, const Tensor* B, Tensor* C);
bool tensor_add_inplace(Tensor* A, const Tensor* B);

#endif
