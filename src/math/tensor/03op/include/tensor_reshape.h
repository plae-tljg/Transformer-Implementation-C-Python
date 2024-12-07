#ifndef TENSOR_RESHAPE_H
#define TENSOR_RESHAPE_H

#include "tensor_type.h"


bool tensor_reshape_3d_to_4d(const Tensor* input, int num_heads, Tensor* output);
bool tensor_reshape_4d_to_3d(const Tensor* input, Tensor* output);

#endif // TENSOR_RESHAPE_H
