#ifndef TENSOR_CHECK_H
#define TENSOR_CHECK_H

#include <stdbool.h>

bool check_matmul_dims(int A_dim, int B_dim, const int* A_shape, const int* B_shape, int* out_shape, int* out_ndims);

#endif
