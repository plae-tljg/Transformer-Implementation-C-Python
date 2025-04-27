#ifndef RELU_BACKWARD_H
#define RELU_BACKWARD_H

#include "relu.h"

bool relu_backward(
    Tensor* grad_output,
    Tensor* input,        // 需要原始输入来判断哪些位置是负数
    Tensor* grad_input
);

#endif