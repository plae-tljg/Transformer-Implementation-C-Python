#ifndef DROPOUT_H
#define DROPOUT_H

#include "tensor.h"

// 在训练时执行dropout操作
// prob: dropout概率(0-1之间)
// training: 是否处于训练模式
Tensor* dropout_forward(Tensor* input, float prob, bool training);

// dropout的反向传播
Tensor* dropout_backward(Tensor* grad_output, Tensor* input, float prob);

#endif
