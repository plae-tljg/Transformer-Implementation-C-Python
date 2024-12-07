#ifndef TENSOR_LOGIC_H
#define TENSOR_LOGIC_H

#include "tensor_type.h"
#include <stdbool.h>

// 掩码应用操作
bool tensor_apply_mask(
    const Tensor* input,      // 输入张量
    const Tensor* mask,       // 掩码张量
    Tensor* output,           // 输出张量
    float mask_value          // 掩码值（用于替换被掩码的位置）
);

// 在训练时执行dropout操作
// prob: dropout概率(0-1之间)
// training: 是否处于训练模式
bool dropout_forward(Tensor* input, Tensor* output, float prob);

// dropout的反向传播
bool dropout_backward(Tensor* grad_output, Tensor* grad_input, float prob);

// 张量逐元素与操作
bool tensor_and(Tensor* a, Tensor* b, Tensor* output);

#endif // TENSOR_LOGIC_H
