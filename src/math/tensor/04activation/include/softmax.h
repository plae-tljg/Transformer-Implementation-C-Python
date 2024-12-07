#ifndef SOFTMAX_H
#define SOFTMAX_H

#include "tensor_type.h"

// 在指定维度上执行softmax
// axis: 执行softmax的维度
// bool tensor_softmax(const Tensor* input, int axis, Tensor* output);

// 针对4D注意力分数的特殊softmax实现
// input: [batch_size, num_heads, seq_len, seq_len]
bool attention_scores_softmax(const Tensor* input, Tensor* output);

#endif // SOFTMAX_H
