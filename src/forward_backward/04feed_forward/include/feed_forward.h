#ifndef FEED_FORWARD_H
#define FEED_FORWARD_H

#include "tensor_type.h"

typedef struct FeedForward FeedForward;

struct FeedForward {
    Tensor* w1;  // 第一个线性变换的权重
    Tensor* b1;  // 第一个线性变换的偏置
    Tensor* w2;  // 第二个线性变换的权重
    Tensor* b2;  // 第二个线性变换的偏置
};

// 创建前馈层
FeedForward* feed_forward_create(int input_dim, int hidden_dim);

// 前向传播
bool feed_forward_forward(FeedForward* ff, const Tensor* input, Tensor* output);

// 释放资源
void feed_forward_free(FeedForward* ff);

#endif
