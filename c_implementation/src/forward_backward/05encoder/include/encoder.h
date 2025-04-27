#ifndef ENCODER_H
#define ENCODER_H

#include "encoder_layer.h"

typedef struct Encoder {
    int num_layers;           // 编码器层数量
    EncoderLayer** layers;    // 编码器层数组
} Encoder;

// 创建编码器
Encoder* encoder_create(
    int num_layers,
    int num_heads,
    int model_dim,
    int ff_dim,
    float dropout_prob
);

// 前向传播
bool encoder_forward(
    Encoder* encoder,
    Tensor* input,           // [batch_size, seq_len, model_dim]
    Tensor* output,          // [batch_size, seq_len, model_dim]
    AttentionMask* mask     // 注意力掩码
);

// 释放资源
void encoder_free(Encoder* encoder);

#endif
