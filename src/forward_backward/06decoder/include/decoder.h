#ifndef DECODER_H
#define DECODER_H

#include "decoder_layer.h"

typedef struct Decoder {
    int num_layers;           // 解码器层数量
    DecoderLayer** layers;    // 解码器层数组
} Decoder;

// 创建解码器
Decoder* decoder_create(
    int num_layers,
    int num_heads,
    int model_dim,
    int ff_dim,
    float dropout_prob
);

// 前向传播
bool decoder_forward(
    Decoder* decoder,
    Tensor* input,              // [batch_size, seq_len, model_dim]
    Tensor* encoder_output,     // [batch_size, enc_seq_len, model_dim]
    Tensor* output,             // [batch_size, seq_len, model_dim]
    AttentionMask* self_mask,   // 自注意力掩码
    AttentionMask* cross_mask  // 交叉注意力掩码
);

// 释放资源
void decoder_free(Decoder* decoder);

#endif
