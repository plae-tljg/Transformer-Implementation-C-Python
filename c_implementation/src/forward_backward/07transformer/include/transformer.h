#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include "encoder.h"
#include "decoder.h"
#include "tensor_type.h"
#include "attention_mask.h"

typedef struct Transformer {
    Encoder* encoder;
    Decoder* decoder;
    int model_dim;
    int num_heads;
    int num_layers;
    int ff_dim;
    float dropout_prob;
} Transformer;

// 创建transformer
Transformer* transformer_create(
    int num_layers,
    int num_heads,
    int model_dim,
    int ff_dim,
    float dropout_prob
);

// 前向传播
bool transformer_forward(
    Transformer* transformer,
    Tensor* encoder_input,     // [batch_size, enc_seq_len, model_dim]
    Tensor* decoder_input,     // [batch_size, dec_seq_len, model_dim]
    Tensor* output,            // [batch_size, dec_seq_len, model_dim]
    AttentionMask* enc_mask,   // encoder的自注意力掩码
    AttentionMask* dec_mask,   // decoder的自注意力掩码
    AttentionMask* cross_mask  // decoder的交叉注意力掩码
);

// 释放资源
void transformer_free(Transformer* transformer);

#endif