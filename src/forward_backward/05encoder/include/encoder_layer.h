#ifndef ENCODER_LAYER_H
#define ENCODER_LAYER_H

#include "tensor_type.h"
#include "02multiattention.h"
#include "layer_norm.h"
#include "feed_forward.h"

typedef struct EncoderLayer {
    MultiHeadAttention* self_attn;  // 自注意力层
    LayerNorm* norm1;               // 第一个层归一化
    FeedForward* ff;                // 前馈网络
    LayerNorm* norm2;               // 第二个层归一化
    float dropout_prob;             // dropout概率
} EncoderLayer;

// 创建编码器层
EncoderLayer* encoder_layer_create(
    int num_heads,
    int model_dim,
    int ff_dim,
    float dropout_prob
);

// 前向传播
bool encoder_layer_forward(
    EncoderLayer* layer,
    Tensor* input,           // [batch_size, seq_len, model_dim]
    Tensor* output,          // [batch_size, seq_len, model_dim]
    AttentionMask* mask     // 注意力掩码
);

// 释放资源
void encoder_layer_free(EncoderLayer* layer);

#endif
