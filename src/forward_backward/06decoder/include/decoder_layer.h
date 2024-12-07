#ifndef DECODER_LAYER_H
#define DECODER_LAYER_H

#include "tensor_type.h"
#include "02multiattention.h"
#include "feed_forward.h"
#include "layer_norm.h"

typedef struct DecoderLayer {
    MultiHeadAttention* self_attn;    // 自注意力层
    MultiHeadAttention* cross_attn;   // 交叉注意力层
    LayerNorm* norm1;                 // 第一个层归一化
    LayerNorm* norm2;                 // 第二个层归一化
    LayerNorm* norm3;                 // 第三个层归一化
    FeedForward* ff;                  // 前馈网络
    float dropout_prob;                // dropout概率
} DecoderLayer;

// 创建解码器层
DecoderLayer* decoder_layer_create(
    int num_heads,
    int model_dim,
    int ff_dim,
    float dropout_prob
);

// 前向传播
bool decoder_layer_forward(
    DecoderLayer* layer,
    Tensor* input,              // [batch_size, seq_len, model_dim]
    Tensor* encoder_output,     // [batch_size, enc_seq_len, model_dim]
    Tensor* output,             // [batch_size, seq_len, model_dim]
    AttentionMask* self_mask,   // 自注意力掩码
    AttentionMask* cross_mask  // 交叉注意力掩码
);

// 释放资源
void decoder_layer_free(DecoderLayer* layer);

#endif
