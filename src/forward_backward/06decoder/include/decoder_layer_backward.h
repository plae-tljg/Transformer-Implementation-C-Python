#ifndef DECODER_LAYER_BACKWARD_H
#define DECODER_LAYER_BACKWARD_H

#include "decoder_layer.h"

bool decoder_layer_backward(
    DecoderLayer* layer,
    Tensor* grad_output,           // 输出梯度
    Tensor* input,                 // 原始输入
    Tensor* encoder_output,        // 编码器输出
    Tensor* grad_encoder_output,   // 编码器输出的梯度
    Tensor* grad_input,            // 输入梯度
    AttentionMask* self_mask,
    AttentionMask* cross_mask
);

#endif