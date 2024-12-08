#ifndef TRANSFORMER_BACKWARD_H
#define TRANSFORMER_BACKWARD_H

#include "transformer.h"
#include "decoder_backward.h"
#include "encoder_backward.h"

bool transformer_backward(
    Transformer* transformer,
    Tensor* grad_output,
    Tensor* grad_encoder_input,
    Tensor* grad_decoder_input,
    AttentionMask* enc_mask,
    AttentionMask* dec_mask,
    AttentionMask* cross_mask
);

#endif