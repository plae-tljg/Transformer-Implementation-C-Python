#ifndef ENCODER_LAYER_BACKWARD_H
#define ENCODER_LAYER_BACKWARD_H

#include "encoder_layer.h"

bool encoder_layer_backward(
    EncoderLayer* layer,
    Tensor* grad_output,
    Tensor* input,
    Tensor* grad_input,
    AttentionMask* mask
);

#endif