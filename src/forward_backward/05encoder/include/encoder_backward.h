#ifndef ENCODER_BACKWARD_H
#define ENCODER_BACKWARD_H

#include "encoder.h"

bool encoder_backward(
    Encoder* encoder,
    Tensor* grad_output,
    Tensor* grad_input,
    AttentionMask* mask
);

#endif