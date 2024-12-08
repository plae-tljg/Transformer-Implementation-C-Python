#ifndef LAYER_NORM_BACKWARD_H
#define LAYER_NORM_BACKWARD_H

#include "layer_norm.h"

bool layer_norm_backward(
    LayerNorm* ln,
    Tensor* grad_output,    // [batch_size, seq_len, model_dim]
    Tensor* grad_input      // [batch_size, seq_len, model_dim]
);

#endif