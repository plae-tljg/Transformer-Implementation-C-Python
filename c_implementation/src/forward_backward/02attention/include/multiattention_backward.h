#ifndef MULTIATTENTION_BACKWARD_H
#define MULTIATTENTION_BACKWARD_H

#include "multiattention.h"

bool multihead_attention_backward(
    MultiHeadAttention* mha,
    Tensor* grad_output,      // [batch_size, seq_len, model_dim]
    Tensor* input,            // [batch_size, seq_len, model_dim]
    Tensor* grad_input,       // [batch_size, seq_len, model_dim]
    AttentionMask* mask
);

#endif