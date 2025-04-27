#ifndef PRODUCT_ATTENTION_BACKWARD_H
#define PRODUCT_ATTENTION_BACKWARD_H

#include "02multiattention.h"

bool scaled_dot_product_attention_backward(
    Tensor* grad_output,        // [batch_size, num_heads, seq_len_q, d_k]
    Tensor* Q,                  // [batch_size, num_heads, seq_len_q, d_k]
    Tensor* K,                  // [batch_size, num_heads, seq_len_k, d_k]
    Tensor* V,                  // [batch_size, num_heads, seq_len_v, d_v]
    Tensor* grad_Q,            // [batch_size, num_heads, seq_len_q, d_k]
    Tensor* grad_K,            // [batch_size, num_heads, seq_len_k, d_k]
    Tensor* grad_V,            // [batch_size, num_heads, seq_len_v, d_v]
    AttentionMask* mask,
    float scale
);

#endif