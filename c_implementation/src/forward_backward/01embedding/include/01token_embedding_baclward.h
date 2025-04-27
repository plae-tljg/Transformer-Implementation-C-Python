#ifndef TOKEN_EMBEDDING_BACKWARD_H
#define TOKEN_EMBEDDING_BACKWARD_H

#include "01token_embedding.h"

bool token_embedding_backward(
    TokenEmbedding* embedding,
    const Tensor* tokens,     // [batch_size, seq_length]
    const Tensor* grad_output // [batch_size, seq_length, embedding_dim]
);

#endif