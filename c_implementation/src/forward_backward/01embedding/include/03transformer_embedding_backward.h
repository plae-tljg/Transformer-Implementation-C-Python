#ifndef TRANSFORMER_EMBEDDING_BACKWARD_H
#define TRANSFORMER_EMBEDDING_BACKWARD_H

#include "03transformer_embedding.h"

bool transformer_embedding_backward(
    TransformerEmbedding* trans_emb,
    const Tensor* tokens,
    const Tensor* grad_output
);

#endif