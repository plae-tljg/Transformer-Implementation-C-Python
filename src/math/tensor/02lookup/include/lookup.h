#ifndef LOOKUP_H
#define LOOKUP_H

#include <stdbool.h>
#include "tensor_type.h"

bool perform_embedding_lookup(
    const Tensor* embedding_matrix,
    const Tensor* tokens,
    Tensor* output,
    int batch_size,
    int seq_length,
    int embedding_dim
);

#endif // LOOKUP_H

