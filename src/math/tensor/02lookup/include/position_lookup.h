#ifndef POSITION_LOOKUP_H
#define POSITION_LOOKUP_H

#include <stdbool.h>
#include "tensor_type.h"

bool compute_positional_encodings(
    Tensor* encodings,
    int batch_size,
    int max_seq_length,
    int encoding_dim
);

#endif // POSITION_LOOKUP_H
