#ifndef POSITIONAL_ENCODING_H
#define POSITIONAL_ENCODING_H

#include "tensor_type.h"
#include <stdbool.h>

typedef struct PositionalEncoding PositionalEncoding;

// 位置编码结构
struct PositionalEncoding {
    Tensor* encodings;          // 位置编码矩阵 [batch_size, max_seq_length, encoding_dim]
    int max_seq_length;         // 最大序列长度, for sentence
    int encoding_dim;           // 编码维度, d_model, number of features to represent a position, same as embedding_dim
    // no need to store requires_grad, because it's a fixed value
};

PositionalEncoding* positional_encoding_create(int max_seq_length, int encoding_dim);
void free_positional_encoding(PositionalEncoding* pos_enc);

bool positional_encoding_forward(const PositionalEncoding* pos_enc, Tensor* input);

#endif // POSITIONAL_ENCODING_H