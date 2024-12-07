#ifndef TOKEN_EMBEDDING_H
#define TOKEN_EMBEDDING_H

#include "tensor_type.h"
#include <stdbool.h>
typedef struct TokenEmbedding TokenEmbedding;

// Token嵌入结构
struct TokenEmbedding {
    Tensor* embedding_matrix;    // 嵌入矩阵 [batch_size, vocab_size, embedding_dim]
    int vocab_size;             // 词汇表大小, number of unique tokens in the vocabulary dictionary
    int embedding_dim;          // 嵌入维度, d_model, number of features to represent a token, same as encoding_dim
    // no need to store requires_grad, because token embedding need to be trained
};

TokenEmbedding* token_embedding_create(int vocab_size, int embedding_dim);
void token_embedding_free(TokenEmbedding* token_embedding);

bool token_embedding_forward(const TokenEmbedding* embedding, const Tensor* tokens, Tensor* output);

#endif // TOKEN_EMBEDDING_H
