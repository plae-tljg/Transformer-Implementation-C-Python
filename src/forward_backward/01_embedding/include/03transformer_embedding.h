#ifndef TRANSFORMER_EMBEDDING_H
#define TRANSFORMER_EMBEDDING_H

#include "01token_embedding.h"
#include "02positional_embedding.h"
#include <stdbool.h>

typedef struct TransformerEmbedding TransformerEmbedding;   // to integrate both embedding and positional encoding

// Transformer嵌入结构
struct TransformerEmbedding {
    TokenEmbedding* token_embedding;          // Token嵌入层
    PositionalEncoding* positional_encoding;  // 位置编码层
};

TransformerEmbedding* transformer_embedding_create(int vocab_size, int embedding_dim, int max_seq_length);
void free_transformer_embedding(TransformerEmbedding* transformer_embedding);

bool transformer_embedding_forward(const TransformerEmbedding* trans_emb, const Tensor* tokens, Tensor* output);

#endif // TRANSFORMER_EMBEDDING_H