#ifndef EMBEDDINGS_H
#define EMBEDDINGS_H

#include "types.h"

// 位置编码函数
PositionalEncoding* create_positional_encoding(int max_seq_length, int encoding_dim);
void initialize_positional_encoding(PositionalEncoding* pos_enc);
float* add_positional_encoding(PositionalEncoding* pos_enc, float* embeddings, 
                             int seq_length);
void free_positional_encoding(PositionalEncoding* pos_enc);

// Token嵌入函数
float* compute_embeddings(TokenEmbedding* embedding, int* tokens, int seq_length);
void update_embedding_gradients(TokenEmbedding* embedding, float* gradients);

#endif // EMBEDDINGS_H 