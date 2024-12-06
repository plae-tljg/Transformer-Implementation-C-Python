#ifndef EMBEDDINGS_H
#define EMBEDDINGS_H

#include <stdbool.h>
#include "tensor.h"
#include "model_config.h"
typedef struct PositionalEncoding PositionalEncoding;
typedef struct TokenEmbedding TokenEmbedding;
typedef struct TransformerEmbedding TransformerEmbedding;   // to integrate both embedding and positional encoding

// Token嵌入结构
struct TokenEmbedding {
    Tensor* embedding_matrix;    // 嵌入矩阵 [batch_size, vocab_size, embedding_dim]
    int vocab_size;             // 词汇表大小, number of unique tokens in the vocabulary dictionary
    int embedding_dim;          // 嵌入维度, d_model, number of features to represent a token, same as encoding_dim
    // no need to store requires_grad, because token embedding need to be trained
};

// 位置编码结构
struct PositionalEncoding {
    Tensor* encodings;          // 位置编码矩阵 [batch_size, max_seq_length, encoding_dim]
    int max_seq_length;         // 最大序列长度, for sentence
    int encoding_dim;           // 编码维度, d_model, number of features to represent a position, same as embedding_dim
    // no need to store requires_grad, because it's a fixed value
};

// Transformer嵌入结构
struct TransformerEmbedding {
    TokenEmbedding* token_embedding;          // Token嵌入层
    PositionalEncoding* positional_encoding;  // 位置编码层
};

// 创建函数
TokenEmbedding* token_embedding_create(int vocab_size, int embedding_dim);
PositionalEncoding* positional_encoding_create(int max_seq_length, int encoding_dim);
TransformerEmbedding* transformer_embedding_create(
    int vocab_size, 
    int embedding_dim, 
    int max_seq_length
);

// 前向传播函数
bool token_embedding_forward(const TokenEmbedding* embedding, const Tensor* tokens, Tensor* output);
bool positional_encoding_forward(const PositionalEncoding* pos_enc, Tensor* input);
bool transformer_embedding_forward(const TransformerEmbedding* trans_emb, 
                                const Tensor* tokens, Tensor* output);

// 内存管理函数
void free_token_embedding(TokenEmbedding* token_emb);
void free_positional_encoding(PositionalEncoding* pos_enc);
void free_transformer_embedding(TransformerEmbedding* trans_emb);

#endif

