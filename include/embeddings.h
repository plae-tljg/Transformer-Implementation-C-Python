#ifndef EMBEDDINGS_H
#define EMBEDDINGS_H


#include <stdbool.h>
typedef struct PositionalEncoding PositionalEncoding;
typedef struct TokenEmbedding TokenEmbedding;

// 位置编码结构
struct PositionalEncoding {
    float* encodings;          // 位置编码矩阵, [max_seq_length, encoding_dim]
    int max_seq_length;        // 最大序列长度, for sentence
    int encoding_dim;          // 编码维度, d_model
};

// Token嵌入结构
struct TokenEmbedding {
    float* embedding_matrix;    // 嵌入矩阵, [vocab_size, embedding_dim]
    int vocab_size;            // 词汇表大小
    int embedding_dim;         // 嵌入维度, d_model
    bool requires_grad;        // 是否需要梯度
};

// 创建位置编码结构
PositionalEncoding* positional_encoding_create(int max_seq_length, int encoding_dim);

// 创建Token嵌入结构
TokenEmbedding* token_embedding_create(int vocab_size, int embedding_dim, bool requires_grad);

// 释放位置编码结构内存
void free_positional_encoding(PositionalEncoding* pos_enc);

// 释放Token嵌入结构内存
void free_token_embedding(TokenEmbedding* token_emb);

// Token嵌入前向传播
void token_embedding_forward(TokenEmbedding* embedding, int* tokens, float* output);

// 位置编码前向传播
void positional_encoding_forward(PositionalEncoding* pos_enc, int seq_length, float* output);


#endif

