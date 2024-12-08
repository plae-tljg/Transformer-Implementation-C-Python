// 嵌入层实现, code to implement the embedding layers, 
// first by implementing the token embedding layer, then the positional encoding layer, 
// and finally the transformer embedding layer by adding their outputs together
#include "03transformer_embedding.h"
#include "model_config.h"
#include <stdio.h>

TransformerEmbedding* transformer_embedding_create(
    int vocab_size, 
    int embedding_dim, 
    int max_seq_length
) {
    TransformerEmbedding* trans_emb = (TransformerEmbedding*)malloc(sizeof(TransformerEmbedding));
    if (!trans_emb) {
        return NULL;
    }

    // 创建token嵌入
    trans_emb->token_embedding = token_embedding_create(vocab_size, embedding_dim);
    if (!trans_emb->token_embedding) {
        free(trans_emb);
        return NULL;
    }

    // 创建位置编码
    trans_emb->positional_encoding = positional_encoding_create(max_seq_length, embedding_dim);
    if (!trans_emb->positional_encoding) {
        free_token_embedding(trans_emb->token_embedding);
        free(trans_emb);
        return NULL;
    }

    return trans_emb;
}

void free_transformer_embedding(TransformerEmbedding* trans_emb) {
    if (trans_emb) {
        free_token_embedding(trans_emb->token_embedding);
        free_positional_encoding(trans_emb->positional_encoding);
        free(trans_emb);
    }
}

// transformer_embedding forward pass
// input tokens shape: [batch_size, seq_length]
// output tensor shape: [batch_size, seq_length, embedding_dim]
bool transformer_embedding_forward(
    const TransformerEmbedding* trans_emb,
    const Tensor* tokens,
    Tensor* output
) {
    // 执行token嵌入
    if (!token_embedding_forward(trans_emb->token_embedding, tokens, output)) {
        return false;
    }
    
    // 添加位置编码
    if (!positional_encoding_forward(trans_emb->positional_encoding, output)) {
        return false;
    }

    // dropout for training
    
    return true;
}