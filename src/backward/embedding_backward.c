#include "training.h"
#include "grad.h"
#include "embeddings.h"
#include <stdlib.h>

void token_embedding_backward(
    TokenEmbedding* emb,
    float* grad_output,    // [batch_size, seq_len, embedding_dim]
    int* tokens,           // [batch_size, seq_len]
    int batch_size,
    int seq_len,
    TokenEmbeddingGrad* grad
) {
    // 只累积词嵌入的梯度
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_len; s++) {
            int token = tokens[b * seq_len + s];
            for (int d = 0; d < emb->embedding_dim; d++) {
                grad->grad_embedding[token * emb->embedding_dim + d] +=
                    grad_output[(b * seq_len + s) * emb->embedding_dim + d];
            }
        }
    }
}

// 释放词嵌入梯度
void free_token_embedding_grad(TokenEmbeddingGrad* grad) {
    if (grad) {
        if (grad->grad_embedding) {
            free(grad->grad_embedding);
        }
        free(grad);
    }
}