#include "03transformer_embedding_backward.h"

bool transformer_embedding_backward(
    TransformerEmbedding* trans_emb,
    const Tensor* tokens,
    const Tensor* grad_output
) {
    if (!trans_emb || !tokens || !grad_output) {
        return false;
    }

    // 位置编码是固定的,不需要反向传播
    // 直接将梯度传给token embedding
    return token_embedding_backward(trans_emb->token_embedding, tokens, grad_output);
}