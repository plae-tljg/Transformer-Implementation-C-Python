#include "01token_embedding_backward.h"

bool token_embedding_backward(
    TokenEmbedding* embedding,
    const Tensor* tokens,
    const Tensor* grad_output
) {
    if (!embedding || !tokens || !grad_output) {
        return false;
    }

    int batch_size = tokens->shape[0];
    int seq_length = tokens->shape[1];
    int embedding_dim = embedding->embedding_dim;

    // 检查维度
    if (grad_output->num_dims != 3 ||
        grad_output->shape[0] != batch_size ||
        grad_output->shape[1] != seq_length ||
        grad_output->shape[2] != embedding_dim) {
        fprintf(stderr, "Invalid grad_output dimensions in token_embedding_backward\n");
        return false;
    }

    // 创建梯度累积张量（如果还没有）
    if (!embedding->grad_embedding) {
        int shape[] = {batch_size, embedding->vocab_size, embedding_dim};
        embedding->grad_embedding = tensor_create(shape, 3);
        if (!embedding->grad_embedding) {
            return false;
        }
        tensor_fill(embedding->grad_embedding, 0.0f); // 初始化为0
    }

    // 对每个batch和序列位置
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_length; s++) {
            // 获取当前位置的token id
            int token_id = (int)tokens->data[b * seq_length + s];
            
            // 检查token_id是否有效
            if (token_id < 0 || token_id >= embedding->vocab_size) {
                fprintf(stderr, "Invalid token_id in token_embedding_backward\n");
                return false;
            }

            // 累积梯度到对应的embedding向量
            for (int d = 0; d < embedding_dim; d++) {
                int grad_idx = (b * seq_length * embedding_dim) + 
                             (s * embedding_dim) + d;
                int emb_idx = (b * embedding->vocab_size * embedding_dim) + 
                             (token_id * embedding_dim) + d;
                
                embedding->grad_embedding->data[emb_idx] += grad_output->data[grad_idx];
            }
        }
    }

    return true;
}