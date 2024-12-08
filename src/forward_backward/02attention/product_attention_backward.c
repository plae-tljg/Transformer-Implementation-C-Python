#include "03product_attention_backward.h"

bool scaled_dot_product_attention_backward(
    Tensor* grad_output,        // [batch_size, num_heads, seq_len_q, d_k]
    Tensor* Q,                  // [batch_size, num_heads, seq_len_q, d_k]
    Tensor* K,                  // [batch_size, num_heads, seq_len_k, d_k]
    Tensor* V,                  // [batch_size, num_heads, seq_len_v, d_v]
    Tensor* grad_Q,            // [batch_size, num_heads, seq_len_q, d_k]
    Tensor* grad_K,            // [batch_size, num_heads, seq_len_k, d_k]
    Tensor* grad_V,            // [batch_size, num_heads, seq_len_v, d_v]
    AttentionMask* mask,
    float scale
) {
    if (!grad_output || !Q || !K || !V || !grad_Q || !grad_K || !grad_V) {
        return false;
    }

    int batch_size = Q->shape[0];
    int num_heads = Q->shape[1];
    int seq_len_q = Q->shape[2];
    int seq_len_k = K->shape[2];
    int d_k = Q->shape[3];

    // 创建临时张量
    Tensor* scores = tensor_create_4d(batch_size, num_heads, seq_len_q, seq_len_k);
    Tensor* softmax_grad = tensor_create_4d(batch_size, num_heads, seq_len_q, seq_len_k);
    if (!scores || !softmax_grad) {
        tensor_free(scores);
        tensor_free(softmax_grad);
        return false;
    }

    // 1. 计算注意力分数 QK^T/sqrt(d_k)
    if (!matrix_multiply(Q, K, scores, false, true)) {
        goto cleanup;
    }
    tensor_scale(scores, scale);

    // 2. 应用mask（如果有）
    if (mask) {
        apply_attention_mask(scores, mask);
    }

    // 3. 计算softmax的梯度
    compute_softmax_gradient(scores, grad_output, V, softmax_grad);

    // 4. 计算Q的梯度
    if (!matrix_multiply(softmax_grad, K, grad_Q, false, false)) {
        goto cleanup;
    }
    tensor_scale(grad_Q, scale);

    // 5. 计算K的梯度
    if (!matrix_multiply(softmax_grad, Q, grad_K, true, false)) {
        goto cleanup;
    }
    tensor_scale(grad_K, scale);

    // 6. 计算V的梯度
    if (!matrix_multiply(scores, grad_output, grad_V, true, false)) {
        goto cleanup;
    }

    tensor_free(scores);
    tensor_free(softmax_grad);
    return true;

cleanup:
    tensor_free(scores);
    tensor_free(softmax_grad);
    return false;
}