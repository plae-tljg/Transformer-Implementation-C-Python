#include "03multiattention_backward.h"

bool multihead_attention_backward(
    MultiHeadAttention* mha,
    Tensor* grad_output,      // [batch_size, seq_len, model_dim]
    Tensor* input,            // [batch_size, seq_len, model_dim]
    Tensor* grad_input,       // [batch_size, seq_len, model_dim]
    AttentionMask* mask
) {
    if (!mha || !grad_output || !input || !grad_input) {
        return false;
    }

    // 创建临时张量
    Tensor* grad_q = tensor_create(input->shape, input->num_dims);
    Tensor* grad_k = tensor_create(input->shape, input->num_dims);
    Tensor* grad_v = tensor_create(input->shape, input->num_dims);
    if (!grad_q || !grad_k || !grad_v) {
        goto cleanup;
    }

    // 1. 输出投影的反向传播
    Tensor* grad_multihead = tensor_create(grad_output->shape, grad_output->num_dims);
    if (!linear_backward(mha->output_proj, grad_output, grad_multihead)) {
        goto cleanup;
    }

    // 2. 注意力计算的反向传播
    if (!scaled_dot_product_attention_backward(
            grad_multihead,
            mha->q_proj_output,
            mha->k_proj_output,
            mha->v_proj_output,
            grad_q,
            grad_k,
            grad_v,
            mask,
            mha->scale)) {
        goto cleanup;
    }

    // 3. Q、K、V投影的反向传播
    if (!linear_backward(mha->q_proj, grad_q, grad_input)) {
        goto cleanup;
    }
    if (!linear_backward(mha->k_proj, grad_k, grad_input)) {
        goto cleanup;
    }
    if (!linear_backward(mha->v_proj, grad_v, grad_input)) {
        goto cleanup;
    }

    tensor_free(grad_q);
    tensor_free(grad_k);
    tensor_free(grad_v);
    tensor_free(grad_multihead);
    return true;

cleanup:
    tensor_free(grad_q);
    tensor_free(grad_k);
    tensor_free(grad_v);
    if (grad_multihead) tensor_free(grad_multihead);
    return false;
}