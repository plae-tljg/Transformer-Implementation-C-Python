#include "encoder_layer_backward.h"

bool encoder_layer_backward(
    EncoderLayer* layer,
    Tensor* grad_output,
    Tensor* input,
    Tensor* grad_input,
    AttentionMask* mask
) {
    if (!layer || !grad_output || !input || !grad_input) {
        return false;
    }

    // 创建临时张量
    Tensor* temp_grad = tensor_create(grad_output->shape, grad_output->num_dims);
    Tensor* residual_grad = tensor_create(grad_output->shape, grad_output->num_dims);
    if (!temp_grad || !residual_grad) {
        tensor_free(temp_grad);
        tensor_free(residual_grad);
        return false;
    }

    // 1. 前馈网络层的反向传播
    if (!layer_norm_backward(layer->norm2, grad_output, temp_grad)) {
        goto cleanup;
    }

    tensor_copy(temp_grad, residual_grad);

    if (!feed_forward_backward(layer->ff, temp_grad, input, temp_grad)) {
        goto cleanup;
    }

    if (!dropout_backward(temp_grad, temp_grad, layer->dropout_prob)) {
        goto cleanup;
    }

    if (!tensor_add(temp_grad, residual_grad, temp_grad)) {
        goto cleanup;
    }

    // 2. 自注意力层的反向传播
    if (!layer_norm_backward(layer->norm1, temp_grad, temp_grad)) {
        goto cleanup;
    }

    tensor_copy(temp_grad, residual_grad);

    if (!multihead_attention_backward(
            layer->self_attn,
            temp_grad,
            input,
            grad_input,
            mask)) {
        goto cleanup;
    }

    if (!dropout_backward(grad_input, grad_input, layer->dropout_prob)) {
        goto cleanup;
    }

    if (!tensor_add(grad_input, residual_grad, grad_input)) {
        goto cleanup;
    }

    tensor_free(temp_grad);
    tensor_free(residual_grad);
    return true;

cleanup:
    tensor_free(temp_grad);
    tensor_free(residual_grad);
    return false;
}