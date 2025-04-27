#include "decoder_layer_backward.h"
#include "layer_norm_backward.h"
#include "multiattention_backward.h"
#include "cross_attention_backward.h"
#include "feed_forward_backward.h"


bool decoder_layer_backward(
    DecoderLayer* layer,
    Tensor* grad_output,           // 输出梯度
    Tensor* input,                 // 原始输入
    Tensor* encoder_output,        // 编码器输出
    Tensor* grad_encoder_output,   // 编码器输出的梯度
    Tensor* grad_input,            // 输入梯度
    AttentionMask* self_mask,
    AttentionMask* cross_mask
) {
    if (!layer || !grad_output || !input || !encoder_output || 
        !grad_encoder_output || !grad_input) {
        return false;
    }

    // 为中间梯度创建临时张量
    Tensor* temp_grad = tensor_create(grad_output->shape, grad_output->num_dims);
    Tensor* residual_grad = tensor_create(grad_output->shape, grad_output->num_dims);
    if (!temp_grad || !residual_grad) {
        tensor_free(temp_grad);
        tensor_free(residual_grad);
        return false;
    }

    // 保存前向传播的中间结果
    Tensor* self_attn_output = tensor_create(input->shape, input->num_dims);
    Tensor* cross_attn_output = tensor_create(input->shape, input->num_dims);
    if (!self_attn_output || !cross_attn_output) {
        tensor_free(temp_grad);
        tensor_free(residual_grad);
        tensor_free(self_attn_output);
        tensor_free(cross_attn_output);
        return false;
    }

    // 1. 前馈网络层的反向传播
    // 首先计算层归一化的反向传播
    if (!layer_norm_backward(layer->norm3, grad_output, temp_grad)) {
        goto cleanup;
    }

    // 分离残差连接的梯度
    tensor_copy(temp_grad, residual_grad);  // 保存一份用于残差连接

    // 前馈网络的反向传播
    if (!feed_forward_backward(layer->ff, temp_grad, cross_attn_output, temp_grad)) {
        goto cleanup;
    }

    // 应用dropout的反向传播
    if (!dropout_backward(temp_grad, temp_grad, layer->dropout_prob)) {
        goto cleanup;
    }

    // 添加残差连接的梯度
    if (!tensor_add(temp_grad, residual_grad, temp_grad)) {
        goto cleanup;
    }

    // 2. 交叉注意力层的反向传播
    if (!layer_norm_backward(layer->norm2, temp_grad, temp_grad)) {
        goto cleanup;
    }

    tensor_copy(temp_grad, residual_grad);  // 保存残差梯度

    // 交叉注意力的反向传播
    if (!cross_attention_backward(
            layer->cross_attn,
            temp_grad,              // 输出梯度
            self_attn_output,       // Q的输入
            encoder_output,         // K和V的输入
            temp_grad,              // Q的梯度
            grad_encoder_output,    // K和V的梯度
            cross_mask)) {
        goto cleanup;
    }

    if (!dropout_backward(temp_grad, temp_grad, layer->dropout_prob)) {
        goto cleanup;
    }

    // 添加残差连接的梯度
    if (!tensor_add(temp_grad, residual_grad, temp_grad)) {
        goto cleanup;
    }

    // 3. 自注意力层的反向传播
    if (!layer_norm_backward(layer->norm1, temp_grad, temp_grad)) {
        goto cleanup;
    }

    tensor_copy(temp_grad, residual_grad);  // 保存残差梯度

    // 自注意力的反向传播
    if (!multihead_attention_backward(
            layer->self_attn,
            temp_grad,      // 输出梯度
            input,          // 原始输入
            grad_input,     // 输入梯度
            self_mask)) {
        goto cleanup;
    }

    if (!dropout_backward(grad_input, grad_input, layer->dropout_prob)) {
        goto cleanup;
    }

    // 添加最后的残差连接梯度
    if (!tensor_add(grad_input, residual_grad, grad_input)) {
        goto cleanup;
    }

    // 清理并返回
    tensor_free(temp_grad);
    tensor_free(residual_grad);
    tensor_free(self_attn_output);
    tensor_free(cross_attn_output);
    return true;

cleanup:
    tensor_free(temp_grad);
    tensor_free(residual_grad);
    tensor_free(self_attn_output);
    tensor_free(cross_attn_output);
    return false;
}