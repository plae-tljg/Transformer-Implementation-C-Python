#include "transformer_backward.h"

bool transformer_backward(
    Transformer* transformer,
    Tensor* grad_output,
    Tensor* grad_encoder_input,
    Tensor* grad_decoder_input,
    AttentionMask* enc_mask,
    AttentionMask* dec_mask,
    AttentionMask* cross_mask
) {
    if (!transformer || !grad_output || !grad_encoder_input || !grad_decoder_input) {
        return false;
    }

    // 创建临时张量存储编码器输出的梯度
    Tensor* grad_encoder_output = tensor_create(grad_encoder_input->shape, grad_encoder_input->num_dims);
    if (!grad_encoder_output) return false;

    // 解码器反向传播
    if (!decoder_backward(transformer->decoder, grad_output, grad_encoder_output,
                         grad_decoder_input, dec_mask, cross_mask)) {
        tensor_free(grad_encoder_output);
        return false;
    }

    // 编码器反向传播
    if (!encoder_backward(transformer->encoder, grad_encoder_output,
                         grad_encoder_input, enc_mask)) {
        tensor_free(grad_encoder_output);
        return false;
    }

    tensor_free(grad_encoder_output);
    return true;
}