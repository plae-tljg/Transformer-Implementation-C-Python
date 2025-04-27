#include "decoder_backward.h"

bool decoder_backward(
    Decoder* decoder,
    Tensor* grad_output,           
    Tensor* encoder_output,        
    Tensor* grad_encoder_output,   
    Tensor* grad_input,            
    AttentionMask* self_mask,
    AttentionMask* cross_mask
) {
    if (!decoder || !grad_output || !encoder_output || 
        !grad_encoder_output || !grad_input) {
        return false;
    }

    // 创建临时张量存储每层的梯度
    Tensor* layer_grad = tensor_create(grad_output->shape, grad_output->num_dims);
    if (!layer_grad) return false;

    // 初始化编码器输出的梯度为0
    tensor_fill(grad_encoder_output, 0.0f);

    // 从最后一层开始反向传播
    tensor_copy(grad_output, layer_grad);
    
    // 从最后一层向第一层反向传播
    for (int i = decoder->num_layers - 1; i >= 0; i--) {
        Tensor* temp_grad_encoder = tensor_create(grad_encoder_output->shape, 
                                                grad_encoder_output->num_dims);
        if (!temp_grad_encoder) {
            tensor_free(layer_grad);
            return false;
        }

        if (!decoder_layer_backward(
                decoder->layers[i],
                layer_grad,
                (i == 0) ? grad_input : decoder->layer_outputs[i-1],
                encoder_output,
                temp_grad_encoder,
                (i == 0) ? grad_input : decoder->layer_outputs[i-1],
                self_mask,
                cross_mask)) {
            tensor_free(layer_grad);
            tensor_free(temp_grad_encoder);
            return false;
        }

        // 累积编码器输出的梯度
        if (!tensor_add(grad_encoder_output, temp_grad_encoder, grad_encoder_output)) {
            tensor_free(layer_grad);
            tensor_free(temp_grad_encoder);
            return false;
        }

        tensor_free(temp_grad_encoder);
    }

    tensor_free(layer_grad);
    return true;
}