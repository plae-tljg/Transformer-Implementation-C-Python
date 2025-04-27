#include "encoder_backward.h"

bool encoder_backward(
    Encoder* encoder,
    Tensor* grad_output,
    Tensor* grad_input,
    AttentionMask* mask
) {
    if (!encoder || !grad_output || !grad_input) {
        return false;
    }

    // 创建临时张量存储每层的梯度
    Tensor* layer_grad = tensor_create(grad_output->shape, grad_output->num_dims);
    if (!layer_grad) return false;

    // 从最后一层开始反向传播
    tensor_copy(grad_output, layer_grad);

    for (int i = encoder->num_layers - 1; i >= 0; i--) {
        if (!encoder_layer_backward(
                encoder->layers[i],
                layer_grad,
                (i == 0) ? grad_input : encoder->layer_outputs[i-1],
                (i == 0) ? grad_input : encoder->layer_outputs[i-1],
                mask)) {
            tensor_free(layer_grad);
            return false;
        }
    }

    tensor_free(layer_grad);
    return true;
}