#include "feed_forward_backward.h"
#include "relu_backward.h"
#include "linear_backward.h"

bool feed_forward_backward(
    FeedForward* ff,
    Tensor* grad_output,
    Tensor* input,
    Tensor* grad_input
) {
    if (!ff || !grad_output || !input || !grad_input) {
        return false;
    }

    // 创建临时张量存储中间梯度
    Tensor* hidden_grad = tensor_create(ff->hidden->shape, ff->hidden->num_dims);
    if (!hidden_grad) return false;

    // 1. 输出层的反向传播
    if (!linear_backward(ff->output_linear, grad_output, hidden_grad)) {
        tensor_free(hidden_grad);
        return false;
    }

    // 2. GELU激活函数的反向传播
    if (!relu_backward(hidden_grad, ff->hidden, hidden_grad)) {
        tensor_free(hidden_grad);
        return false;
    }

    // 3. 输入层的反向传播
    if (!linear_backward(ff->input_linear, hidden_grad, grad_input)) {
        tensor_free(hidden_grad);
        return false;
    }

    tensor_free(hidden_grad);
    return true;
}