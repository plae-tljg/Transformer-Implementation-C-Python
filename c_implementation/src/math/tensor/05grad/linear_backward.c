#include "linear_backward.h"

bool linear_backward(
    Linear* linear,
    Tensor* grad_output,    // [batch_size, *, out_features]
    Tensor* grad_input      // [batch_size, *, in_features]
) {
    if (!linear || !grad_output || !grad_input) {
        return false;
    }

    // 1. 计算输入的梯度: grad_input = grad_output * W^T
    if (!matrix_multiply(grad_output, linear->weight, grad_input, false, true)) {
        return false;
    }

    // 2. 计算权重的梯度: grad_W = X^T * grad_output
    if (!matrix_multiply(linear->input_cache, grad_output, linear->grad_weight, true, false)) {
        return false;
    }

    // 3. 计算偏置的梯度 (如果有偏置)
    if (linear->bias && linear->grad_bias) {
        // 沿着batch和序列长度维度求和
        if (!tensor_sum_dims(grad_output, linear->grad_bias, 0, -2)) {
            return false;
        }
    }

    return true;
}