#include "relu_backward.h"

bool relu_backward(
    Tensor* grad_output,
    Tensor* input,        // 需要原始输入来判断哪些位置是负数
    Tensor* grad_input
) {
    if (!grad_output || !input || !grad_input) {
        return false;
    }

    int size = tensor_size(input);

    // ReLU的导数很简单:
    // 如果输入 > 0，导数为1
    // 如果输入 <= 0，导数为0
    for (int i = 0; i < size; i++) {
        grad_input->data[i] = input->data[i] > 0 ? grad_output->data[i] : 0.0f;
    }

    return true;
}