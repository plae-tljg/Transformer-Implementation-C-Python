#include "tensor_add.h"
#include <stdio.h>

// 张量加法操作
bool tensor_add(const Tensor* A, const Tensor* B, Tensor* output) {
    // 检查输入是否有效
    if (!A || !B || !output) {
        fprintf(stderr, "Null tensor pointer(s)\n");
        return false;
    }

    // 检查形状是否匹配
    if (!check_same_shape(A, B) || !check_same_shape(A, output)) {
        fprintf(stderr, "Tensor shapes do not match\n");
        return false;
    }

    // 执行加法运算
    size_t total_size = calculate_total_size(A->shape, A->num_dims);
    for (size_t i = 0; i < total_size; i++) {
        output->data[i] = A->data[i] + B->data[i];
    }

    return true;
}

// 将偏置添加到3D张量
// input: [batch_size, seq_len, model_dim]
// bias: [model_dim]
// output: [batch_size, seq_len, model_dim]
bool tensor_add_bias_3d(
    const Tensor* input,
    const Tensor* bias,
    Tensor* output
) {

    int batch_size = input->shape[0];
    int seq_len = input->shape[1];
    int model_dim = input->shape[2];

    // 将输入复制到输出
    memcpy(output->data, input->data, batch_size * seq_len * model_dim * sizeof(float));

    // 为每个位置添加偏置
    // optimize later for gpu and blas
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_len; s++) {
            for (int d = 0; d < model_dim; d++) {
                // output[b,s,d] += bias[d]
                int idx = (b * seq_len * model_dim) + (s * model_dim) + d;
                output->data[idx] += bias->data[d];
            }
        }
    }

    return true;
}