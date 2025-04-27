#include "tensor_expand.h"
#include <string.h>

bool tensor_broadcast_2d_to_4d(
    const Tensor* input,
    int batch_size,
    int num_heads,
    Tensor* output
) {
    if (!input || !output || input->num_dims != 2) return false;

    const int M = input->shape[0];
    const int N = input->shape[1];
    const int input_size = M * N;

    // 验证输出维度
    if (output->num_dims != 4 ||
        output->shape[0] != batch_size ||
        output->shape[1] != num_heads ||
        output->shape[2] != M ||
        output->shape[3] != N) {
        return false;
    }

    // 计算步长
    const int head_stride = M * N;
    const int batch_stride = num_heads * head_stride;

    // 执行广播
    #pragma omp parallel for collapse(2) if (batch_size * num_heads > 4)
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < num_heads; h++) {
            float* target = output->data + b * batch_stride + h * head_stride;
            memcpy(target, input->data, input_size * sizeof(float));
        }
    }

    return true;
}

// 通用广播实现（可以根据需要扩展）
bool tensor_broadcast(
    const Tensor* input,
    const int* target_shape,
    int target_dims,
    Tensor* output
) {
    // TODO: 实现更通用的广播逻辑
    // 这里可以根据实际需求实现更复杂的广播规则
    return false;
}
