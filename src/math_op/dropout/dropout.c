#include <stdlib.h>
#include <time.h>
#include "dropout.h"

// dropout_forward: 执行dropout的前向传播
// 参数:
//   input: 输入张量 [任意形状]
//   prob: dropout概率(0-1之间)
//   training: 是否处于训练模式, 测试阶段不进行dropout
// 返回:
//   output: 输出张量 [与input相同形状]
Tensor* dropout_forward(Tensor* input, float prob, bool training) {
    size_t total_elements = calculate_total_size(input->shape, input->num_dims);
    Tensor* output = tensor_create(input->shape, input->num_dims);
    
    if (!training || prob <= 0) {
        // 测试阶段或prob为0时,直接复制输入
        for (int i = 0; i < total_elements; i++) {
            output->data[i] = input->data[i];
        }
        return output;
    }

    // 训练阶段
    srand(time(NULL));
    float scale = 1.0f / (1.0f - prob); // 缩放因子

    for (int i = 0; i < total_elements; i++) {
        float random = (float)rand() / RAND_MAX;
        if (random > prob) {
            output->data[i] = input->data[i] * scale;
        } else {
            output->data[i] = 0;
        }
    }

    return output;
}

// dropout_backward: 执行dropout的反向传播
// 参数:
//   grad_output: 输出梯度 [任意形状]
//   input: 前向传播的输入 [与grad_output相同形状] 
//   prob: dropout概率(0-1之间)
// 返回:
//   grad_input: 输入梯度 [与input相同形状]
// 说明:
//   - 对于前向传播时被dropout(值为0)的位置,梯度也设为0
//   - 对于保留的位置,梯度需要乘以缩放因子scale
//   - scale = 1/(1-prob)用于保持期望值不变

Tensor* dropout_backward(Tensor* grad_output, Tensor* input, float prob) {
    size_t total_elements = calculate_total_size(grad_output->shape, grad_output->num_dims);
    Tensor* grad_input = tensor_create(grad_output->shape, grad_output->num_dims);
    
    float scale = 1.0f / (1.0f - prob);
    
    for (int i = 0; i < total_elements; i++) {
        // 如果前向传播时该位置被dropout(为0),则梯度也为0
        grad_input->data[i] = input->data[i] == 0 ? 0 : grad_output->data[i] * scale;
    }

    return grad_input;
}
