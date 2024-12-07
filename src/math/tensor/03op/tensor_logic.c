#include "tensor_logic.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
bool tensor_apply_mask(
    const Tensor* input,
    const Tensor* mask,
    Tensor* output,
    float mask_value
) {
    // 基本验证
    if (!input || !mask || !output) return false;
    if (calculate_total_size(input->shape, input->num_dims) != calculate_total_size(output->shape, output->num_dims)) return false;

    // 复制输入到输出
    memcpy(output->data, input->data, calculate_total_size(input->shape, input->num_dims) * sizeof(float));

    // 应用掩码
    for (int i = 0; i < calculate_total_size(output->shape, output->num_dims); i++) {
        if (mask->data[i] == 0.0f) {
            output->data[i] = mask_value;
        }
    }

    return true;
}

bool dropout_forward(Tensor* input, Tensor* output, float prob) {
    if (!input || !output) return false;
    
    size_t total_elements = calculate_total_size(input->shape, input->num_dims);
    
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
    
    return true;
}

bool dropout_backward(Tensor* grad_output, Tensor* grad_input, float prob) {
    if (!grad_output || !grad_input) return false;
    
    size_t total_elements = calculate_total_size(grad_output->shape, grad_output->num_dims);
    
    float scale = 1.0f / (1.0f - prob);
    
    for (int i = 0; i < total_elements; i++) {
        // 如果前向传播时该位置被dropout(为0),则梯度也为0
        grad_input->data[i] = grad_input->data[i] == 0 ? 0 : grad_output->data[i] * scale;
    }
    
    return true;
}

bool tensor_and(Tensor* a, Tensor* b, Tensor* output) {
    if (!a || !b || !output) {
        fprintf(stderr, "输入张量不能为空\n");
        return false;
    }

    // 检查维度是否匹配
    if (a->num_dims != b->num_dims || a->num_dims != output->num_dims) {
        fprintf(stderr, "张量维度不匹配\n");
        return false;
    }

    // 检查每个维度的大小是否匹配
    for (int i = 0; i < a->num_dims; i++) {
        if (a->shape[i] != b->shape[i] || a->shape[i] != output->shape[i]) {
            fprintf(stderr, "张量形状不匹配\n");
            return false;
        }
    }

    // 执行逐元素与操作
    size_t total_size = calculate_total_size(a->shape, a->num_dims);
    #pragma omp parallel for if(total_size > 1000)
    for (size_t i = 0; i < total_size; i++) {
        output->data[i] = (a->data[i] != 0.0f && b->data[i] != 0.0f) ? 1.0f : 0.0f;
    }

    return true;
}
