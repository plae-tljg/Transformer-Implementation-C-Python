#include "tensor_reshape.h"

// 将3D张量重塑为4D张量
// input: [batch_size, seq_len, model_dim]
// output: [batch_size, num_heads, seq_len, head_dim]
bool tensor_reshape_3d_to_4d(
    const Tensor* input,
    int num_heads,
    Tensor* output
) {
    int batch_size = input->shape[0];
    int seq_len = input->shape[1];
    int model_dim = input->shape[2];
    int head_dim = model_dim / num_heads;

    // 检查维度是否匹配
    if (model_dim % num_heads != 0) {
        return false;
    }

    // 检查输出张量形状是否正确
    if (output->shape[0] != batch_size ||
        output->shape[1] != num_heads ||
        output->shape[2] != seq_len ||
        output->shape[3] != head_dim) {
        return false;
    }

    // 重新排列数据
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_len; s++) {
            for (int h = 0; h < num_heads; h++) {
                for (int d = 0; d < head_dim; d++) {
                    // 源索引: [b, s, h * head_dim + d]
                    int src_idx = (b * seq_len * model_dim) + 
                                (s * model_dim) + 
                                (h * head_dim + d);
                    
                    // 目标索引: [b, h, s, d]
                    int dst_idx = (b * num_heads * seq_len * head_dim) +
                                (h * seq_len * head_dim) +
                                (s * head_dim) + d;
                    
                    output->data[dst_idx] = input->data[src_idx];
                }
            }
        }
    }

    return true;
}



// 将4D张量重塑为3D张量, reshape with transpose
// input: [batch_size, num_heads, seq_len, head_dim] 
// output: [batch_size, seq_len, model_dim]
bool tensor_reshape_4d_to_3d(
    const Tensor* input,
    Tensor* output
) {
    // 这个函数用于将注意力计算后的4D张量重塑回3D张量
    // 输入: [batch_size, num_heads, seq_len, head_dim]
    // 输出: [batch_size, seq_len, model_dim]
    int batch_size = input->shape[0];
    int num_heads = input->shape[1];
    int seq_len = input->shape[2];
    int head_dim = input->shape[3];
    int model_dim = num_heads * head_dim;

    // 重新排列数据
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_len; s++) {
            for (int h = 0; h < num_heads; h++) {
                for (int d = 0; d < head_dim; d++) {
                    // 源索引: [b, h, s, d]
                    int src_idx = (b * num_heads * seq_len * head_dim) +
                                (h * seq_len * head_dim) +
                                (s * head_dim) + d;
                    
                    // 目标索引: [b, s, h * head_dim + d]
                    int dst_idx = (b * seq_len * model_dim) +
                                (s * model_dim) +
                                (h * head_dim + d);
                    
                    output->data[dst_idx] = input->data[src_idx];
                }
            }
        }
    }

    return true;
}