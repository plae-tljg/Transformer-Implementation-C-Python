#include "softmax.h"
#include <math.h>
#include <float.h>

bool attention_scores_softmax(const Tensor* input, Tensor* output) {
    int batch_size = input->shape[0];
    int num_heads = input->shape[1];
    int seq_len = input->shape[2];
    // 最后一维也是seq_len
    
    // 对每个batch和head分别计算softmax
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < num_heads; h++) {
            for (int i = 0; i < seq_len; i++) {
                // 1. 找到最大值
                float max_val = -FLT_MAX;
                for (int j = 0; j < seq_len; j++) {
                    int idx = ((b * num_heads + h) * seq_len + i) * seq_len + j;
                    max_val = fmaxf(max_val, input->data[idx]);
                }
                
                // 2. 计算exp并求和
                float sum = 0.0f;
                for (int j = 0; j < seq_len; j++) {
                    int idx = ((b * num_heads + h) * seq_len + i) * seq_len + j;
                    output->data[idx] = expf(input->data[idx] - max_val);
                    sum += output->data[idx];
                }
                
                // 3. 归一化
                for (int j = 0; j < seq_len; j++) {
                    int idx = ((b * num_heads + h) * seq_len + i) * seq_len + j;
                    output->data[idx] /= sum;
                }
            }
        }
    }
    return true;
}
