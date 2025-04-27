#include "feed_forward.h"
#include "relu.h"
#include "tensor_mul.h"
#include "tensor_add.h"
#include <stdlib.h>

FeedForward* feed_forward_create(int input_dim, int hidden_dim) {
    FeedForward* ff = (FeedForward*)malloc(sizeof(FeedForward));
    if (!ff) return NULL;

    // 创建权重矩阵
    int w1_shape[] = {input_dim, hidden_dim};  // [input_dim, hidden_dim]
    int w2_shape[] = {hidden_dim, input_dim};  // [hidden_dim, input_dim]
    int b1_shape[] = {hidden_dim};             // [hidden_dim]
    int b2_shape[] = {input_dim};              // [input_dim]

    ff->w1 = tensor_create(w1_shape, 2);
    ff->w2 = tensor_create(w2_shape, 2);
    ff->b1 = tensor_create(b1_shape, 1);
    ff->b2 = tensor_create(b2_shape, 1);

    return ff;
}

bool feed_forward_forward(FeedForward* ff, const Tensor* input, Tensor* output) {
    // input shape: [batch_size, input_dim]
    int batch_size = input->shape[0];
    int input_dim = input->shape[1];
    int hidden_dim = ff->w1->shape[0];

    // 第一个线性变换: intermediate = x * W1 + b1 (注意这里是x乘以W1,而不是W1乘以x)
    if (!tensor_matmul_2d(input, ff->w1, output)) return false;
    if (!tensor_add(output, ff->b1, output)) return false;

    // ReLU激活
    relu_forward(output, output);

    // 第二个线性变换: output = relu_output * W2 + b2
    if (!tensor_matmul_2d(output, ff->w2, output)) return false;
    if (!tensor_add(output, ff->b2, output)) return false;

    return true;
}

void feed_forward_free(FeedForward* ff) {
    if (ff) {
        tensor_free(ff->w1);
        tensor_free(ff->b1);
        tensor_free(ff->w2);
        tensor_free(ff->b2);
        free(ff);
    }
}
