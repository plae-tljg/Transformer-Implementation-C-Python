#include "layers.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// ReLU激活函数的辅助函数
static float relu(float x) {
    return x > 0 ? x : 0;
}

FeedForward* feed_forward_create(int input_dim, int hidden_dim, bool requires_grad) {
    FeedForward* ff = (FeedForward*)malloc(sizeof(FeedForward));
    if (!ff) return NULL;

    ff->input_dim = input_dim;
    ff->hidden_dim = hidden_dim;
    ff->requires_grad = requires_grad;

    // 分配权重和偏置的内存
    ff->w1 = (float*)calloc(hidden_dim * input_dim, sizeof(float));
    ff->b1 = (float*)calloc(hidden_dim, sizeof(float));
    ff->w2 = (float*)calloc(input_dim * hidden_dim, sizeof(float));
    ff->b2 = (float*)calloc(input_dim, sizeof(float));

    if (!ff->w1 || !ff->b1 || !ff->w2 || !ff->b2) {
        feed_forward_free(ff);
        return NULL;
    }

    // 初始化权重（使用简单的Xavier初始化）
    float w1_scale = sqrt(2.0f / input_dim);
    float w2_scale = sqrt(2.0f / hidden_dim);
    
    for (int i = 0; i < hidden_dim * input_dim; i++) {
        ff->w1[i] = ((float)rand() / RAND_MAX * 2 - 1) * w1_scale;
    }
    for (int i = 0; i < input_dim * hidden_dim; i++) {
        ff->w2[i] = ((float)rand() / RAND_MAX * 2 - 1) * w2_scale;
    }

    return ff;
}

void feed_forward_free(FeedForward* ff) {
    if (ff) {
        free(ff->w1);
        free(ff->b1);
        free(ff->w2);
        free(ff->b2);
        free(ff);
    }
}

void feed_forward_forward(
    FeedForward* ff,
    float* input,       // [batch_size, input_dim]
    int batch_size,
    float* output      // [batch_size, input_dim]
) {
    // 临时缓冲区用于存储中间结果
    float* hidden = (float*)malloc(batch_size * ff->hidden_dim * sizeof(float));
    if (!hidden) return;

    // 第一个线性变换: input -> hidden
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < ff->hidden_dim; h++) {
            float sum = ff->b1[h];
            for (int i = 0; i < ff->input_dim; i++) {
                sum += input[b * ff->input_dim + i] * ff->w1[h * ff->input_dim + i];
            }
            // 应用ReLU激活函数
            hidden[b * ff->hidden_dim + h] = relu(sum);
        }
    }

    // 第二个线性变换: hidden -> output
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < ff->input_dim; i++) {
            float sum = ff->b2[i];
            for (int h = 0; h < ff->hidden_dim; h++) {
                sum += hidden[b * ff->hidden_dim + h] * ff->w2[i * ff->hidden_dim + h];
            }
            output[b * ff->input_dim + i] = sum;
        }
    }

    free(hidden);
} 