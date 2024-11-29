#include "layers.h"
#include "utils.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// 激活函数
static float relu(float x) {
    return x > 0 ? x : 0;
}

// 初始化权重
static void initialize_weights(float* weights, int size) {
    for (int i = 0; i < size; i++) {
        weights[i] = ((float)rand() / RAND_MAX) * 0.02f - 0.01f;
    }
}

FeedForward* initialize_feed_forward(int input_dim, int hidden_dim) {
    if (input_dim <= 0 || hidden_dim <= 0) return NULL;
    
    FeedForward* ff = malloc(sizeof(FeedForward));
    if (!ff) return NULL;
    
    // 初始化基本参数
    ff->input_dim = input_dim;
    ff->hidden_dim = hidden_dim;
    ff->requires_grad = true;
    
    // 分配内存
    ff->weight1 = malloc(input_dim * hidden_dim * sizeof(float));
    ff->weight2 = malloc(hidden_dim * input_dim * sizeof(float));
    ff->bias1 = malloc(hidden_dim * sizeof(float));
    ff->bias2 = malloc(input_dim * sizeof(float));
    
    // 分配梯度内存
    ff->weight1_gradients = malloc(input_dim * hidden_dim * sizeof(float));
    ff->weight2_gradients = malloc(hidden_dim * input_dim * sizeof(float));
    ff->bias1_gradients = malloc(hidden_dim * sizeof(float));
    ff->bias2_gradients = malloc(input_dim * sizeof(float));
    
    // 分配缓存内存
    ff->hidden_layer_output = malloc(hidden_dim * sizeof(float));
    ff->activation_output = malloc(hidden_dim * sizeof(float));
    ff->input_cache = malloc(input_dim * sizeof(float));
    
    // 检查内存分配
    if (!ff->weight1 || !ff->weight2 || !ff->bias1 || !ff->bias2 ||
        !ff->weight1_gradients || !ff->weight2_gradients ||
        !ff->bias1_gradients || !ff->bias2_gradients ||
        !ff->hidden_layer_output || !ff->activation_output ||
        !ff->input_cache) {
        free_feed_forward(ff);
        return NULL;
    }
    
    // 初始化权重和偏置
    initialize_weights(ff->weight1, input_dim * hidden_dim);
    initialize_weights(ff->weight2, hidden_dim * input_dim);
    memset(ff->bias1, 0, hidden_dim * sizeof(float));
    memset(ff->bias2, 0, input_dim * sizeof(float));
    
    return ff;
}

void free_feed_forward(FeedForward* ff) {
    if (!ff) return;
    
    // 释放权重和偏置
    free(ff->weight1);
    free(ff->weight2);
    free(ff->bias1);
    free(ff->bias2);
    
    // 释放梯度
    free(ff->weight1_gradients);
    free(ff->weight2_gradients);
    free(ff->bias1_gradients);
    free(ff->bias2_gradients);
    
    // 释放缓存
    free(ff->hidden_layer_output);
    free(ff->activation_output);
    free(ff->input_cache);
    
    // 释放结构体
    free(ff);
}

float* feed_forward_forward(FeedForward* ff, float* input, int seq_length) {
    if (!ff || !input || seq_length <= 0) return NULL;
    
    // 缓存输入用于反向传播
    if (ff->requires_grad) {
        memcpy(ff->input_cache, input, seq_length * ff->input_dim * sizeof(float));
    }
    
    // 第一层线性变换
    float* hidden = matrix_multiply(input, ff->weight1, seq_length, ff->input_dim, ff->hidden_dim);
    if (!hidden) return NULL;
    
    // 添加偏置并应用激活函数
    for (int i = 0; i < seq_length * ff->hidden_dim; i++) {
        hidden[i] += ff->bias1[i % ff->hidden_dim];
        hidden[i] = relu(hidden[i]);
    }
    
    // 缓存隐藏层输出用于反向传播
    if (ff->requires_grad) {
        memcpy(ff->hidden_layer_output, hidden, seq_length * ff->hidden_dim * sizeof(float));
    }
    
    // 第二层线性变换
    float* output = matrix_multiply(hidden, ff->weight2, seq_length, ff->hidden_dim, ff->input_dim);
    free(hidden);
    if (!output) return NULL;
    
    // 添加偏置
    for (int i = 0; i < seq_length * ff->input_dim; i++) {
        output[i] += ff->bias2[i % ff->input_dim];
    }
    
    return output;
} 