#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "layers.h"

LayerNorm* layer_norm_create(int normalized_shape, float epsilon, bool requires_grad) {
    LayerNorm* ln = (LayerNorm*)malloc(sizeof(LayerNorm));
    if (!ln) {
        fprintf(stderr, "Failed to allocate memory for layer norm\n");
        return NULL;
    }

    ln->normalized_shape = normalized_shape;
    ln->epsilon = epsilon;
    ln->requires_grad = requires_grad;

    // 分配并初始化gamma和beta
    ln->gamma = (float*)malloc(normalized_shape * sizeof(float));
    ln->beta = (float*)malloc(normalized_shape * sizeof(float));
    
    if (!ln->gamma || !ln->beta) {
        fprintf(stderr, "Failed to allocate memory for layer norm parameters\n");
        layer_norm_free(ln);
        return NULL;
    }

    // 初始化gamma为1，beta为0
    for (int i = 0; i < normalized_shape; i++) {
        ln->gamma[i] = 1.0f;
        ln->beta[i] = 0.0f;
    }

    return ln;
}

void layer_norm_free(LayerNorm* ln) {
    if (!ln) return;
    
    free(ln->gamma);
    free(ln->beta);
    free(ln);
}

void layer_norm_forward(
    LayerNorm* ln,
    float* input,
    int batch_size,
    float* output
) {
    if (!ln || !input || !output) {
        fprintf(stderr, "NULL pointer in layer_norm_forward\n");
        return;
    }

    // 为每个样本计算均值和方差
    for (int i = 0; i < batch_size; i++) {
        float* current_input = input + i * ln->normalized_shape;
        float* current_output = output + i * ln->normalized_shape;
        
        // 计算均值
        float mean = 0.0f;
        for (int j = 0; j < ln->normalized_shape; j++) {
            mean += current_input[j];
        }
        mean /= ln->normalized_shape;
        
        // 计算方差
        float variance = 0.0f;
        for (int j = 0; j < ln->normalized_shape; j++) {
            float diff = current_input[j] - mean;
            variance += diff * diff;
        }
        variance /= ln->normalized_shape;
        
        // 归一化、缩放和偏移
        float std_dev = sqrtf(variance + ln->epsilon);
        for (int j = 0; j < ln->normalized_shape; j++) {
            float normalized = (current_input[j] - mean) / std_dev;
            current_output[j] = ln->gamma[j] * normalized + ln->beta[j];
        }
    }
}