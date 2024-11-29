#include "utils.h"
#include "training.h"
#include <stdlib.h>
#include <stdio.h>

bool load_training_data(float*** train_data, float*** train_labels, int* num_samples) {
    // 这里实现数据加载逻辑
    // 示例实现：
    *num_samples = 1000; // 示例数据量
    
    // 分配内存
    *train_data = malloc(*num_samples * sizeof(float*));
    *train_labels = malloc(*num_samples * sizeof(float*));
    if (!*train_data || !*train_labels) {
        fprintf(stderr, "Failed to allocate memory for training data\n");
        return false;
    }
    
    // 为每个样本分配内存并初始化（示例）
    for (int i = 0; i < *num_samples; i++) {
        (*train_data)[i] = calloc(512, sizeof(float));  // 假设序列长度为512
        (*train_labels)[i] = calloc(512, sizeof(float));
        if (!(*train_data)[i] || !(*train_labels)[i]) {
            fprintf(stderr, "Failed to allocate memory for sample %d\n", i);
            return false;
        }
    }
    
    return true;
}

void free_training_data(float** train_data, float** train_labels, int num_samples) {
    if (train_data) {
        for (int i = 0; i < num_samples; i++) {
            free(train_data[i]);
        }
        free(train_data);
    }
    
    if (train_labels) {
        for (int i = 0; i < num_samples; i++) {
            free(train_labels[i]);
        }
        free(train_labels);
    }
}

int calculate_total_params(TransformerModel* model) {
    if (!model) return 0;
    
    int total = 0;
    
    // 词嵌入参数
    total += model->vocab_size * model->model_dim;
    
    // 编码器层参数
    for (int i = 0; i < model->num_encoder_layers; i++) {
        // 自注意力层参数
        total += 4 * model->model_dim * model->model_dim;  // Q, K, V, 和输出投影
        // 前馈网络参数
        total += 2 * model->model_dim * (4 * model->model_dim);  // 两个线性层
    }
    
    // 解码器层参数
    for (int i = 0; i < model->num_decoder_layers; i++) {
        // 自注意力层参数
        total += 4 * model->model_dim * model->model_dim;
        // 交叉注意力层参数
        total += 4 * model->model_dim * model->model_dim;
        // 前馈网络参数
        total += 2 * model->model_dim * (4 * model->model_dim);
    }
    
    return total;
} 