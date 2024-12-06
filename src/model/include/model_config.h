#ifndef MODEL_CONFIG_H
#define MODEL_CONFIG_H

#include <stdbool.h>

typedef struct ModelConfig {
    int batch_size;      // 批处理大小
    int max_seq_length;  // 最大序列长度
    int vocab_size;      // 词汇表大小
    int d_model;         // 模型维度 (embedding_dim)
    int num_heads;       // 注意力头数量
    float dropout_prob;  // dropout概率
    bool is_training;    // 是否处于训练模式
} ModelConfig;

// 全局配置实例
extern ModelConfig g_model_config;

// 初始化配置
void init_model_config(int batch_size, int max_seq_length, int vocab_size, 
                      int d_model, int num_heads, float dropout_prob);

#endif