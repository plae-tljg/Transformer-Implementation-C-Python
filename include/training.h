#ifndef TRAINING_H
#define TRAINING_H

#include <stdbool.h>

typedef struct TrainingConfig TrainingConfig;

struct TrainingConfig {
    float learning_rate;        // 学习率, 0.001f
    int batch_size;            // 批次大小, 4
    int num_epochs;            // 训练轮数, 10
    float dropout_rate;        // Dropout率, 0.1f, start from 0.1-0.3
    float gradient_clip_value; // 梯度裁剪值, 1.0f, usually 0.5-5.0
    int warmup_steps;          // 预热步数, 4000, usually 10% of total steps
    bool use_gradient_clipping; // 是否使用梯度裁剪
    int save_interval;         // 保存间隔, 1, save model every 1 epoch
    char* checkpoint_dir;      // 检查点目录
    int vocab_size;            // 词汇表大小
    int max_seq_length;        // 最大序列长度
    int model_dim;            // 模型维度
    int num_heads;            // 注意力头数
    int num_layers;           // 层数
    bool use_bias;            // 是否使用偏置
    float weight_decay;       // 权重衰减
    int seed;                 // 随机种子
};





#endif