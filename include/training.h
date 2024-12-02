#ifndef TRAINING_H
#define TRAINING_H

typedef struct TrainingConfig TrainingConfig;
typedef struct OptimizerState OptimizerState;

// 训练配置
struct TrainingConfig {
    float learning_rate;
    int batch_size;
    int max_epochs;
    int warmup_steps;
    float clip_grad;
    float weight_decay;
    float dropout_rate;
    int gradient_accumulation_steps;
    int print_every;
    int save_interval;
};

// 优化器状态
struct OptimizerState {
    float* momentum;  // Adam m
    float* velocity;  // Adam v
    float beta1;      // 通常是0.9
    float beta2;      // 通常是0.999
    float epsilon;    // 通常是1e-8
    int step;         // 当前步数
};


#endif // TRAINING_H