#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <stdbool.h>

// Adam优化器结构体
typedef struct {
    float learning_rate;    // 学习率
    float beta1;           // 一阶矩估计的指数衰减率
    float beta2;           // 二阶矩估计的指数衰减率
    float epsilon;         // 数值稳定性常数
    
    // Adam状态
    float* m;              // 一阶矩估计
    float* v;              // 二阶矩估计
    int t;                 // 时间步
    int param_size;        // 参数总数
} AdamOptimizer;

// 函数声明
void adam_init(AdamOptimizer* optimizer, int param_size);
void adam_step(AdamOptimizer* optimizer, float* params, float* gradients, 
               int param_size);
AdamOptimizer* create_adam_optimizer(float learning_rate, float beta1, 
                                   float beta2, float epsilon, int param_size);
void free_adam_optimizer(AdamOptimizer* optimizer);

#endif // OPTIMIZER_H 