#include "optimizer.h"
#include "utils.h"
#include <stdlib.h>
#include <math.h>

void adam_init(AdamOptimizer* optimizer, int param_size) {
    if (!optimizer || param_size <= 0) return;
    
    optimizer->t = 0;
    optimizer->param_size = param_size;
    
    // 分配并初始化动量
    optimizer->m = calloc(param_size, sizeof(float));
    optimizer->v = calloc(param_size, sizeof(float));
}

void adam_step(AdamOptimizer* optimizer, float* params, float* gradients, 
               int param_size) {
    if (!optimizer || !params || !gradients || param_size <= 0) return;
    if (!optimizer->m || !optimizer->v) return;
    
    optimizer->t++;
    
    float alpha = optimizer->learning_rate * 
                 sqrt(1.0f - pow(optimizer->beta2, optimizer->t)) /
                 (1.0f - pow(optimizer->beta1, optimizer->t));
    
    for (int i = 0; i < param_size; i++) {
        // 更新偏差修正的一阶矩估计
        optimizer->m[i] = optimizer->beta1 * optimizer->m[i] + 
                         (1.0f - optimizer->beta1) * gradients[i];
        
        // 更新偏差修正的二阶矩估计
        optimizer->v[i] = optimizer->beta2 * optimizer->v[i] + 
                         (1.0f - optimizer->beta2) * gradients[i] * gradients[i];
        
        // 更新参数
        params[i] -= alpha * optimizer->m[i] / 
                    (sqrt(optimizer->v[i]) + optimizer->epsilon);
    }
}

AdamOptimizer* create_adam_optimizer(float learning_rate, float beta1, 
                                   float beta2, float epsilon, int param_size) {
    AdamOptimizer* optimizer = malloc(sizeof(AdamOptimizer));
    if (!optimizer) return NULL;
    
    optimizer->learning_rate = learning_rate;
    optimizer->beta1 = beta1;
    optimizer->beta2 = beta2;
    optimizer->epsilon = epsilon;
    
    adam_init(optimizer, param_size);
    
    if (!optimizer->m || !optimizer->v) {
        free_adam_optimizer(optimizer);
        return NULL;
    }
    
    return optimizer;
}

void free_adam_optimizer(AdamOptimizer* optimizer) {
    if (!optimizer) return;
    
    free(optimizer->m);
    free(optimizer->v);
    free(optimizer);
} 