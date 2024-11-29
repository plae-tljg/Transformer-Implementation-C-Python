#include "layers.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>

LayerNorm* initialize_layer_norm(int dim) {
    LayerNorm* ln = malloc(sizeof(LayerNorm));
    if (!ln) return NULL;
    
    ln->dim = dim;
    ln->epsilon = 1e-5f;
    ln->requires_grad = true;
    
    // 分配参数内存
    ln->gamma = malloc(dim * sizeof(float));
    ln->beta = malloc(dim * sizeof(float));
    ln->gamma_grad = malloc(dim * sizeof(float));
    ln->beta_grad = malloc(dim * sizeof(float));
    
    // 分配缓存内存
    ln->mean_cache = malloc(dim * sizeof(float));
    ln->var_cache = malloc(dim * sizeof(float));
    ln->norm_cache = malloc(dim * sizeof(float));
    ln->input_cache = malloc(dim * sizeof(float));
    
    // 检查内存分配
    if (!ln->gamma || !ln->beta || !ln->gamma_grad || !ln->beta_grad ||
        !ln->mean_cache || !ln->var_cache || !ln->norm_cache || !ln->input_cache) {
        free_layer_norm(ln);
        return NULL;
    }
    
    // 初始化参数
    for (int i = 0; i < dim; i++) {
        ln->gamma[i] = 1.0f;
        ln->beta[i] = 0.0f;
        ln->gamma_grad[i] = 0.0f;
        ln->beta_grad[i] = 0.0f;
    }
    
    return ln;
}

void free_layer_norm(LayerNorm* ln) {
    if (!ln) return;
    
    free(ln->gamma);
    free(ln->beta);
    free(ln->gamma_grad);
    free(ln->beta_grad);
    free(ln->mean_cache);
    free(ln->var_cache);
    free(ln->norm_cache);
    free(ln->input_cache);
    
    free(ln);
} 