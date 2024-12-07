#include "model_config.h"
#include <stdbool.h>

// 定义全局配置实例
ModelConfig g_model_config = {0};  // 初始化为0

void init_model_config(
    int batch_size, 
    int max_seq_length, 
    int vocab_size, 
    int d_model, 
    int ff_dim,
    int num_heads, 
    float dropout_prob
) {
    g_model_config.batch_size = batch_size;
    g_model_config.max_seq_length = max_seq_length;
    g_model_config.vocab_size = vocab_size;
    g_model_config.d_model = d_model;
    g_model_config.ff_dim = ff_dim;
    g_model_config.num_heads = num_heads;
    g_model_config.dropout_prob = dropout_prob;
    g_model_config.is_training = true;  // 默认为训练模式
}