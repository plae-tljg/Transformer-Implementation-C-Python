#include "training.h"
#include "types.h"
#include "utils.h"
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

TransformerModel* initialize_model(int model_dim, 
                                 int num_heads, 
                                 int vocab_size,
                                 int num_encoder_layers, 
                                 int num_decoder_layers,
                                 float dropout_rate) {
    if (model_dim <= 0 || num_heads <= 0 || vocab_size <= 0 ||
        num_encoder_layers <= 0 || num_decoder_layers <= 0 ||
        dropout_rate < 0.0f || dropout_rate > 1.0f) {
        return NULL;
    }
    
    TransformerModel* model = malloc(sizeof(TransformerModel));
    if (!model) return NULL;
    
    // 初始化基本参数
    model->model_dim = model_dim;
    model->num_heads = num_heads;
    model->vocab_size = vocab_size;
    model->num_encoder_layers = num_encoder_layers;
    model->num_decoder_layers = num_decoder_layers;
    model->dropout_rate = dropout_rate;
    model->max_seq_length = 512;  // 默认值
    
    // 初始化配置
    model->config = malloc(sizeof(ModelConfig));
    if (!model->config) {
        free(model);
        return NULL;
    }
    
    // 设置配置参数
    model->config->model_dim = model_dim;
    model->config->num_heads = num_heads;
    model->config->vocab_size = vocab_size;
    model->config->max_seq_length = 512;
    model->config->num_layers = num_encoder_layers;
    model->config->learning_rate = 0.001f;
    model->config->dropout_rate = dropout_rate;
    model->config->use_bias = true;
    model->config->warmup_steps = 4000;
    model->config->weight_decay = 0.01f;
    model->config->use_layer_norm = true;
    model->config->seed = 42;
    
    // 初始化编码器层
    model->encoder_layers = malloc(num_encoder_layers * sizeof(EncoderLayer*));
    if (!model->encoder_layers) {
        free(model->config);
        free(model);
        return NULL;
    }
    
    // 初始化解码器层
    model->decoder_layers = malloc(num_decoder_layers * sizeof(DecoderLayer*));
    if (!model->decoder_layers) {
        free(model->encoder_layers);
        free(model->config);
        free(model);
        return NULL;
    }
    
    model->num_encoder_layers = num_encoder_layers;
    model->num_decoder_layers = num_decoder_layers;
    model->model_dim = model_dim;
    model->max_seq_length = 512;
    
    // 初始化每一层
    for (int i = 0; i < num_encoder_layers; i++) {
        model->encoder_layers[i] = initialize_encoder_layer(model_dim, num_heads);
        model->decoder_layers[i] = initialize_decoder_layer(model_dim, num_heads);
        if (!model->encoder_layers[i] || !model->decoder_layers[i]) {
            free_model(model);  // 清理已分配的内存
            return NULL;
        }
    }
    
    return model;
}

TrainingConfig* create_training_config(void) {
    TrainingConfig* config = malloc(sizeof(TrainingConfig));
    if (!config) return NULL;
    
    // 设置默认值
    config->learning_rate = 0.001f;
    config->batch_size = 32;
    config->num_epochs = 10;
    config->dropout_rate = 0.1f;
    config->gradient_clip_value = 1.0f;
    config->warmup_steps = 4000;
    config->use_gradient_clipping = true;
    config->save_interval = 1;
    config->checkpoint_dir = strdup("checkpoints/");
    
    // 其他默认值
    config->vocab_size = 0;  // 需要外部设置
    config->max_seq_length = 512;
    config->model_dim = 512;
    config->use_bias = true;
    config->weight_decay = 0.01f;
    config->seed = 42;
    
    // 检查内存分配
    if (!config->checkpoint_dir) {
        free(config);
        return NULL;
    }
    
    return config;
}

void free_training_config(TrainingConfig* config) {
    if (!config) return;
    
    free(config->checkpoint_dir);
    free(config);
} 