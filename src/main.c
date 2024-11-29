#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "attention.h"
#include "layers.h"
#include "embeddings.h"
#include "utils.h"
#include "training.h"

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif

// 初始化训练配置
TrainingConfig* create_training_config() {
    TrainingConfig* config = malloc(sizeof(TrainingConfig));
    if (!config) {
        fprintf(stderr, "Failed to allocate training config\n");
        return NULL;
    }
    
    config->learning_rate = 0.0001f;
    config->batch_size = 32;
    config->max_epochs = 10;
    config->dropout_rate = 0.1f;
    config->gradient_clip_value = 1.0f;
    config->warmup_steps = 4000;
    config->use_gradient_clipping = true;
    config->save_interval = 1;
    config->checkpoint_dir = strdup("checkpoints/");
    
    return config;
}

// 初始化模型参数
TransformerModel* initialize_model(int model_dim, int num_heads, int vocab_size,
                                 int num_encoder_layers, int num_decoder_layers,
                                 int max_seq_length) {
    TransformerModel* model = malloc(sizeof(TransformerModel));
    if (!model) {
        fprintf(stderr, "Failed to allocate model\n");
        return NULL;
    }
    
    // 设置模型参数
    model->model_dim = model_dim;
    model->vocab_size = vocab_size;
    model->max_seq_length = max_seq_length;
    model->num_encoder_layers = num_encoder_layers;
    model->num_decoder_layers = num_decoder_layers;
    
    // 初始化词嵌入
    model->token_embedding = malloc(sizeof(struct TokenEmbedding));
    if (!model->token_embedding) goto cleanup;
    model->token_embedding->embedding_matrix = malloc(vocab_size * model_dim * sizeof(float));
    if (!model->token_embedding->embedding_matrix) goto cleanup;
    model->token_embedding->vocab_size = vocab_size;
    model->token_embedding->embedding_dim = model_dim;
    // 初始化位置编码
    model->positional_encoding = malloc(sizeof(PositionalEncoding));
    if (!model->positional_encoding) goto cleanup;
    model->positional_encoding->positional_encoding = 
        malloc(max_seq_length * model_dim * sizeof(float));
    if (!model->positional_encoding->positional_encoding) goto cleanup;
    model->positional_encoding->max_seq_length = max_seq_length;
    model->positional_encoding->embedding_dim = model_dim;
    
    // 初始化编码器层
    model->encoder_layers = malloc(num_encoder_layers * sizeof(EncoderLayer*));
    if (!model->encoder_layers) goto cleanup;
    
    for (int i = 0; i < num_encoder_layers; i++) {
        model->encoder_layers[i] = malloc(sizeof(EncoderLayer));
        if (!model->encoder_layers[i]) goto cleanup;
        
        // 初始化编码器层的组件
        if (!initialize_encoder_layer(model->encoder_layers[i], model_dim, num_heads)) {
            goto cleanup;
        }
    }
    
    // 初始化解码器层
    model->decoder_layers = malloc(num_decoder_layers * sizeof(DecoderLayer*));
    if (!model->decoder_layers) goto cleanup;
    
    for (int i = 0; i < num_decoder_layers; i++) {
        model->decoder_layers[i] = malloc(sizeof(DecoderLayer));
        if (!model->decoder_layers[i]) goto cleanup;
        
        // 初始化解码器层的组件
        if (!initialize_decoder_layer(model->decoder_layers[i], model_dim, num_heads)) {
            goto cleanup;
        }
    }
    
    // 初始化训练配置
    model->config = create_training_config();
    if (!model->config) goto cleanup;
    
    return model;

cleanup:
    fprintf(stderr, "Failed to initialize model components\n");
    free_model(model);
    return NULL;
}

void train_model(TransformerModel* model, float** train_data, float** train_labels,
                int num_samples) {
    if (!model || !train_data || !train_labels) {
        fprintf(stderr, "Invalid training inputs\n");
        return;
    }
    
    // 初始化优化器
    AdamOptimizer optimizer;
    adam_init(&optimizer, calculate_total_params(model), model->config->learning_rate);
    
    printf("Starting training...\n");
    float total_loss = 0.0f;
    
    // 训练循环
    for (int epoch = 0; epoch < model->config->max_epochs; epoch++) {
        float epoch_loss = 0.0f;
        
        // 训练一个epoch
        train_epoch(model, train_data, train_labels, num_samples, &optimizer);
        
        // 计算epoch损失
        epoch_loss = total_loss / num_samples;
        
        // 保存检查点
        if (epoch % model->config->save_interval == 0) {
            save_checkpoint(model, epoch, epoch_loss);
        }
        
        printf("Epoch %d/%d - Loss: %.4f\n", 
               epoch + 1, model->config->max_epochs, epoch_loss);
        
        total_loss = 0.0f;  // 重置总损失
    }
    
    // 清理优化器
    adam_free(&optimizer);
}

int main() {
    // 模型参数
    const int model_dim = 512;
    const int num_heads = 8;
    const int vocab_size = 30000;
    const int num_encoder_layers = 6;
    const int num_decoder_layers = 6;
    const int max_seq_length = 512;
    
    // 初始化模型
    TransformerModel* model = initialize_model(model_dim, num_heads, vocab_size,
                                             num_encoder_layers, num_decoder_layers,
                                             max_seq_length);
    if (!model) {
        fprintf(stderr, "Failed to initialize model\n");
        return 1;
    }
    
    // 这里应该加载训练数据
    float** train_data = NULL;
    float** train_labels = NULL;
    int num_samples = 0;
    
    // 加载训练数据的代码
    if (!load_training_data(&train_data, &train_labels, &num_samples)) {
        fprintf(stderr, "Failed to load training data\n");
        free_model(model);
        return 1;
    }
    
    // 创建检查点目录
    #ifdef _WIN32
    _mkdir(model->config->checkpoint_dir);
    #else
    mkdir(model->config->checkpoint_dir, 0777);
    #endif
    
    // 训练模型
    train_model(model, train_data, train_labels, num_samples);
    
    // 保存最终模型
    if (!save_model(model, "final_model.bin")) {
        fprintf(stderr, "Failed to save final model\n");
    }
    
    // 清理资源
    free_training_data(train_data, train_labels, num_samples);
    free_model(model);
    
    return 0;
} 