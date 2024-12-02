/* Standard Library Headers */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

/* Project Headers */
#include "training.h"
#include "model.h"
#include "embeddings.h"
#include "layers.h"
#include "attention.h"

// 默认配置值
#define DEFAULT_VOCAB_SIZE 32000
#define DEFAULT_MODEL_DIM 512
#define DEFAULT_NUM_HEADS 8
#define DEFAULT_NUM_LAYERS 6
#define DEFAULT_FF_DIM 2048
#define MAX_SEQ_LENGTH 1024

void print_usage() {
    printf("用法：\n");
    printf("训练模式：./program train --config <config_path> [options]\n");
    printf("生成模式：./program generate --model <model_path> [options]\n");
    printf("\n选项：\n");
    printf("  --batch-size <num>     批次大小 (默认: 32)\n");
    printf("  --model-dim <num>      模型维度 (默认: 512)\n");
    printf("  --num-heads <num>      注意力头数 (默认: 8)\n");
    printf("  --num-layers <num>     层数 (默认: 6)\n");
}

TrainingConfig* create_training_config() {
    TrainingConfig* config = (TrainingConfig*)malloc(sizeof(TrainingConfig));
    if (!config) {
        fprintf(stderr, "Failed to allocate memory for TrainingConfig\n");
        exit(EXIT_FAILURE);
    }
    config->learning_rate = 0.0001f;
    config->batch_size = 4;
    config->max_epochs = 10;
    config->dropout_rate = 0.1f;
    config->gradient_accumulation_steps = 1;
    config->warmup_steps = 4000;
    config->print_every = 100;
    config->save_interval = 1000;

    return config;
}

TransformerModel* initialize_model(int model_dim, int num_heads, int vocab_size,
                                 int num_encoder_layers, int num_decoder_layers,
                                 int max_seq_length, bool is_training) {
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
    model->dropout_rate = is_training ? 0.1f : 0.0f;
    
    // 初始化编码器层
    model->encoder_layers = malloc(num_encoder_layers * sizeof(EncoderLayer*));
    if (!model->encoder_layers) goto cleanup;
    
    for (int i = 0; i < num_encoder_layers; i++) {
        model->encoder_layers[i] = encoder_layer_create(
            model_dim, 
            num_heads, 
            model_dim * 4,  // ff_hidden_dim
            is_training    // requires_grad
        );
        if (!model->encoder_layers[i]) goto cleanup;
    }
    
    // 初始化解码器层
    model->decoder_layers = malloc(num_decoder_layers * sizeof(DecoderLayer*));
    if (!model->decoder_layers) goto cleanup;
    
    for (int i = 0; i < num_decoder_layers; i++) {
        model->decoder_layers[i] = decoder_layer_create(
            model_dim, 
            num_heads, 
            model_dim * 4,  // ff_hidden_dim
            is_training    // requires_grad
        );
        if (!model->decoder_layers[i]) goto cleanup;
    }
    
    // 初始化配置
    if (is_training) {
        model->config = (ModelConfig*)create_training_config();
        if (!model->config) goto cleanup;
    }
    
    return model;

cleanup:
    fprintf(stderr, "Failed to initialize model components\n");
    free_model(model);
    return NULL;
}

void generate_text(TransformerModel* model, const char* input_text) {
    // TODO: 实现文本生成逻辑
    printf("Text generation not implemented yet\n");
}

int main(int argc, char **argv) {
    if (argc < 2) {
        print_usage();
        return 1;
    }

    // 解析命令行参数
    bool is_training = strcmp(argv[1], "train") == 0;
    int model_dim = DEFAULT_MODEL_DIM;
    int num_heads = DEFAULT_NUM_HEADS;
    int num_layers = DEFAULT_NUM_LAYERS;
    int vocab_size = DEFAULT_VOCAB_SIZE;
    const char* model_path = NULL;

    for (int i = 2; i < argc; i += 2) {
        if (i + 1 >= argc) {
            fprintf(stderr, "Error: option %s requires a value\n", argv[i]);
            return 1;
        }

        if (strcmp(argv[i], "--model-dim") == 0) {
            model_dim = atoi(argv[i + 1]);
        } else if (strcmp(argv[i], "--num-heads") == 0) {
            num_heads = atoi(argv[i + 1]);
        } else if (strcmp(argv[i], "--num-layers") == 0) {
            num_layers = atoi(argv[i + 1]);
        } else if (strcmp(argv[i], "--model") == 0) {
            model_path = argv[i + 1];
        }
    }

    // 初始化模型
    TransformerModel* model = initialize_model(
        model_dim,
        num_heads,
        vocab_size,
        num_layers,
        num_layers,  // 编码器和解码器使用相同的层数
        MAX_SEQ_LENGTH,
        is_training
    );

    if (!model) {
        fprintf(stderr, "Model initialization failed\n");
        return 1;
    }

    if (is_training) {
        // 训练模式
        TrainingConfig* config = create_training_config();
        if (!config) {
            fprintf(stderr, "Failed to create training config\n");
            free_model(model);
            return 1;
        }

        printf("Starting training...\n");
        
        // 创建优化器
        OptimizerState* optimizer = create_optimizer(model);
        if (!optimizer) {
            fprintf(stderr, "Failed to create optimizer\n");
            free(config);
            free_model(model);
            return 1;
        }

        // 加载训练数据
        int num_samples = 1000; // 示例数据量
        int src_len = 50;  // 源序列长度
        int tgt_len = 50;  // 目标序列长度
        
        int* src_data = malloc(num_samples * src_len * sizeof(int));
        int* tgt_data = malloc(num_samples * tgt_len * sizeof(int));
        
        if (!src_data || !tgt_data) {
            fprintf(stderr, "Failed to allocate training data\n");
            free(optimizer);
            free(config);
            free_model(model);
            return 1;
        }

        // 训练循环
        for (int epoch = 0; epoch < config->max_epochs; epoch++) {
            printf("Epoch %d/%d\n", epoch + 1, config->max_epochs);
            
            train_epoch(model, src_data, tgt_data, num_samples, 
                       src_len, tgt_len, config, optimizer);
        }

        // 清理
        free(src_data);
        free(tgt_data);
        free(optimizer);
        free(config);
    } else {
        // 生成模式
        if (model_path) {
            FILE* model_file = fopen(model_path, "rb");
            if (!model_file) {
                fprintf(stderr, "Failed to open model file: %s\n", model_path);
                free_model(model);
                return 1;
            }
            
            // 加载模型参数
            if (!load_model_parameters(model, model_file)) {
                fprintf(stderr, "Failed to load model parameters\n");
                fclose(model_file);
                free_model(model);
                return 1;
            }
            
            fclose(model_file);
            printf("Successfully loaded model from: %s\n", model_path);
        }

        printf("Entering interactive generation mode (type 'quit' to exit)\n");
        char input_buffer[4096];
        
        while (1) {
            printf("\nEnter text > ");
            if (!fgets(input_buffer, sizeof(input_buffer), stdin)) break;
            
            // 检查是否退出
            if (strncmp(input_buffer, "quit", 4) == 0) break;
            
            // 生成文本
            generate_text(model, input_buffer);
        }
    }

    free_model(model);
    return 0;
}
