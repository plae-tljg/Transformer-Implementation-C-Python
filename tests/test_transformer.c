#include "model.h"
#include "training.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>

void test_transformer() {
    printf("Initializing model...\n");
    
    // 创建一个小型的测试模型
    int model_dim = 64;
    int num_heads = 4;
    int vocab_size = 1000;
    int max_seq_length = 32;
    int num_layers = 2;
    float learning_rate = 0.001f;
    
    TransformerModel* model = initialize_model(model_dim, num_heads, vocab_size,
                                             max_seq_length, num_layers, learning_rate);
    if (!model) {
        printf("Failed to initialize model\n");
        return;
    }
    printf("Model initialized successfully\n");
    
    // 创建测试数据
    printf("Creating test data...\n");
    float* input = malloc(max_seq_length * model_dim * sizeof(float));
    float* target = malloc(max_seq_length * model_dim * sizeof(float));
    if (!input || !target) {
        printf("Failed to allocate test data\n");
        free_model(model);
        free(input);
        free(target);
        return;
    }
    
    // 初始化测试数据
    for (int i = 0; i < max_seq_length * model_dim; i++) {
        input[i] = (float)rand() / RAND_MAX;
        target[i] = (float)rand() / RAND_MAX;
    }
    printf("Test data created successfully\n");
    
    // 测试前向传播
    printf("Testing forward pass...\n");
    float* output = forward_pass(model, input, max_seq_length);
    if (!output) {
        printf("Forward pass failed\n");
        free_model(model);
        free(input);
        free(target);
        return;
    }
    printf("Forward pass completed successfully\n");
    
    // 测试损失计算
    printf("Testing loss computation...\n");
    float loss = compute_loss(output, target, max_seq_length * model_dim);
    printf("Initial loss: %f\n", loss);
    
    // 测试反向传播
    printf("Testing backward pass...\n");
    float* loss_grad = compute_loss_gradient(output, target, max_seq_length * model_dim);
    if (!loss_grad) {
        printf("Loss gradient computation failed\n");
        free_model(model);
        free(input);
        free(target);
        free(output);
        return;
    }
    
    float* decoder_grad = decoder_backward_pass(model, loss_grad);
    if (!decoder_grad) {
        printf("Backward pass failed\n");
        free_model(model);
        free(input);
        free(target);
        free(output);
        free(loss_grad);
        return;
    }
    printf("Backward pass completed successfully\n");
    
    // 测试梯度更新
    printf("Testing gradient update...\n");
    for (int i = 0; i < model->num_decoder_layers; i++) {
        apply_layer_gradients(model->decoder_layers[i], model->config->learning_rate);
    }
    printf("Gradient update completed successfully\n");
    
    // 清理
    printf("Cleaning up...\n");
    free_model(model);
    free(input);
    free(target);
    free(output);
    free(loss_grad);
    free(decoder_grad);
    printf("All tests completed successfully\n");
}

int main() {
    printf("Running tests...\n");
    test_transformer();
    return 0;
} 