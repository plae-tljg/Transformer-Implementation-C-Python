#include "training.h"
#include "utils.h"
#include "optimizer.h"
#include <stdlib.h>
#include <stdio.h>

void train_epoch(TransformerModel* model, float** input_data, float** target_data,
                int num_samples, AdamOptimizer* optimizer) {
    if (!model || !input_data || !target_data || !optimizer || num_samples <= 0) {
        return;
    }

    float total_loss = 0.0f;
    
    for (int i = 0; i < num_samples; i++) {
        // 前向传播
        float* predictions = forward_pass(model, input_data[i], model->max_seq_length);
        if (!predictions) continue;
        
        // 计算损失
        float loss = compute_loss(predictions, target_data[i], model->max_seq_length);
        total_loss += loss;
        
        // 计算损失梯度
        float* loss_grad = malloc(model->max_seq_length * model->model_dim * sizeof(float));
        if (!loss_grad) {
            free(predictions);
            continue;
        }
        
        // TODO: 计算实际的损失梯度
        
        // 反向传播
        backward_pass(model, loss_grad);
        
        // 应用梯度
        apply_gradients(model, optimizer);
        
        // 清理
        free(predictions);
        free(loss_grad);
        
        // 打印进度
        if ((i + 1) % 100 == 0) {
            printf("Processed %d/%d samples, Average Loss: %f\n", 
                   i + 1, num_samples, total_loss / (i + 1));
        }
    }
    
    printf("Epoch completed. Average Loss: %f\n", total_loss / num_samples);
}

void train_step(TransformerModel* model, float* input, float* target) {
    if (!model || !input || !target) return;

    // 前向传播
    float* output = forward_pass(model, input, model->max_seq_length);
    if (!output) return;

    // 计算损失梯度
    int output_size = model->max_seq_length * model->model_dim;
    float* output_grad = compute_loss_gradient(output, target, output_size);
    if (!output_grad) {
        free(output);
        return;
    }

    // 反向传播，累积梯度
    float* decoder_grad = decoder_backward_pass(model, output_grad);
    if (decoder_grad) {
        // 应用累积的梯度
        float learning_rate = model->config->learning_rate;
        for (int i = 0; i < model->num_decoder_layers; i++) {
            apply_layer_gradients(model->decoder_layers[i], learning_rate);
        }
        free(decoder_grad);
    }

    // 清理
    free(output);
    free(output_grad);
}