#include "model.h"
#include <stdlib.h>
#include <string.h>

void free_model(TransformerModel* model) {
    if (!model) return;
    
    // 释放编码器层
    if (model->encoder_layers) {
        for (int i = 0; i < model->num_encoder_layers; i++) {
            if (model->encoder_layers[i]) {
                free_encoder_layer(model->encoder_layers[i]);
            }
        }
        free(model->encoder_layers);
    }
    
    // 释放解码器层
    if (model->decoder_layers) {
        for (int i = 0; i < model->num_decoder_layers; i++) {
            if (model->decoder_layers[i]) {
                free_decoder_layer(model->decoder_layers[i]);
            }
        }
        free(model->decoder_layers);
    }
    
    // 释放配置
    if (model->config) {
        free(model->config);
    }
    
    // 释放模型本身
    free(model);
}

void reset_model_gradients(TransformerModel* model) {
    if (!model) return;
    
    // 重置编码器梯度
    for (int i = 0; i < model->num_encoder_layers; i++) {
        EncoderLayer* layer = model->encoder_layers[i];
        if (!layer) continue;
        
        // 重置自注意力梯度
        if (layer->self_attention) {
            for (int h = 0; h < layer->self_attention->num_heads; h++) {
                SelfAttention* head = layer->self_attention->attention_heads[h];
                if (!head || !head->requires_grad) continue;
                
                memset(head->query_gradients, 0, 
                       sizeof(float) * layer->self_attention->head_dim * layer->self_attention->head_dim);
                memset(head->key_gradients, 0, 
                       sizeof(float) * layer->self_attention->head_dim * layer->self_attention->head_dim);
                memset(head->value_gradients, 0, 
                       sizeof(float) * layer->self_attention->head_dim * layer->self_attention->head_dim);
            }
        }
        
        // 重置前馈网络梯度
        if (layer->feed_forward && layer->feed_forward->requires_grad) {
            const int ff_dim = 4 * model->model_dim;
            memset(layer->feed_forward->weight_gradients1, 0, 
                   sizeof(float) * model->model_dim * ff_dim);
            memset(layer->feed_forward->weight_gradients2, 0, 
                   sizeof(float) * ff_dim * model->model_dim);
            memset(layer->feed_forward->bias_gradients1, 0, 
                   sizeof(float) * ff_dim);
            memset(layer->feed_forward->bias_gradients2, 0, 
                   sizeof(float) * model->model_dim);
        }
    }
    
    // 重置解码器梯度（类似编码器的逻辑）
    // ...
}

void update_model_weights(TransformerModel* model, float learning_rate) {
    if (!model) return;
    
    // 更新编码器权重
    for (int i = 0; i < model->num_encoder_layers; i++) {
        EncoderLayer* layer = model->encoder_layers[i];
        if (!layer) continue;
        
        // 更新自注意力权重
        if (layer->self_attention) {
            for (int h = 0; h < layer->self_attention->num_heads; h++) {
                SelfAttention* head = layer->self_attention->attention_heads[h];
                if (!head || !head->requires_grad) continue;
                
                const int weights_size = layer->self_attention->head_dim * 
                                       layer->self_attention->head_dim;
                
                for (int j = 0; j < weights_size; j++) {
                    head->query_weights[j] -= learning_rate * head->query_gradients[j];
                    head->key_weights[j] -= learning_rate * head->key_gradients[j];
                    head->value_weights[j] -= learning_rate * head->value_gradients[j];
                }
            }
        }
        
        // 更新前馈网络权重
        if (layer->feed_forward && layer->feed_forward->requires_grad) {
            const int ff_dim = 4 * model->model_dim;
            
            for (int j = 0; j < model->model_dim * ff_dim; j++) {
                layer->feed_forward->weights1[j] -= 
                    learning_rate * layer->feed_forward->weight_gradients1[j];
            }
            
            for (int j = 0; j < ff_dim * model->model_dim; j++) {
                layer->feed_forward->weights2[j] -= 
                    learning_rate * layer->feed_forward->weight_gradients2[j];
            }
            
            for (int j = 0; j < ff_dim; j++) {
                layer->feed_forward->bias1[j] -= 
                    learning_rate * layer->feed_forward->bias_gradients1[j];
            }
            
            for (int j = 0; j < model->model_dim; j++) {
                layer->feed_forward->bias2[j] -= 
                    learning_rate * layer->feed_forward->bias_gradients2[j];
            }
        }
    }
}