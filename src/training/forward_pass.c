#include "training.h"
#include "utils.h"
#include <stdlib.h>
#include <string.h>

float* forward_pass(TransformerModel* model, float* input, int seq_length) {
    if (!model || !input || seq_length <= 0) return NULL;
    
    // 计算输入嵌入
    float* embeddings = malloc(seq_length * model->model_dim * sizeof(float));
    if (!embeddings) return NULL;
    
    // TODO: 计算实际的嵌入值
    memcpy(embeddings, input, seq_length * model->model_dim * sizeof(float));
    
    // 编码器前向传播
    float* encoder_output = encoder_forward_pass(model, embeddings, seq_length);
    if (!encoder_output) {
        free(embeddings);
        return NULL;
    }
    
    // 解码器前向传播
    float* decoder_output = decoder_forward_pass(model, encoder_output, 
                                               input, seq_length);
    
    // 清理
    free(embeddings);
    free(encoder_output);
    
    return decoder_output;
}

float* encoder_forward_pass(TransformerModel* model, float* input, int seq_length) {
    if (!model || !input || seq_length <= 0) return NULL;
    
    float* current_output = input;
    float* layer_output = NULL;
    
    // 遍历所有编码器层
    for (int i = 0; i < model->num_encoder_layers; i++) {
        EncoderLayer* layer = model->encoder_layers[i];
        
        // 自注意力
        float* attn_output = compute_masked_attention(layer->self_attention,
                                                    current_output, seq_length);
        if (!attn_output) goto cleanup;
        
        // 第一个层归一化
        float* norm_output = malloc(seq_length * model->model_dim * sizeof(float));
        if (!norm_output) {
            free(attn_output);
            goto cleanup;
        }
        
        layer_normalize(norm_output, layer->layer_norm1, NULL,
                       attn_output, seq_length * model->model_dim);
        free(attn_output);
        
        // 前馈网络
        float* ff_output = feed_forward_forward(layer->feed_forward, 
                                              norm_output, seq_length);
        free(norm_output);
        if (!ff_output) goto cleanup;
        
        // 第二个层归一化
        layer_output = malloc(seq_length * model->model_dim * sizeof(float));
        if (!layer_output) {
            free(ff_output);
            goto cleanup;
        }
        
        layer_normalize(layer_output, layer->layer_norm2, NULL,
                       ff_output, seq_length * model->model_dim);
        free(ff_output);
        
        // 更新当前输出
        if (i > 0) free(current_output);
        current_output = layer_output;
    }
    
    return current_output;

cleanup:
    if (layer_output) free(layer_output);
    if (current_output && current_output != input) free(current_output);
    return NULL;
}

float* decoder_forward_pass(TransformerModel* model, float* encoder_output,
                          float* decoder_input, int seq_length) {
    if (!model || !encoder_output || !decoder_input || seq_length <= 0) 
        return NULL;
    
    float* current_output = decoder_input;
    float* layer_output = NULL;
    
    // 遍历所有解码器层
    for (int i = 0; i < model->num_decoder_layers; i++) {
        DecoderLayer* layer = model->decoder_layers[i];
        
        // 带掩码的自注意力
        float* self_attention_output = compute_masked_attention(
            layer->self_attention, current_output, seq_length);
        if (!self_attention_output) goto cleanup;
        
        // 第一个层归一化
        float* norm1_output = malloc(seq_length * model->model_dim * sizeof(float));
        if (!norm1_output) {
            free(self_attention_output);
            goto cleanup;
        }
        
        layer_normalize(norm1_output, layer->layer_norm1, NULL,
                       self_attention_output, seq_length * model->model_dim);
        free(self_attention_output);
        
        // 交叉注意力
        float* cross_attention_output = compute_masked_attention(
            layer->cross_attention, norm1_output, seq_length);
        free(norm1_output);
        if (!cross_attention_output) goto cleanup;
        
        // 第二个层归一化
        float* norm2_output = malloc(seq_length * model->model_dim * sizeof(float));
        if (!norm2_output) {
            free(cross_attention_output);
            goto cleanup;
        }
        
        layer_normalize(norm2_output, layer->layer_norm2, NULL,
                       cross_attention_output, seq_length * model->model_dim);
        free(cross_attention_output);
        
        // 前馈网络
        float* ff_output = feed_forward_forward(layer->feed_forward, 
                                              norm2_output, seq_length);
        free(norm2_output);
        if (!ff_output) goto cleanup;
        
        // 第三个层归一化
        layer_output = malloc(seq_length * model->model_dim * sizeof(float));
        if (!layer_output) {
            free(ff_output);
            goto cleanup;
        }
        
        layer_normalize(layer_output, layer->layer_norm3, NULL,
                       ff_output, seq_length * model->model_dim);
        free(ff_output);
        
        // 更新当前输出
        if (i > 0) free(current_output);
        current_output = layer_output;
    }
    
    return current_output;

cleanup:
    if (layer_output) free(layer_output);
    if (current_output && current_output != decoder_input) free(current_output);
    return NULL;
} 