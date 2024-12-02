#include "layers.h"
#include <stdlib.h>
#include <string.h>

EncoderLayer* encoder_layer_create(int model_dim, int num_heads, int ff_hidden_dim, bool requires_grad) {
    EncoderLayer* encoder = (EncoderLayer*)malloc(sizeof(EncoderLayer));
    if (!encoder) return NULL;

    encoder->model_dim = model_dim;

    // 创建多头自注意力层
    encoder->self_attention = multihead_attention_create(model_dim, num_heads, requires_grad);
    
    // 创建两个层归一化
    encoder->norm1 = layer_norm_create(model_dim, 1e-5, requires_grad);
    encoder->norm2 = layer_norm_create(model_dim, 1e-5, requires_grad);
    
    // 创建前馈网络
    encoder->feed_forward = feed_forward_create(model_dim, ff_hidden_dim, requires_grad);

    // 检查所有组件是否创建成功
    if (!encoder->self_attention || !encoder->norm1 || 
        !encoder->norm2 || !encoder->feed_forward) {
        encoder_layer_free(encoder);
        return NULL;
    }

    return encoder;
}

void encoder_layer_free(EncoderLayer* encoder) {
    if (encoder) {
        if (encoder->self_attention) {
            multihead_attention_free(encoder->self_attention);
        }
        if (encoder->norm1) {
            layer_norm_free(encoder->norm1);
        }
        if (encoder->norm2) {
            layer_norm_free(encoder->norm2);
        }
        if (encoder->feed_forward) {
            feed_forward_free(encoder->feed_forward);
        }
        free(encoder);
    }
}

void encoder_layer_forward(
    EncoderLayer* encoder,
    float* input,        // [batch_size, seq_len, model_dim]
    int batch_size,
    int seq_len,
    float* output       // [batch_size, seq_len, model_dim]
) {
    // 分配临时缓冲区
    int batch_seq_dim = batch_size * seq_len * encoder->model_dim;
    float* attn_output = (float*)malloc(batch_seq_dim * sizeof(float));
    float* norm1_output = (float*)malloc(batch_seq_dim * sizeof(float));
    float* ff_output = (float*)malloc(batch_seq_dim * sizeof(float));
    
    if (!attn_output || !norm1_output || !ff_output) {
        free(attn_output);
        free(norm1_output);
        free(ff_output);
        return;
    }

    // 1. 自注意力层
    multihead_attention_forward(
        encoder->self_attention,
        input,          // query
        seq_len,
        attn_output,
        NULL           // mask参数
    );

    // 2. 第一个残差连接和层归一化
    // 首先添加残差连接
    for (int i = 0; i < batch_seq_dim; i++) {
        attn_output[i] += input[i];
    }
    
    // 应用层归一化
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_len; s++) {
            layer_norm_forward(
                encoder->norm1,
                &attn_output[(b * seq_len + s) * encoder->model_dim],
                1,
                &norm1_output[(b * seq_len + s) * encoder->model_dim]
            );
        }
    }

    // 3. 前馈网络
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_len; s++) {
            feed_forward_forward(
                encoder->feed_forward,
                &norm1_output[(b * seq_len + s) * encoder->model_dim],
                1,
                &ff_output[(b * seq_len + s) * encoder->model_dim]
            );
        }
    }

    // 4. 第二个残差连接和层归一化
    // 添加残差连接
    for (int i = 0; i < batch_seq_dim; i++) {
        ff_output[i] += norm1_output[i];
    }
    
    // 最终的层归一化
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_len; s++) {
            layer_norm_forward(
                encoder->norm2,
                &ff_output[(b * seq_len + s) * encoder->model_dim],
                1,
                &output[(b * seq_len + s) * encoder->model_dim]
            );
        }
    }

    // 释放临时缓冲区
    free(attn_output);
    free(norm1_output);
    free(ff_output);
}