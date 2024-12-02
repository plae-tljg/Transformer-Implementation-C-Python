#include "layers.h"
#include "embeddings.h"
#include <stdlib.h>
#include <string.h>

DecoderLayer* decoder_layer_create(int model_dim, int num_heads, int ff_hidden_dim, bool requires_grad) {
    DecoderLayer* decoder = (DecoderLayer*)malloc(sizeof(DecoderLayer));
    if (!decoder) return NULL;

    decoder->model_dim = model_dim;

    // 创建自注意力层
    decoder->self_attention = multihead_attention_create(model_dim, num_heads, requires_grad);
    
    // 创建交叉注意力层
    decoder->cross_attention = multihead_attention_create(model_dim, num_heads, requires_grad);
    
    // 创建三个层归一化
    decoder->norm1 = layer_norm_create(model_dim, 1e-5, requires_grad);
    decoder->norm2 = layer_norm_create(model_dim, 1e-5, requires_grad);
    decoder->norm3 = layer_norm_create(model_dim, 1e-5, requires_grad);
    
    // 创建前馈网络
    decoder->feed_forward = feed_forward_create(model_dim, ff_hidden_dim, requires_grad);

    // 检查所有组件是否创建成功
    if (!decoder->self_attention || !decoder->cross_attention || 
        !decoder->norm1 || !decoder->norm2 || !decoder->norm3 || 
        !decoder->feed_forward) {
        decoder_layer_free(decoder);
        return NULL;
    }

    return decoder;
}

void decoder_layer_free(DecoderLayer* decoder) {
    if (decoder) {
        if (decoder->self_attention) {
            multihead_attention_free(decoder->self_attention);
        }
        if (decoder->cross_attention) {
            multihead_attention_free(decoder->cross_attention);
        }
        if (decoder->norm1) {
            layer_norm_free(decoder->norm1);
        }
        if (decoder->norm2) {
            layer_norm_free(decoder->norm2);
        }
        if (decoder->norm3) {
            layer_norm_free(decoder->norm3);
        }
        if (decoder->feed_forward) {
            feed_forward_free(decoder->feed_forward);
        }
        free(decoder);
    }
}

void decoder_layer_forward(
    DecoderLayer* decoder,
    float* input,          // [batch_size, tgt_len, model_dim]
    float* encoder_output, // [batch_size, src_len, model_dim]
    float* tgt_mask,      // [batch_size, tgt_len, tgt_len]
    int batch_size,
    int tgt_len,
    int src_len,
    float* output         // [batch_size, tgt_len, model_dim]
) {
    // 分配临时缓冲区
    int batch_tgt_dim = batch_size * tgt_len * decoder->model_dim;
    float* self_attn_output = (float*)malloc(batch_tgt_dim * sizeof(float));
    float* norm1_output = (float*)malloc(batch_tgt_dim * sizeof(float));
    float* cross_attn_output = (float*)malloc(batch_tgt_dim * sizeof(float));
    float* norm2_output = (float*)malloc(batch_tgt_dim * sizeof(float));
    float* ff_output = (float*)malloc(batch_tgt_dim * sizeof(float));
    
    if (!self_attn_output || !norm1_output || !cross_attn_output || 
        !norm2_output || !ff_output) {
        free(self_attn_output);
        free(norm1_output);
        free(cross_attn_output);
        free(norm2_output);
        free(ff_output);
        return;
    }

    // 1. 掩码自注意力层
    multihead_attention_forward(
        decoder->self_attention,
        input,           // query
        tgt_len,         // seq_length
        self_attn_output,
        tgt_mask           // mask
    );

    // 2. 第一个残差连接和层归一化
    for (int i = 0; i < batch_tgt_dim; i++) {
        self_attn_output[i] += input[i];
    }
    
    for (int b = 0; b < batch_size; b++) {
        for (int t = 0; t < tgt_len; t++) {
            layer_norm_forward(
                decoder->norm1,
                &self_attn_output[(b * tgt_len + t) * decoder->model_dim],
                1,
                &norm1_output[(b * tgt_len + t) * decoder->model_dim]
            );
        }
    }

    // 3. 交叉注意力层
    multihead_attention_forward(
        decoder->cross_attention,
        norm1_output,    // query
        src_len,         // seq_length
        cross_attn_output,
        NULL           // 交叉注意力不需要mask
    );

    // 4. 第二个残差连接和层归一化
    for (int i = 0; i < batch_tgt_dim; i++) {
        cross_attn_output[i] += norm1_output[i];
    }
    
    for (int b = 0; b < batch_size; b++) {
        for (int t = 0; t < tgt_len; t++) {
            layer_norm_forward(
                decoder->norm2,
                &cross_attn_output[(b * tgt_len + t) * decoder->model_dim],
                1,
                &norm2_output[(b * tgt_len + t) * decoder->model_dim]
            );
        }
    }

    // 5. 前馈网络
    for (int b = 0; b < batch_size; b++) {
        for (int t = 0; t < tgt_len; t++) {
            feed_forward_forward(
                decoder->feed_forward,
                &norm2_output[(b * tgt_len + t) * decoder->model_dim],
                1,
                &ff_output[(b * tgt_len + t) * decoder->model_dim]
            );
        }
    }

    // 6. 第三个残差连接和层归一化
    for (int i = 0; i < batch_tgt_dim; i++) {
        ff_output[i] += norm2_output[i];
    }
    
    for (int b = 0; b < batch_size; b++) {
        for (int t = 0; t < tgt_len; t++) {
            layer_norm_forward(
                decoder->norm3,
                &ff_output[(b * tgt_len + t) * decoder->model_dim],
                1,
                &output[(b * tgt_len + t) * decoder->model_dim]
            );
        }
    }

    // 释放临时缓冲区
    free(self_attn_output);
    free(norm1_output);
    free(cross_attn_output);
    free(norm2_output);
    free(ff_output);
}