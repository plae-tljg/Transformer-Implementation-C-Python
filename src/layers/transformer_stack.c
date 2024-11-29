#include "layers.h"
#include <stdlib.h>
#include <string.h>

// 编码器堆栈实现
Encoder* encoder_create(int num_layers, int model_dim, int num_heads, int ff_hidden_dim, bool requires_grad) {
    Encoder* encoder = (Encoder*)malloc(sizeof(Encoder));
    if (!encoder) return NULL;

    encoder->num_layers = num_layers;
    encoder->model_dim = model_dim;
    
    // 分配层数组内存
    encoder->layers = (EncoderLayer**)malloc(num_layers * sizeof(EncoderLayer*));
    if (!encoder->layers) {
        free(encoder);
        return NULL;
    }

    // 创建每一层
    for (int i = 0; i < num_layers; i++) {
        encoder->layers[i] = encoder_layer_create(model_dim, num_heads, ff_hidden_dim, requires_grad);
        if (!encoder->layers[i]) {
            encoder_free(encoder);
            return NULL;
        }
    }

    return encoder;
}

void encoder_free(Encoder* encoder) {
    if (encoder) {
        if (encoder->layers) {
            for (int i = 0; i < encoder->num_layers; i++) {
                if (encoder->layers[i]) {
                    encoder_layer_free(encoder->layers[i]);
                }
            }
            free(encoder->layers);
        }
        free(encoder);
    }
}

void encoder_forward(
    Encoder* encoder,
    float* input,        // [batch_size, src_len, model_dim]
    int batch_size,
    int src_len,
    float* output       // [batch_size, src_len, model_dim]
) {
    int layer_size = batch_size * src_len * encoder->model_dim;
    float* layer_input = (float*)malloc(layer_size * sizeof(float));
    float* layer_output = (float*)malloc(layer_size * sizeof(float));
    
    if (!layer_input || !layer_output) {
        free(layer_input);
        free(layer_output);
        return;
    }

    // 复制输入到临时缓冲区
    memcpy(layer_input, input, layer_size * sizeof(float));

    // 依次通过每一层
    for (int i = 0; i < encoder->num_layers; i++) {
        encoder_layer_forward(
            encoder->layers[i],
            layer_input,
            batch_size,
            src_len,
            layer_output
        );
        
        // 交换输入输出缓冲区
        float* temp = layer_input;
        layer_input = layer_output;
        layer_output = temp;
    }

    // 复制最终结果到输出
    memcpy(output, layer_input, layer_size * sizeof(float));

    free(layer_input);
    free(layer_output);
}

// 解码器堆栈实现
Decoder* decoder_create(int num_layers, int model_dim, int num_heads, int ff_hidden_dim, bool requires_grad) {
    Decoder* decoder = (Decoder*)malloc(sizeof(Decoder));
    if (!decoder) return NULL;

    decoder->num_layers = num_layers;
    decoder->model_dim = model_dim;
    
    // 分配层数组内存
    decoder->layers = (DecoderLayer**)malloc(num_layers * sizeof(DecoderLayer*));
    if (!decoder->layers) {
        free(decoder);
        return NULL;
    }

    // 创建每一层
    for (int i = 0; i < num_layers; i++) {
        decoder->layers[i] = decoder_layer_create(model_dim, num_heads, ff_hidden_dim, requires_grad);
        if (!decoder->layers[i]) {
            decoder_free(decoder);
            return NULL;
        }
    }

    return decoder;
}

void decoder_free(Decoder* decoder) {
    if (decoder) {
        if (decoder->layers) {
            for (int i = 0; i < decoder->num_layers; i++) {
                if (decoder->layers[i]) {
                    decoder_layer_free(decoder->layers[i]);
                }
            }
            free(decoder->layers);
        }
        free(decoder);
    }
}

void decoder_forward(
    Decoder* decoder,
    float* input,          // [batch_size, tgt_len, model_dim]
    float* encoder_output, // [batch_size, src_len, model_dim]
    float* tgt_mask,      // [batch_size, tgt_len, tgt_len]
    int batch_size,
    int tgt_len,
    int src_len,
    float* output         // [batch_size, tgt_len, model_dim]
) {
    int layer_size = batch_size * tgt_len * decoder->model_dim;
    float* layer_input = (float*)malloc(layer_size * sizeof(float));
    float* layer_output = (float*)malloc(layer_size * sizeof(float));
    
    if (!layer_input || !layer_output) {
        free(layer_input);
        free(layer_output);
        return;
    }

    // 复制输入到临时缓冲区
    memcpy(layer_input, input, layer_size * sizeof(float));

    // 依次通过每一层
    for (int i = 0; i < decoder->num_layers; i++) {
        decoder_layer_forward(
            decoder->layers[i],
            layer_input,
            encoder_output,
            tgt_mask,
            batch_size,
            tgt_len,
            src_len,
            layer_output
        );
        
        // 交换输入输出缓冲区
        float* temp = layer_input;
        layer_input = layer_output;
        layer_output = temp;
    }

    // 复制最终结果到输出
    memcpy(output, layer_input, layer_size * sizeof(float));

    free(layer_input);
    free(layer_output);
}