#include "decoder.h"
#include <stdlib.h>
#include <stdio.h>

Decoder* decoder_create(int num_layers, int num_heads, int model_dim, 
                       int ff_dim, float dropout_prob) {
    Decoder* decoder = (Decoder*)malloc(sizeof(Decoder));
    if (!decoder) return NULL;

    decoder->num_layers = num_layers;
    decoder->layers = (DecoderLayer**)malloc(sizeof(DecoderLayer*) * num_layers);
    if (!decoder->layers) {
        free(decoder);
        return NULL;
    }
    
    // 创建每一层解码器
    for (int i = 0; i < num_layers; i++) {
        decoder->layers[i] = decoder_layer_create(num_heads, model_dim, 
                                                ff_dim, dropout_prob);
        if (!decoder->layers[i]) {
            decoder_free(decoder);
            return NULL;
        }
    }
    
    return decoder;
}

bool decoder_forward(
    Decoder* decoder,
    Tensor* input,           // [batch_size, seq_len, model_dim] 解码器输入
    Tensor* encoder_output,  // [batch_size, enc_seq_len, model_dim] 编码器输出
    Tensor* output,          // [batch_size, seq_len, model_dim] 解码器输出
    AttentionMask* self_mask,    // [batch_size, num_heads, seq_len, seq_len] 自注意力掩码
    AttentionMask* cross_mask    // [batch_size, num_heads, seq_len, enc_seq_len] 交叉注意力掩码
) {
    if (!decoder || !input || !encoder_output || !output) {
        return false;
    }

    // 第一层使用input作为输入
    if (!decoder_layer_forward(decoder->layers[0], input, encoder_output,
                             output, self_mask, cross_mask)) {
        return false;
    }
    
    // 后续层使用前一层的输出作为输入
    for (int i = 1; i < decoder->num_layers; i++) {
        if (!decoder_layer_forward(decoder->layers[i], output, encoder_output,
                                 output, self_mask, cross_mask)) {
            return false;
        }
    }
    
    return true;
}

void decoder_free(Decoder* decoder) {
    if (decoder) {
        if (decoder->layers) {
            for (int i = 0; i < decoder->num_layers; i++) {
                decoder_layer_free(decoder->layers[i]);
            }
            free(decoder->layers);
        }
        free(decoder);
    }
}