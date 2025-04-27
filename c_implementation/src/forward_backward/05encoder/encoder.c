#include "encoder.h"
#include <stdlib.h>

Encoder* encoder_create(int num_layers, int num_heads, int model_dim, 
                       int ff_dim, float dropout_prob) {
    Encoder* encoder = (Encoder*)malloc(sizeof(Encoder));
    if (!encoder) return NULL;

    encoder->num_layers = num_layers;
    encoder->layers = (EncoderLayer**)malloc(sizeof(EncoderLayer*) * num_layers);
    
    for (int i = 0; i < num_layers; i++) {
        encoder->layers[i] = encoder_layer_create(num_heads, model_dim, 
                                                ff_dim, dropout_prob);
        if (!encoder->layers[i]) {
            encoder_free(encoder);
            return NULL;
        }
    }
    
    return encoder;
}

bool encoder_forward(Encoder* encoder, Tensor* input, Tensor* output, 
                    AttentionMask* mask) {
    // 第一层使用input作为输入
    if (!encoder_layer_forward(encoder->layers[0], input, output, mask)) {
        return false;
    }
    
    // 后续层使用前一层的输出作为输入
    for (int i = 1; i < encoder->num_layers; i++) {
        if (!encoder_layer_forward(encoder->layers[i], output, output, mask)) {
            return false;
        }
    }
    
    return true;
}

void encoder_free(Encoder* encoder) {
    if (encoder) {
        for (int i = 0; i < encoder->num_layers; i++) {
            encoder_layer_free(encoder->layers[i]);
        }
        free(encoder->layers);
        free(encoder);
    }
}
