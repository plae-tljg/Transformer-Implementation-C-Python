#include "encoder_layer.h"
#include "layer_norm.h"
#include "tensor_add.h"
#include "tensor_logic.h"
#include <stdlib.h>

EncoderLayer* encoder_layer_create(int num_heads, int model_dim, int ff_dim, float dropout_prob) {
    EncoderLayer* layer = (EncoderLayer*)malloc(sizeof(EncoderLayer));
    if (!layer) return NULL;

    layer->self_attn = multihead_attention_create(num_heads, model_dim);
    layer->norm1 = layer_norm_create(model_dim, 1e-5);
    layer->ff = feed_forward_create(model_dim, ff_dim);
    layer->norm2 = layer_norm_create(model_dim, 1e-5);
    layer->dropout_prob = dropout_prob;

    return layer;
}

bool encoder_layer_forward(EncoderLayer* layer, Tensor* input, Tensor* output, 
                         AttentionMask* mask) {
    // 1. 自注意力子层
    if (!multihead_attention_forward(layer->self_attn, input, output, mask)) {
        return false;
    }
    
    // Dropout
    if (!dropout_forward(output, output, layer->dropout_prob)) {
        return false;
    }
    
    // 残差连接和层归一化
    if (!tensor_add(output, input, output)) {  // 残差连接
        return false;
    }
    if (!layer_norm_forward(layer->norm1, output, output)) {
        return false;
    }
    
    // 2. 前馈网络子层
    Tensor* ff_output = tensor_create(output->shape, output->num_dims);
    if (!ff_output) return false;
    if (!feed_forward_forward(layer->ff, output, ff_output)) {
        tensor_free(ff_output);
        return false;
    }
    
    // Dropout
    if (!dropout_forward(ff_output, ff_output, layer->dropout_prob)) {
        tensor_free(ff_output);
        return false;
    }
    
    // 残差连接和层归一化
    if (!tensor_add(ff_output, output, output)) {
        tensor_free(ff_output);
        return false;
    }
    if (!layer_norm_forward(layer->norm2, output, output)) {
        tensor_free(ff_output);
        return false;
    }
    
    tensor_free(ff_output);
    return true;
}

void encoder_layer_free(EncoderLayer* layer) {
    if (layer) {
        multihead_attention_free(layer->self_attn);
        layer_norm_free(layer->norm1);
        feed_forward_free(layer->ff);
        layer_norm_free(layer->norm2);
        free(layer);
    }
}
