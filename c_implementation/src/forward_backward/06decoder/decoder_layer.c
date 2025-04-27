#include "decoder_layer.h"
#include "tensor_logic.h"
#include "tensor_add.h"
#include <stdlib.h>

DecoderLayer* decoder_layer_create(int num_heads, int model_dim, 
                                 int ff_dim, float dropout_prob) {
    DecoderLayer* layer = (DecoderLayer*)malloc(sizeof(DecoderLayer));
    if (!layer) return NULL;

    // 创建各个子层
    layer->self_attn = multihead_attention_create(num_heads, model_dim);
    layer->cross_attn = multihead_attention_create(num_heads, model_dim);
    layer->norm1 = layer_norm_create(model_dim, 1e-5);
    layer->norm2 = layer_norm_create(model_dim, 1e-5);
    layer->norm3 = layer_norm_create(model_dim, 1e-5);
    layer->ff = feed_forward_create(model_dim, ff_dim);
    layer->dropout_prob = dropout_prob;

    return layer;
}

void decoder_layer_free(DecoderLayer* layer) {
    if (layer) {
        multihead_attention_free(layer->self_attn);
        multihead_attention_free(layer->cross_attn);
        layer_norm_free(layer->norm1);
        layer_norm_free(layer->norm2);
        layer_norm_free(layer->norm3);
        feed_forward_free(layer->ff);
        free(layer);
    }
}

bool decoder_layer_forward(DecoderLayer* layer, Tensor* input, 
                         Tensor* encoder_output, Tensor* output,
                         AttentionMask* self_mask, AttentionMask* cross_mask) {
    // 1. 自注意力子层
    if (!multihead_attention_forward(layer->self_attn, input, output, self_mask)) {
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
    
    // 2. 交叉注意力子层
    Tensor* temp = tensor_create(input->shape, input->num_dims);
    if (!temp) return false;

    
    if (!cross_attention_forward(layer->cross_attn, output, encoder_output, encoder_output, temp, cross_mask)) {
        tensor_free(temp);
        return false;
    }
    
    if (!dropout_forward(temp, temp, layer->dropout_prob)) {
        tensor_free(temp);
        return false;
    }
    
    if (!tensor_add(output, temp, temp)) {  // 残差连接
        tensor_free(temp);
        return false;
    }
    if (!layer_norm_forward(layer->norm2, temp, temp)) {
        tensor_free(temp);
        return false;
    }
    

    // 3. 前馈网络子层

    tensor_copy(output, temp);
    if (!feed_forward_forward(layer->ff, output, output)) {
        tensor_free(temp);
        return false;
    }
    
    if (!dropout_forward(output, output, layer->dropout_prob)) {
        tensor_free(temp);
        return false;
    }
    
    if (!tensor_add(output, temp, output)) {  // 残差连接
        tensor_free(temp);
        return false;
    }
    if (!layer_norm_forward(layer->norm3, output, output)) {
        tensor_free(temp);
        return false;
    }
    
    tensor_free(temp);
    return true;
}


