#include "layers.h"
#include <stdlib.h>

DecoderLayer* initialize_decoder_layer(int model_dim, int num_heads) {
    DecoderLayer* layer = malloc(sizeof(DecoderLayer));
    if (!layer) return NULL;
    
    layer->model_dim = model_dim;
    
    // 初始化自注意力层
    layer->self_attention = initialize_multi_head_attention(model_dim, num_heads);
    if (!layer->self_attention) {
        free(layer);
        return NULL;
    }
    
    // 初始化交叉注意力层
    layer->cross_attention = initialize_multi_head_attention(model_dim, num_heads);
    if (!layer->cross_attention) {
        free_decoder_layer(layer);
        return NULL;
    }
    
    // 初始化层归一化
    layer->norm1 = initialize_layer_norm(model_dim);
    layer->norm2 = initialize_layer_norm(model_dim);
    layer->norm3 = initialize_layer_norm(model_dim);
    if (!layer->norm1 || !layer->norm2 || !layer->norm3) {
        free_decoder_layer(layer);
        return NULL;
    }
    
    // 初始化前馈网络
    layer->feed_forward = initialize_feed_forward(model_dim, 4 * model_dim);
    if (!layer->feed_forward) {
        free_decoder_layer(layer);
        return NULL;
    }
    
    return layer;
}

void free_decoder_layer(DecoderLayer* layer) {
    if (!layer) return;
    
    if (layer->self_attention) free_multi_head_attention(layer->self_attention);
    if (layer->cross_attention) free_multi_head_attention(layer->cross_attention);
    if (layer->norm1) free_layer_norm(layer->norm1);
    if (layer->norm2) free_layer_norm(layer->norm2);
    if (layer->norm3) free_layer_norm(layer->norm3);
    if (layer->feed_forward) free_feed_forward(layer->feed_forward);
    
    free(layer);
}