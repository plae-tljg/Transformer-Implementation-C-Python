#include "layers.h"
#include <stdlib.h>

EncoderLayer* initialize_encoder_layer(int model_dim, int num_heads) {
    EncoderLayer* layer = malloc(sizeof(EncoderLayer));
    if (!layer) return NULL;
    
    layer->model_dim = model_dim;
    
    // 初始化自注意力层
    layer->self_attention = initialize_multi_head_attention(model_dim, num_heads);
    if (!layer->self_attention) {
        free(layer);
        return NULL;
    }
    
    // 初始化层归一化
    layer->norm1 = initialize_layer_norm(model_dim);
    layer->norm2 = initialize_layer_norm(model_dim);
    if (!layer->norm1 || !layer->norm2) {
        free_encoder_layer(layer);
        return NULL;
    }
    
    // 初始化前馈网络
    layer->feed_forward = initialize_feed_forward(model_dim, 4 * model_dim);
    if (!layer->feed_forward) {
        free_encoder_layer(layer);
        return NULL;
    }
    
    return layer;
}

void free_encoder_layer(EncoderLayer* layer) {
    if (!layer) return;
    
    if (layer->self_attention) free_multi_head_attention(layer->self_attention);
    if (layer->norm1) free_layer_norm(layer->norm1);
    if (layer->norm2) free_layer_norm(layer->norm2);
    if (layer->feed_forward) free_feed_forward(layer->feed_forward);
    
    free(layer);
} 