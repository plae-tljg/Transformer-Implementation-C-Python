#ifndef LAYERS_H
#define LAYERS_H

#include "types.h"
#include <stdbool.h>

// 层归一化函数
LayerNorm* initialize_layer_norm(int dim);
void free_layer_norm(LayerNorm* ln);
float* layer_norm_forward(LayerNorm* ln, float* input, int seq_length);
float* layer_norm_backward(LayerNorm* ln, float* grad_output, int seq_length);

// 层初始化函数
EncoderLayer* initialize_encoder_layer(int model_dim, int num_heads);
DecoderLayer* initialize_decoder_layer(int model_dim, int num_heads);

// 层释放函数
void free_encoder_layer(EncoderLayer* layer);
void free_decoder_layer(DecoderLayer* layer);

// 前馈网络函数
FeedForward* initialize_feed_forward(int input_dim, int hidden_dim);
void free_feed_forward(FeedForward* ff);
float* feed_forward_forward(FeedForward* ff, float* input, int seq_length);

// 多头注意力函数
MultiHeadAttention* initialize_multi_head_attention(int model_dim, int num_heads);
void free_multi_head_attention(MultiHeadAttention* mha);
float* multi_head_attention_forward(MultiHeadAttention* mha, float* query, float* key, float* value, int seq_length);

#endif // LAYERS_H 