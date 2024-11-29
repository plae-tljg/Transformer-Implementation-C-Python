#ifndef TRAINING_H
#define TRAINING_H

#include "types.h"
#include "layers.h"
#include "embeddings.h"
#include "attention.h"
#include "optimizer.h"
#include <stdbool.h>

// 函数声明
void free_model(TransformerModel* model);
TransformerModel* initialize_model(int model_dim, int num_heads, int vocab_size,
                                 int num_encoder_layers, int num_decoder_layers,
                                 float dropout_rate);
TrainingConfig* create_training_config(void);
void free_training_config(TrainingConfig* config);

// 反向传播函数声明
float* decoder_backward_pass(TransformerModel* model, float* output_grad);
float* encoder_backward_pass(TransformerModel* model, float* decoder_grad);
void update_layer_gradients(DecoderLayer* layer, 
                          float* self_attn_grad,
                          float* cross_attn_grad,
                          float* ff_grad);
void update_attention_gradients(MultiHeadAttention* mha, float learning_rate);
void update_self_attention_gradients(SelfAttention* attn, float learning_rate);
void update_feed_forward_gradients(FeedForward* ff, float learning_rate);
void update_embedding_gradients(TokenEmbedding* embedding, float* gradients);

// 注意力反向传播
float* attention_backward(MultiHeadAttention* attn, float* grad_output);
float* masked_attention_backward(MultiHeadAttention* attn, float* grad_output);

// 前向传播函数声明
float* forward_pass(TransformerModel* model, float* input, int seq_length);
float* encoder_forward_pass(TransformerModel* model, float* input, int seq_length);
float* decoder_forward_pass(TransformerModel* model, float* encoder_output,
                          float* decoder_input, int seq_length);
void layer_normalize(float* output, float* weights, float* bias,
                    float* input, int size);

// 训练相关函数声明
void train_epoch(TransformerModel* model, float** input_data, float** target_data,
                int num_samples, AdamOptimizer* optimizer);
float compute_loss(float* predictions, float* targets, int seq_length);
void backward_pass(TransformerModel* model, float* loss_grad);
void apply_gradients(TransformerModel* model, AdamOptimizer* optimizer);

// 前馈网络训练函数
void feed_forward_backward(FeedForward* ff, float* grad_output, float* input);
void update_feed_forward_gradients(FeedForward* ff, float learning_rate);
void accumulate_feed_forward_gradients(FeedForward* ff, float* grad);

// 添加一个新的函数用于最终的权重更新
void apply_layer_gradients(DecoderLayer* layer, float learning_rate);

// 注意力层的梯度更新函数
void update_attention_gradients(MultiHeadAttention* mha, float learning_rate);
void update_self_attention_gradients(SelfAttention* attn, float learning_rate);
void accumulate_attention_gradients(MultiHeadAttention* mha, float* grad);

// 损失函数相关
float compute_loss(float* predictions, float* targets, int size);
float* compute_loss_gradient(float* output, float* target, int size);

#endif // TRAINING_H 