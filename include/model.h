#ifndef MODEL_H
#define MODEL_H

#include "types.h"
#include "layers.h"
#include "embeddings.h"
#include "attention.h"
#include "training.h"

// 模型管理函数
void free_model(TransformerModel* model);
void reset_model_gradients(TransformerModel* model);
void update_model_weights(TransformerModel* model, float learning_rate);

// 模型初始化函数
TransformerModel* initialize_model(int model_dim, int num_heads, int vocab_size,
                                 int num_encoder_layers, int num_decoder_layers,
                                 int max_seq_length);

// 模型 IO 函数
void save_model(TransformerModel* model, const char* filepath);
TransformerModel* load_model(const char* filepath);

#endif // MODEL_H 