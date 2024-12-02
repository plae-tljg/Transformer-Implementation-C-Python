#include "model.h"
#include "layers.h"
#include <stdio.h>

bool load_model_parameters(TransformerModel* model, FILE* file) {
    // 临时实现：创建一个新的 transformer
    Transformer* transformer = transformer_create(
        model->vocab_size,
        model->model_dim,
        model->num_heads,
        model->num_encoder_layers,
        model->encoder_layers[0]->feed_forward->hidden_dim,  // 从第一个编码器层获取 hidden_dim
        false  // requires_grad 设为 false，因为是推理模式
    );
    
    if (!transformer) {
        return false;
    }
    
    // 将创建的 transformer 赋值给各个层
    for (int i = 0; i < model->num_encoder_layers; i++) {
        model->encoder_layers[i] = transformer->encoder->layers[i];
    }
    for (int i = 0; i < model->num_decoder_layers; i++) {
        model->decoder_layers[i] = transformer->decoder->layers[i];
    }
    
    return true;
} 