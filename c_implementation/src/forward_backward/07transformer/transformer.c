#include "transformer.h"
#include <stdlib.h>

Transformer* transformer_create(int num_layers, int num_heads, int model_dim,
                              int ff_dim, float dropout_prob) {
    Transformer* transformer = (Transformer*)malloc(sizeof(Transformer));
    if (!transformer) return NULL;
    
    // 保存配置
    transformer->model_dim = model_dim;
    transformer->num_heads = num_heads;
    transformer->num_layers = num_layers;
    transformer->ff_dim = ff_dim;
    transformer->dropout_prob = dropout_prob;
    
    // 创建编码器
    transformer->encoder = encoder_create(num_layers, num_heads, model_dim,
                                       ff_dim, dropout_prob);
    if (!transformer->encoder) {
        transformer_free(transformer);
        return NULL;
    }
    
    // 创建解码器
    transformer->decoder = decoder_create(num_layers, num_heads, model_dim,
                                       ff_dim, dropout_prob);
    if (!transformer->decoder) {
        transformer_free(transformer);
        return NULL;
    }
    
    return transformer;
}

bool transformer_forward(Transformer* transformer,
                       Tensor* encoder_input, Tensor* decoder_input,
                       Tensor* output,
                       AttentionMask* enc_mask, AttentionMask* dec_mask,
                       AttentionMask* cross_mask) {
    if (!transformer || !encoder_input || !decoder_input || !output) {
        return false;
    }
    
    // 创建一个临时张量存储编码器输出
    Tensor* encoder_output = tensor_create(encoder_input->shape, encoder_input->num_dims);
    if (!encoder_output) return false;
    
    // 编码器前向传播
    if (!encoder_forward(transformer->encoder, encoder_input, encoder_output, enc_mask)) {
        tensor_free(encoder_output);
        return false;
    }
    
    // 解码器前向传播
    if (!decoder_forward(transformer->decoder, decoder_input, encoder_output,
                        output, dec_mask, cross_mask)) {
        tensor_free(encoder_output);
        return false;
    }
    
    tensor_free(encoder_output);
    return true;
}

void transformer_free(Transformer* transformer) {
    if (transformer) {
        encoder_free(transformer->encoder);
        decoder_free(transformer->decoder);
        free(transformer);
    }
}