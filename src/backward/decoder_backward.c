#include "layers.h"
#include "grad.h"
#include <stdlib.h>

void decoder_layer_backward(
    DecoderLayer* decoder,
    float* input,
    float* grad_output,
    float* encoder_output,
    int batch_size,
    int tgt_len,
    int src_len,
    DecoderLayerGrad* grad
) {
    int model_dim = decoder->model_dim;
    int batch_tgt_dim = batch_size * tgt_len * model_dim;
    
    // 分配临时内存
    float* grad_norm3 = (float*)malloc(batch_tgt_dim * sizeof(float));
    float* grad_ff = (float*)malloc(batch_tgt_dim * sizeof(float));
    float* grad_norm2 = (float*)malloc(batch_tgt_dim * sizeof(float));
    float* grad_cross = (float*)malloc(batch_tgt_dim * sizeof(float));
    float* grad_norm1 = (float*)malloc(batch_tgt_dim * sizeof(float));
    float* grad_self = (float*)malloc(batch_tgt_dim * sizeof(float));
    
    // 1. 反向传播通过第三个LayerNorm
    layer_norm_backward(
        decoder->norm3,
        input,
        grad_output,
        batch_size,
        grad_norm3,
        grad->norm3_grad->grad_gamma,
        grad->norm3_grad->grad_beta
    );
    
    // 2. 反向传播通过FeedForward
    feed_forward_backward(
        decoder->feed_forward,
        input,
        grad_norm3,
        batch_size,
        grad_ff,
        grad->feed_forward_grad
    );
    
    // 3. 反向传播通过第二个LayerNorm
    layer_norm_backward(
        decoder->norm2,
        input,
        grad_ff,
        batch_size,
        grad_norm2,
        grad->norm2_grad->grad_gamma,
        grad->norm2_grad->grad_beta
    );
    
    // 4. 反向传播通过交叉注意力
    self_attention_backward(
        decoder->cross_attention,
        input,
        grad_norm2,
        NULL,
        src_len,
        grad->cross_attention_grad
    );
    
    // 5. 反向传播通过第一个LayerNorm
    layer_norm_backward(
        decoder->norm1,
        input,
        grad_cross,
        batch_size,
        grad_norm1,
        grad->norm1_grad->grad_gamma,
        grad->norm1_grad->grad_beta
    );
    
    // 6. 反向传播通过自注意力
    self_attention_backward(
        decoder->self_attention,
        input,
        grad_norm1,
        NULL,  // 这里应该传入实际的掩码
        tgt_len,
        grad->self_attention_grad
    );
    
    // 释放临时内存
    free(grad_norm3);
    free(grad_ff);
    free(grad_norm2);
    free(grad_cross);
    free(grad_norm1);
    free(grad_self);
} 