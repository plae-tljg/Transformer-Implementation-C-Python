#include <float.h>
#include <stdlib.h>
#include "layers.h"
#include "grad.h"

void encoder_layer_backward(
    EncoderLayer* encoder,
    float* input,
    float* grad_output,
    int batch_size,
    int seq_len,
    EncoderLayerGrad* grad
) {
    int model_dim = encoder->model_dim;
    int batch_seq_dim = batch_size * seq_len * model_dim;
    
    // 分配临时内存
    float* grad_norm2 = (float*)malloc(batch_seq_dim * sizeof(float));
    float* grad_ff = (float*)malloc(batch_seq_dim * sizeof(float));
    float* grad_norm1 = (float*)malloc(batch_seq_dim * sizeof(float));
    float* grad_attn = (float*)malloc(batch_seq_dim * sizeof(float));
    
    // 1. 反向传播通过第二个LayerNorm
    layer_norm_backward(
        encoder->norm2,
        input,
        grad_output,
        batch_size,
        grad_norm2,
        grad->norm2_grad->grad_gamma,
        grad->norm2_grad->grad_beta
    );
    
    // 2. 反向传播通过FeedForward
    feed_forward_backward(
        encoder->feed_forward,
        input,
        grad_norm2,
        batch_size,
        grad_ff,
        grad->feed_forward_grad->grad_w1,
        grad->feed_forward_grad->grad_b1,
        grad->feed_forward_grad->grad_w2,
        grad->feed_forward_grad->grad_b2
    );
    
    // 3. 反向传播通过第一个LayerNorm
    layer_norm_backward(
        encoder->norm1,
        input,
        grad_ff,
        batch_size,
        grad_norm1,
        grad->norm1_grad->grad_gamma,
        grad->norm1_grad->grad_beta
    );
    
    // 4. 反向传播通过自注意力层
    self_attention_backward(
        encoder->self_attention,
        input,
        grad_norm1,
        NULL,  // 无掩码
        seq_len,
        grad->self_attention_grad
    );
    
    // 释放临时内存
    free(grad_norm2);
    free(grad_ff);
    free(grad_norm1);
    free(grad_attn);
} 