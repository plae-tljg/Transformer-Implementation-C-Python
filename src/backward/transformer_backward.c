#include "layers.h"
#include "grad.h"
#include <stdlib.h>

void transformer_backward(
    Transformer* model,
    float* loss_grad,      // [batch_size, seq_len, vocab_size]
    float* encoder_output, // [batch_size, src_len, model_dim]
    float* decoder_output, // [batch_size, tgt_len, model_dim]
    int batch_size,
    int src_len,
    int tgt_len,
    TransformerGrad* grad
) {
    // 1. 输出层的反向传播
    float* grad_decoder = (float*)malloc(batch_size * tgt_len * model->model_dim * sizeof(float));
    
    // 计算输出层的梯度
    for (int b = 0; b < batch_size; b++) {
        for (int t = 0; t < tgt_len; t++) {
            for (int d = 0; d < model->model_dim; d++) {
                float sum = 0.0f;
                for (int v = 0; v < model->vocab_size; v++) {
                    sum += loss_grad[(b * tgt_len + t) * model->vocab_size + v] *
                           model->linear_weight[v * model->model_dim + d];
                    grad->grad_linear_weight[v * model->model_dim + d] +=
                        loss_grad[(b * tgt_len + t) * model->vocab_size + v] *
                        decoder_output[(b * tgt_len + t) * model->model_dim + d];
                }
                grad_decoder[(b * tgt_len + t) * model->model_dim + d] = sum;
            }
        }
    }
    
    // 2. 解码器层的反向传播
    for (int l = model->decoder->num_layers - 1; l >= 0; l--) {
        decoder_layer_backward(
            model->decoder->layers[l],
            decoder_output,
            grad_decoder,
            encoder_output,
            batch_size,
            tgt_len,
            src_len,
            grad->decoder_layer_grads[l]
        );
    }
    
    // 3. 编码器层的反向传播
    float* grad_encoder = (float*)malloc(batch_size * src_len * model->model_dim * sizeof(float));
    for (int l = model->encoder->num_layers - 1; l >= 0; l--) {
        encoder_layer_backward(
            model->encoder->layers[l],
            encoder_output,
            grad_encoder,
            batch_size,
            src_len,
            grad->encoder_layer_grads[l]
        );
    }
    
    // 4. 嵌入层的反向传播
    token_embedding_backward(
        model->src_embed,
        grad_encoder,
        NULL,  // src_tokens
        batch_size,
        src_len,
        grad->grad_src_embed
    );
    
    // 释放临时内存
    free(grad_decoder);
    free(grad_encoder);
} 