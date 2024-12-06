#include "grad.h"
#include <stdlib.h>
#include <string.h>

TransformerGrad* create_transformer_grad(Transformer* model) {
    TransformerGrad* grad = (TransformerGrad*)malloc(sizeof(TransformerGrad));
    if (!grad) return NULL;
    // 分配词嵌入梯度内存
    grad->grad_src_embed = (float*)calloc(model->vocab_size * model->model_dim, sizeof(float));
    grad->grad_tgt_embed = (float*)calloc(model->vocab_size * model->model_dim, sizeof(float));

    // 分配编码器层梯度内存
    grad->encoder_layer_grads = (EncoderLayerGrad**)malloc(model->encoder->num_layers * sizeof(EncoderLayerGrad*));
    for (int i = 0; i < model->encoder->num_layers; i++) {
        grad->encoder_layer_grads[i] = (EncoderLayerGrad*)malloc(sizeof(EncoderLayerGrad));
        EncoderLayer* layer = model->encoder->layers[i];
        
        // 创建多头注意力梯度
        grad->encoder_layer_grads[i]->self_attention_grad = (MultiHeadAttentionGrad*)malloc(sizeof(MultiHeadAttentionGrad));
        grad->encoder_layer_grads[i]->self_attention_grad->grad_W_q = (float*)calloc(model->model_dim * model->model_dim, sizeof(float));
        grad->encoder_layer_grads[i]->self_attention_grad->grad_W_k = (float*)calloc(model->model_dim * model->model_dim, sizeof(float));
        grad->encoder_layer_grads[i]->self_attention_grad->grad_W_v = (float*)calloc(model->model_dim * model->model_dim, sizeof(float));
        grad->encoder_layer_grads[i]->self_attention_grad->grad_W_o = (float*)calloc(model->model_dim * model->model_dim, sizeof(float));

        // 创建前馈网络梯度
        grad->encoder_layer_grads[i]->feed_forward_grad = (FeedForwardGrad*)malloc(sizeof(FeedForwardGrad));
        grad->encoder_layer_grads[i]->feed_forward_grad->grad_w1 = (float*)calloc(model->model_dim * layer->feed_forward->hidden_dim, sizeof(float));
        grad->encoder_layer_grads[i]->feed_forward_grad->grad_w2 = (float*)calloc(layer->feed_forward->hidden_dim * model->model_dim, sizeof(float));
        grad->encoder_layer_grads[i]->feed_forward_grad->grad_b1 = (float*)calloc(layer->feed_forward->hidden_dim, sizeof(float));
        grad->encoder_layer_grads[i]->feed_forward_grad->grad_b2 = (float*)calloc(model->model_dim, sizeof(float));
    }

    // 分配解码器层梯度内存
    grad->decoder_layer_grads = (DecoderLayerGrad**)malloc(model->decoder->num_layers * sizeof(DecoderLayerGrad*));
    for (int i = 0; i < model->decoder->num_layers; i++) {
        grad->decoder_layer_grads[i] = (DecoderLayerGrad*)malloc(sizeof(DecoderLayerGrad));
        DecoderLayer* layer = model->decoder->layers[i];
        
        // 创建自注意力梯度
        grad->decoder_layer_grads[i]->self_attention_grad = (MultiHeadAttentionGrad*)malloc(sizeof(MultiHeadAttentionGrad));
        grad->decoder_layer_grads[i]->self_attention_grad->grad_W_q = (float*)calloc(model->model_dim * model->model_dim, sizeof(float));
        grad->decoder_layer_grads[i]->self_attention_grad->grad_W_k = (float*)calloc(model->model_dim * model->model_dim, sizeof(float));
        grad->decoder_layer_grads[i]->self_attention_grad->grad_W_v = (float*)calloc(model->model_dim * model->model_dim, sizeof(float));
        grad->decoder_layer_grads[i]->self_attention_grad->grad_W_o = (float*)calloc(model->model_dim * model->model_dim, sizeof(float));
        
        // 创建交叉注意力梯度
        grad->decoder_layer_grads[i]->cross_attention_grad = (MultiHeadAttentionGrad*)malloc(sizeof(MultiHeadAttentionGrad));
        grad->decoder_layer_grads[i]->cross_attention_grad->grad_W_q = (float*)calloc(model->model_dim * model->model_dim, sizeof(float));
        grad->decoder_layer_grads[i]->cross_attention_grad->grad_W_k = (float*)calloc(model->model_dim * model->model_dim, sizeof(float));
        grad->decoder_layer_grads[i]->cross_attention_grad->grad_W_v = (float*)calloc(model->model_dim * model->model_dim, sizeof(float));
        grad->decoder_layer_grads[i]->cross_attention_grad->grad_W_o = (float*)calloc(model->model_dim * model->model_dim, sizeof(float));

        // 创建前馈网络梯度
        grad->decoder_layer_grads[i]->feed_forward_grad = (FeedForwardGrad*)malloc(sizeof(FeedForwardGrad));
        grad->decoder_layer_grads[i]->feed_forward_grad->grad_w1 = (float*)calloc(model->model_dim * layer->feed_forward->hidden_dim, sizeof(float));
        grad->decoder_layer_grads[i]->feed_forward_grad->grad_w2 = (float*)calloc(layer->feed_forward->hidden_dim * model->model_dim, sizeof(float));
        grad->decoder_layer_grads[i]->feed_forward_grad->grad_b1 = (float*)calloc(layer->feed_forward->hidden_dim, sizeof(float));
        grad->decoder_layer_grads[i]->feed_forward_grad->grad_b2 = (float*)calloc(model->model_dim, sizeof(float));
    }

    // 分配输出层梯度内存
    grad->grad_linear_weight = (float*)calloc(model->vocab_size * model->model_dim, sizeof(float));
    grad->grad_linear_bias = (float*)calloc(model->vocab_size, sizeof(float));

    return grad;
}

void zero_transformer_grad(TransformerGrad* grad, Transformer* model) {
    // 清零词嵌入梯度
    memset(grad->grad_src_embed, 0, model->vocab_size * model->model_dim * sizeof(float));
    memset(grad->grad_tgt_embed, 0, model->vocab_size * model->model_dim * sizeof(float));

    // 清零编码器层梯度
    for (int i = 0; i < model->encoder->num_layers; i++) {
        EncoderLayer* layer = model->encoder->layers[i];
        memset(grad->encoder_layer_grads[i]->self_attention_grad->grad_W_q, 0, model->model_dim * model->model_dim * sizeof(float));
        memset(grad->encoder_layer_grads[i]->self_attention_grad->grad_W_k, 0, model->model_dim * model->model_dim * sizeof(float));
        memset(grad->encoder_layer_grads[i]->self_attention_grad->grad_W_v, 0, model->model_dim * model->model_dim * sizeof(float));
        memset(grad->encoder_layer_grads[i]->self_attention_grad->grad_W_o, 0, model->model_dim * model->model_dim * sizeof(float));
        memset(grad->encoder_layer_grads[i]->feed_forward_grad->grad_w1, 0, model->model_dim * layer->feed_forward->hidden_dim * sizeof(float));
        memset(grad->encoder_layer_grads[i]->feed_forward_grad->grad_w2, 0, layer->feed_forward->hidden_dim * model->model_dim * sizeof(float));
        memset(grad->encoder_layer_grads[i]->feed_forward_grad->grad_b1, 0, layer->feed_forward->hidden_dim * sizeof(float));
        memset(grad->encoder_layer_grads[i]->feed_forward_grad->grad_b2, 0, model->model_dim * sizeof(float));
    }

    // 清零解码器层梯度
    for (int i = 0; i < model->decoder->num_layers; i++) {
        DecoderLayer* layer = model->decoder->layers[i];
        memset(grad->decoder_layer_grads[i]->self_attention_grad->grad_W_q, 0, model->model_dim * model->model_dim * sizeof(float));
        memset(grad->decoder_layer_grads[i]->self_attention_grad->grad_W_k, 0, model->model_dim * model->model_dim * sizeof(float));
        memset(grad->decoder_layer_grads[i]->self_attention_grad->grad_W_v, 0, model->model_dim * model->model_dim * sizeof(float));
        memset(grad->decoder_layer_grads[i]->self_attention_grad->grad_W_o, 0, model->model_dim * model->model_dim * sizeof(float));
        memset(grad->decoder_layer_grads[i]->cross_attention_grad->grad_W_q, 0, model->model_dim * model->model_dim * sizeof(float));
        memset(grad->decoder_layer_grads[i]->cross_attention_grad->grad_W_k, 0, model->model_dim * model->model_dim * sizeof(float));
        memset(grad->decoder_layer_grads[i]->cross_attention_grad->grad_W_v, 0, model->model_dim * model->model_dim * sizeof(float));
        memset(grad->decoder_layer_grads[i]->cross_attention_grad->grad_W_o, 0, model->model_dim * model->model_dim * sizeof(float));
        memset(grad->decoder_layer_grads[i]->feed_forward_grad->grad_w1, 0, model->model_dim * layer->feed_forward->hidden_dim * sizeof(float));
        memset(grad->decoder_layer_grads[i]->feed_forward_grad->grad_w2, 0, layer->feed_forward->hidden_dim * model->model_dim * sizeof(float));
        memset(grad->decoder_layer_grads[i]->feed_forward_grad->grad_b1, 0, layer->feed_forward->hidden_dim * sizeof(float));
        memset(grad->decoder_layer_grads[i]->feed_forward_grad->grad_b2, 0, model->model_dim * sizeof(float));
    }

    // 清零输出层梯度
    memset(grad->grad_linear_weight, 0, model->vocab_size * model->model_dim * sizeof(float));
    memset(grad->grad_linear_bias, 0, model->vocab_size * sizeof(float));
}

void free_transformer_grad(TransformerGrad* grad, Transformer* model) {
    if (!grad) return;

    // 释放词嵌入梯度内存
    free(grad->grad_src_embed);
    free(grad->grad_tgt_embed);

    // 释放编码器层梯度内存
    for (int i = 0; i < model->encoder->num_layers; i++) {
        // 释放自注意力梯度
        free(grad->encoder_layer_grads[i]->self_attention_grad->grad_W_q);
        free(grad->encoder_layer_grads[i]->self_attention_grad->grad_W_k);
        free(grad->encoder_layer_grads[i]->self_attention_grad->grad_W_v);
        free(grad->encoder_layer_grads[i]->self_attention_grad->grad_W_o);
        free(grad->encoder_layer_grads[i]->self_attention_grad);  // 释放注意力梯度结构本身

        // 释放前馈网络梯度
        free(grad->encoder_layer_grads[i]->feed_forward_grad->grad_w1);
        free(grad->encoder_layer_grads[i]->feed_forward_grad->grad_w2);
        free(grad->encoder_layer_grads[i]->feed_forward_grad->grad_b1);
        free(grad->encoder_layer_grads[i]->feed_forward_grad->grad_b2);
        free(grad->encoder_layer_grads[i]->feed_forward_grad);
        free(grad->encoder_layer_grads[i]);
    }
    free(grad->encoder_layer_grads);

    // 释放解码器层梯度内存
    for (int i = 0; i < model->decoder->num_layers; i++) {
        // 释放自注意力梯度
        free(grad->decoder_layer_grads[i]->self_attention_grad->grad_W_q);
        free(grad->decoder_layer_grads[i]->self_attention_grad->grad_W_k);
        free(grad->decoder_layer_grads[i]->self_attention_grad->grad_W_v);
        free(grad->decoder_layer_grads[i]->self_attention_grad->grad_W_o);
        free(grad->decoder_layer_grads[i]->self_attention_grad);  // 释放自注意力梯度结构

        // 释放交叉注意力梯度
        free(grad->decoder_layer_grads[i]->cross_attention_grad->grad_W_q);
        free(grad->decoder_layer_grads[i]->cross_attention_grad->grad_W_k);
        free(grad->decoder_layer_grads[i]->cross_attention_grad->grad_W_v);
        free(grad->decoder_layer_grads[i]->cross_attention_grad->grad_W_o);
        free(grad->decoder_layer_grads[i]->cross_attention_grad);  // 释放交叉注意力梯度结构

        // 释放前馈网络梯度
        free(grad->decoder_layer_grads[i]->feed_forward_grad->grad_w1);
        free(grad->decoder_layer_grads[i]->feed_forward_grad->grad_w2);
        free(grad->decoder_layer_grads[i]->feed_forward_grad->grad_b1);
        free(grad->decoder_layer_grads[i]->feed_forward_grad->grad_b2);
        free(grad->decoder_layer_grads[i]->feed_forward_grad);
        free(grad->decoder_layer_grads[i]);
    }
    free(grad->decoder_layer_grads);

    // 释放输出层梯度内存
    free(grad->grad_linear_weight);
    free(grad->grad_linear_bias);

    free(grad);
}