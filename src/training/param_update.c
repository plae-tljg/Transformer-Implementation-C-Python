#include "training.h"
#include "layers.h"
#include "grad.h"

// 更新注意力层参数
void update_attention_params(
    MultiHeadAttention* attn, 
    MultiHeadAttentionGrad* grad,
    OptimizerState* opt,
    float lr,
    float weight_decay
) {
    // 更新每个注意力头的参数
    for (int h = 0; h < attn->num_heads; h++) {
        SelfAttention* head = attn->attention_heads[h];
        int head_size = attn->head_dim * attn->model_dim;
        
        // 更新权重
        for (int i = 0; i < head_size; i++) {
            // 应用权重衰减
            float w_decay_q = weight_decay * head->query_weights[i];
            float w_decay_k = weight_decay * head->key_weights[i];
            float w_decay_v = weight_decay * head->value_weights[i];
            
            // 更新权重
            head->query_weights[i] -= lr * (grad->grad_W_q[h * head_size + i] + w_decay_q);
            head->key_weights[i] -= lr * (grad->grad_W_k[h * head_size + i] + w_decay_k);
            head->value_weights[i] -= lr * (grad->grad_W_v[h * head_size + i] + w_decay_v);
        }

        // 更新偏置
        for (int i = 0; i < attn->head_dim; i++) {
            head->query_bias[i] -= lr * grad->grad_W_q[h * attn->head_dim + i];
            head->key_bias[i] -= lr * grad->grad_W_k[h * attn->head_dim + i];
            head->value_bias[i] -= lr * grad->grad_W_v[h * attn->head_dim + i];
        }
    }

    // 更新输出层权重和偏置
    for (int i = 0; i < attn->model_dim * attn->model_dim; i++) {
        float w_decay_o = weight_decay * attn->output_weights[i];
        attn->output_weights[i] -= lr * (grad->grad_W_o[i] + w_decay_o);
    }
    
    for (int i = 0; i < attn->model_dim; i++) {
        attn->output_bias[i] -= lr * grad->grad_W_o[i];
    }
}

// 更新前馈网络参数
void update_feedforward_params(
    FeedForward* ff, 
    FeedForwardGrad* grad,
    OptimizerState* opt,
    float lr,
    float weight_decay
) {
    // 更新第一层权重和偏置
    for (int i = 0; i < ff->input_dim * ff->hidden_dim; i++) {
        float w_decay1 = weight_decay * ff->w1[i];
        ff->w1[i] -= lr * (grad->grad_w1[i] + w_decay1);
    }
    
    for (int i = 0; i < ff->hidden_dim; i++) {
        ff->b1[i] -= lr * grad->grad_b1[i];  // 偏置通常不使用权重衰减
    }

    // 更新第二层权重和偏置
    for (int i = 0; i < ff->hidden_dim * ff->input_dim; i++) {
        float w_decay2 = weight_decay * ff->w2[i];
        ff->w2[i] -= lr * (grad->grad_w2[i] + w_decay2);
    }
    
    for (int i = 0; i < ff->input_dim; i++) {
        ff->b2[i] -= lr * grad->grad_b2[i];
    }
}

// 更新层归一化参数
void update_layernorm_params(
    LayerNorm* ln, 
    LayerNormGrad* grad,
    OptimizerState* opt,
    float lr,
    float weight_decay
) {
    // 更新 gamma 和 beta 参数
    for (int i = 0; i < ln->normalized_shape; i++) {
        float w_decay_gamma = weight_decay * ln->gamma[i];
        ln->gamma[i] -= lr * (grad->grad_gamma[i] + w_decay_gamma);
        ln->beta[i] -= lr * grad->grad_beta[i];  // beta 通常不使用权重衰减
    }
}

// 更新解码器层参数
void update_decoder_layer_params(
    DecoderLayer* layer, 
    DecoderLayerGrad* grad, 
    OptimizerState* opt,
    float lr,
    float weight_decay
) {
    // 更新自注意力层参数
    update_attention_params(layer->self_attention, grad->self_attention_grad, opt, lr, weight_decay);
    
    // 更新交叉注意力层参数
    update_attention_params(layer->cross_attention, grad->cross_attention_grad, opt, lr, weight_decay);
    
    // 更新前馈网络参数
    update_feedforward_params(layer->feed_forward, grad->feed_forward_grad, opt, lr, weight_decay);
    
    // 更新层归一化参数
    update_layernorm_params(layer->norm1, grad->norm1_grad, opt, lr, weight_decay);
    update_layernorm_params(layer->norm2, grad->norm2_grad, opt, lr, weight_decay);
    update_layernorm_params(layer->norm3, grad->norm3_grad, opt, lr, weight_decay);
}

// 更新编码器层参数
void update_encoder_layer_params(
    EncoderLayer* layer, 
    EncoderLayerGrad* grad,
    OptimizerState* opt,
    float lr,
    float weight_decay
) {
    // 更新自注意力层参数
    update_attention_params(layer->self_attention, grad->self_attention_grad, opt, lr, weight_decay);
    
    // 更新前馈网络参数
    update_feedforward_params(layer->feed_forward, grad->feed_forward_grad, opt, lr, weight_decay);
    
    // 更新层归一化参数
    update_layernorm_params(layer->norm1, grad->norm1_grad, opt, lr, weight_decay);
    update_layernorm_params(layer->norm2, grad->norm2_grad, opt, lr, weight_decay);
} 