#include "training.h"
#include "layers.h"
#include "grad.h"
#include "embeddings.h"
#include <math.h>
#include <stdlib.h>

OptimizerState* create_optimizer(Transformer* model) {
    OptimizerState* opt = (OptimizerState*)malloc(sizeof(OptimizerState));
    
    // 计算需要优化的参数总数
    size_t total_params = 0;
    // TODO: 计算模型中所有参数的数量
    
    // 初始化优化器状态
    opt->momentum = (float*)calloc(total_params, sizeof(float));
    opt->velocity = (float*)calloc(total_params, sizeof(float));
    opt->beta1 = 0.9f;
    opt->beta2 = 0.999f;
    opt->epsilon = 1e-8f;
    opt->step = 0;
    
    return opt;
}

void optimizer_step(OptimizerState* opt, TrainingConfig* config) {
    opt->step++;
    float lr = config->learning_rate;
    
    // 如果使用warmup，调整学习率
    if (opt->step < config->warmup_steps) {
        lr *= (float)opt->step / config->warmup_steps;
    }
    
    // TODO: 实现AdaM优化器的参数更新逻辑
}

void free_optimizer(OptimizerState* opt) {
    free(opt->momentum);
    free(opt->velocity);
    free(opt);
}

void apply_gradients(
    Transformer* model,
    TransformerGrad* grad,
    OptimizerState* opt,
    float learning_rate,
    float weight_decay
) {
    // 用于Adam优化器的修正系数
    float correction1 = 1.0f / (1.0f - powf(opt->beta1, opt->step));
    float correction2 = 1.0f / (1.0f - powf(opt->beta2, opt->step));
    float lr_t = learning_rate * sqrtf(correction2) / correction1;
    
    // 1. 更新词嵌入参数
    update_embedding_params(
        model->src_embed,
        grad->grad_src_embed,
        opt,
        lr_t,
        weight_decay
    );
    
    // 2. 更新编码器层参数
    for (int i = 0; i < model->encoder->num_layers; i++) {
        update_encoder_layer_params(
            model->encoder->layers[i],
            grad->encoder_layer_grads[i],
            opt,
            lr_t,
            weight_decay
        );
    }
    
    // 3. 更新解码器层参数
    for (int i = 0; i < model->decoder->num_layers; i++) {
        update_decoder_layer_params(
            model->decoder->layers[i],
            grad->decoder_layer_grads[i],
            opt,
            lr_t,
            weight_decay
        );
    }
}

// 辅助函数：更新单个参数
static void update_parameter(
    float* param,
    float* grad,
    float* m,
    float* v,
    int size,
    float lr,
    float beta1,
    float beta2,
    float epsilon,
    float weight_decay
) {
    for (int i = 0; i < size; i++) {
        // 应用权重衰减
        grad[i] += weight_decay * param[i];
        
        // 更新动量
        m[i] = beta1 * m[i] + (1 - beta1) * grad[i];
        // 更新速度
        v[i] = beta2 * v[i] + (1 - beta2) * grad[i] * grad[i];
        
        // 更新参数
        param[i] -= lr * m[i] / (sqrtf(v[i]) + epsilon);
    }
}

void update_embedding_params(
    TokenEmbedding* emb,
    TokenEmbeddingGrad* grad,
    OptimizerState* opt,
    float lr,
    float weight_decay
) {
    int param_size = emb->vocab_size * emb->embedding_dim;
    update_parameter(
        emb->embedding_matrix,  // 修正字段名为embedding
        grad->grad_embedding,
        opt->momentum,
        opt->velocity,
        param_size,
        lr,
        opt->beta1,
        opt->beta2,
        opt->epsilon,
        weight_decay
    );
}

// void update_encoder_layer_params(
//     EncoderLayer* layer,
//     EncoderLayerGrad* grad,
//     OptimizerState* opt,
//     float lr,
//     float weight_decay
// ) {
//     // 更新自注意力层参数
//     update_attention_params(
//         layer->self_attention,
//         grad->self_attention_grad,
//         opt,
//         lr,
//         weight_decay
//     );
    
//     // 更新前馈网络参数
//     update_feedforward_params(
//         layer->feed_forward,
//         grad->feed_forward_grad,
//         opt,
//         lr,
//         weight_decay
//     );
    
//     // 更新层归一化参数
//     update_layernorm_params(
//         layer->norm1,
//         grad->norm1_grad,
//         opt,
//         lr,
//         weight_decay
//     );
//     update_layernorm_params(
//         layer->norm2,
//         grad->norm2_grad,
//         opt,
//         lr,
//         weight_decay
//     );
// }

// // 类似地实现update_decoder_layer_params... 