#include "training.h"
#include "utils.h"
#include "optimizer.h"
#include <stdlib.h>
#include <math.h>

// 前向声明
float* decoder_backward_pass(TransformerModel* model, float* loss_grad);
float* encoder_backward_pass(TransformerModel* model, float* decoder_grad);
void update_embedding_gradients(TokenEmbedding* embedding, float* gradients);

void backward_pass(TransformerModel* model, float* loss_grad) {
    if (!model || !loss_grad) return;
    
    // 解码器反向传播
    float* decoder_grad = decoder_backward_pass(model, loss_grad);
    if (!decoder_grad) return;
    
    // 编码器反向传播
    float* encoder_grad = encoder_backward_pass(model, decoder_grad);
    if (!encoder_grad) {
        free(decoder_grad);
        return;
    }
    
    // 更新嵌入层梯度
    update_embedding_gradients(model->token_embedding, encoder_grad);
    
    // 清理
    free(decoder_grad);
    free(encoder_grad);
}

void clip_gradients(float* gradients, int size, float max_norm) {
    if (!gradients || size <= 0) return;
    
    // 计算梯度范数
    float total_norm = 0.0f;
    for (int i = 0; i < size; i++) {
        total_norm += gradients[i] * gradients[i];
    }
    total_norm = sqrt(total_norm);
    
    // 如果范数超过阈值，进行缩放
    if (total_norm > max_norm) {
        float scale = max_norm / total_norm;
        for (int i = 0; i < size; i++) {
            gradients[i] *= scale;
        }
    }
}

void apply_gradients(TransformerModel* model, AdamOptimizer* optimizer) {
    if (!model || !optimizer) return;
    
    // 应用嵌入层梯度
    if (model->token_embedding && model->token_embedding->requires_grad) {
        int embedding_size = model->token_embedding->vocab_size * 
                           model->token_embedding->embedding_dim;
        
        adam_step(optimizer,
                 model->token_embedding->embedding_matrix,
                 model->token_embedding->embedding_gradients,
                 embedding_size);
    }
    
    // 应用编码器梯度
    for (int i = 0; i < model->num_encoder_layers; i++) {
        // TODO: 实现编码器梯度更新
        // EncoderLayer* layer = model->encoder_layers[i];
    }
    
    // 应用解码器梯度
    for (int i = 0; i < model->num_decoder_layers; i++) {
        // TODO: 实现解码器梯度更新
        // DecoderLayer* layer = model->decoder_layers[i];
    }
} 