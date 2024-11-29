#include "training.h"
#include "utils.h"
#include <stdlib.h>
#include <stddef.h>
#include <string.h>

float* decoder_backward_pass(TransformerModel* model, float* output_grad) {
    if (!model || !output_grad) return NULL;
    
    float* current_grad = malloc(model->model_dim * model->max_seq_length * sizeof(float));
    if (!current_grad) return NULL;
    
    memcpy(current_grad, output_grad, model->model_dim * model->max_seq_length * sizeof(float));
    
    float* layer_input = NULL;  // TODO: 保存前向传播的中间结果
    
    // 从最后一层向前传播
    for (int i = model->num_decoder_layers - 1; i >= 0; i--) {
        DecoderLayer* layer = model->decoder_layers[i];
        
        // 前馈网络反向传播
        float* ff_grad = NULL;
        feed_forward_backward(layer->feed_forward, current_grad, layer_input);
        
        // 交叉注意力反向传播
        float* cross_attn_grad = attention_backward(layer->cross_attention, ff_grad);
        if (!cross_attn_grad) {
            free(ff_grad);
            return NULL;
        }
        
        // 自注意力反向传播
        float* self_attn_grad = masked_attention_backward(layer->self_attention,
                                                        cross_attn_grad);
        if (!self_attn_grad) {
            free(ff_grad);
            free(cross_attn_grad);
            return NULL;
        }
        
        // 更新该层的梯度
        update_layer_gradients(layer, self_attn_grad, cross_attn_grad, ff_grad);
        
        // 准备下一层的梯度
        if (i > 0) {
            current_grad = self_attn_grad;
        }
        
        // 清理本层的中间结果
        free(ff_grad);
        free(cross_attn_grad);
        if (i == 0) {
            free(self_attn_grad);
        }
    }
    
    return current_grad;
}

void update_layer_gradients(DecoderLayer* layer,
                          float* self_attn_grad,
                          float* cross_attn_grad,
                          float* ff_grad) {
    if (!layer) return;
    
    // 累积自注意力梯度
    if (layer->self_attention && self_attn_grad) {
        accumulate_attention_gradients(layer->self_attention, self_attn_grad);
    }
    
    // 累积交叉注意力梯度
    if (layer->cross_attention && cross_attn_grad) {
        accumulate_attention_gradients(layer->cross_attention, cross_attn_grad);
    }
    
    // 累积前馈网络梯度
    if (layer->feed_forward && ff_grad) {
        accumulate_feed_forward_gradients(layer->feed_forward, ff_grad);
    }
}

// 添加新函数用于应用累积的梯度
void apply_layer_gradients(DecoderLayer* layer, float learning_rate) {
    if (!layer) return;
    
    // 更新自注意力权重
    if (layer->self_attention) {
        update_attention_gradients(layer->self_attention, learning_rate);
    }
    
    // 更新交叉注意力权重
    if (layer->cross_attention) {
        update_attention_gradients(layer->cross_attention, learning_rate);
    }
    
    // 更新前馈网络权重
    if (layer->feed_forward) {
        update_feed_forward_gradients(layer->feed_forward, learning_rate);
    }
} 