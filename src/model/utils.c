#include "model.h"

void free_model(TransformerModel* model) {
    if (!model) return;
    
    // 释放编码器层
    if (model->encoder_layers) {
        for (int i = 0; i < model->num_encoder_layers; i++) {
            if (model->encoder_layers[i]) {
                encoder_layer_free(model->encoder_layers[i]);
            }
        }
        free(model->encoder_layers);
    }

    // 释放解码器层
    if (model->decoder_layers) {
        for (int i = 0; i < model->num_decoder_layers; i++) {
            if (model->decoder_layers[i]) {
                decoder_layer_free(model->decoder_layers[i]);
            }
        }
        free(model->decoder_layers);
    }

    // 释放配置
    if (model->config) {
        free(model->config);
    }

    // 最后释放模型结构体本身
    free(model);
}