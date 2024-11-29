#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void save_checkpoint(TransformerModel* model, int epoch, float loss) {
    if (!model || !model->config || !model->config->checkpoint_dir) return;

    char filename[256];
    snprintf(filename, sizeof(filename), "%s/checkpoint_epoch_%d.bin",
             model->config->checkpoint_dir, epoch);

    FILE* fp = fopen(filename, "wb");
    if (!fp) return;

    // 保存基本信息
    fwrite(&model->model_dim, sizeof(int), 1, fp);
    fwrite(&model->vocab_size, sizeof(int), 1, fp);
    fwrite(&model->max_seq_length, sizeof(int), 1, fp);
    fwrite(&model->num_encoder_layers, sizeof(int), 1, fp);
    fwrite(&model->num_decoder_layers, sizeof(int), 1, fp);
    fwrite(&epoch, sizeof(int), 1, fp);
    fwrite(&loss, sizeof(float), 1, fp);

    // 保存编码器层
    for (int i = 0; i < model->num_encoder_layers; i++) {
        EncoderLayer* layer = model->encoder_layers[i];
        if (!layer) continue;

        // 保存自注意力权重
        MultiHeadAttention* mha = layer->self_attention;
        if (mha) {
            for (int h = 0; h < mha->num_heads; h++) {
                SelfAttention* head = mha->attention_heads[h];
                if (!head) continue;

                const int head_dim = mha->head_dim;
                const int weights_size = head_dim * head_dim;

                // 保存注意力权重
                fwrite(head->query_weights, sizeof(float), weights_size, fp);
                fwrite(head->key_weights, sizeof(float), weights_size, fp);
                fwrite(head->value_weights, sizeof(float), weights_size, fp);
                fwrite(head->output_weights, sizeof(float), weights_size, fp);

                // 保存偏置
                fwrite(head->query_bias, sizeof(float), head_dim, fp);
                fwrite(head->key_bias, sizeof(float), head_dim, fp);
                fwrite(head->value_bias, sizeof(float), head_dim, fp);
                fwrite(head->output_bias, sizeof(float), head_dim, fp);
            }
        }

        // 保存前馈网络权重
        if (layer->feed_forward) {
            const int ff_dim = 4 * model->model_dim;
            fwrite(layer->feed_forward->weights1, sizeof(float), 
                   layer->feed_forward->hidden_dim * layer->feed_forward->input_dim, fp);
            fwrite(layer->feed_forward->weights2, sizeof(float), 
                   layer->feed_forward->input_dim * layer->feed_forward->hidden_dim, fp);
            fwrite(layer->feed_forward->bias1, sizeof(float), 
                   layer->feed_forward->hidden_dim, fp);
            fwrite(layer->feed_forward->bias2, sizeof(float), 
                   layer->feed_forward->input_dim, fp);
        }

        // 保存层归一化参数
        fwrite(layer->layer_norm1, sizeof(float), model->model_dim, fp);
        fwrite(layer->layer_norm2, sizeof(float), model->model_dim, fp);
    }

    fclose(fp);
    printf("Checkpoint saved: %s\n", filename);
} 