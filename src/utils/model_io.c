#include "model.h"
#include <stdio.h>
#include <stdlib.h>

void save_model(TransformerModel* model, const char* filename) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) return;
    
    // 保存模型配置
    fwrite(&model->num_encoder_layers, sizeof(int), 1, fp);
    fwrite(&model->num_decoder_layers, sizeof(int), 1, fp);
    fwrite(&model->model_dim, sizeof(int), 1, fp);
    
    // 保存词嵌入
    fwrite(model->token_embedding->embedding_matrix, 
           sizeof(float), 
           model->token_embedding->vocab_size * model->token_embedding->embedding_dim, 
           fp);
    
    // 保存位置编码
    fwrite(model->positional_encoding->positional_encoding,
           sizeof(float),
           model->positional_encoding->max_seq_length * model->positional_encoding->embedding_dim,
           fp);
    
    // 保存编码器层参数
    for (int i = 0; i < model->num_encoder_layers; i++) {
        EncoderLayer* layer = model->encoder_layers[i];
        // 保存注意力层参数
        // 保存前馈网络参数
        // ... 其他参数
    }
    
    fclose(fp);
}

TransformerModel* load_model(const char* filename) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) return NULL;
    
    TransformerModel* model = malloc(sizeof(TransformerModel));
    
    // 读取模型配置
    fread(&model->num_encoder_layers, sizeof(int), 1, fp);
    fread(&model->num_decoder_layers, sizeof(int), 1, fp);
    fread(&model->model_dim, sizeof(int), 1, fp);
    
    // 初始化和加载其他组件
    // ...
    
    fclose(fp);
    return model;
} 