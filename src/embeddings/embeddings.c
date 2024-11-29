#include "embeddings.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>

void initialize_positional_encoding(PositionalEncoding* pos_enc) {
    if (!pos_enc || !pos_enc->encodings) return;
    
    for (int pos = 0; pos < pos_enc->max_seq_length; pos++) {
        for (int i = 0; i < pos_enc->encoding_dim; i += 2) {
            float angle = pos / powf(10000.0f, (float)i / pos_enc->encoding_dim);
            pos_enc->encodings[pos * pos_enc->encoding_dim + i] = sinf(angle);
            
            if (i + 1 < pos_enc->encoding_dim) {
                pos_enc->encodings[pos * pos_enc->encoding_dim + i + 1] = cosf(angle);
            }
        }
    }
}

float* add_positional_encoding(PositionalEncoding* pos_enc, float* embeddings,
                             int seq_length) {
    if (!pos_enc || !embeddings || seq_length <= 0) return NULL;
    
    int encoding_dim = pos_enc->encoding_dim;
    float* output = malloc(seq_length * encoding_dim * sizeof(float));
    if (!output) return NULL;
    
    for (int pos = 0; pos < seq_length; pos++) {
        for (int i = 0; i < encoding_dim; i++) {
            output[pos * encoding_dim + i] = 
                embeddings[pos * encoding_dim + i] +
                pos_enc->encodings[pos * encoding_dim + i];
        }
    }
    
    return output;
}

PositionalEncoding* create_positional_encoding(int max_seq_length, int encoding_dim) {
    if (max_seq_length <= 0 || encoding_dim <= 0) return NULL;
    
    PositionalEncoding* pos_enc = malloc(sizeof(PositionalEncoding));
    if (!pos_enc) return NULL;
    
    // 初始化基本参数
    pos_enc->max_seq_length = max_seq_length;
    pos_enc->encoding_dim = encoding_dim;
    
    // 分配编码矩阵内存
    pos_enc->encodings = malloc(max_seq_length * encoding_dim * sizeof(float));
    if (!pos_enc->encodings) {
        free(pos_enc);
        return NULL;
    }
    
    // 初始化位置编码
    initialize_positional_encoding(pos_enc);
    
    return pos_enc;
}

void free_positional_encoding(PositionalEncoding* pos_enc) {
    if (!pos_enc) return;
    
    if (pos_enc->encodings) {
        free(pos_enc->encodings);
    }
    
    free(pos_enc);
} 