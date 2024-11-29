#include "layers.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// 辅助函数：创建因果注意力掩码
static float* create_causal_mask(int seq_len) {
    float* mask = (float*)malloc(seq_len * seq_len * sizeof(float));
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < seq_len; j++) {
            mask[i * seq_len + j] = j <= i ? 0.0f : -INFINITY;
        }
    }
    return mask;
}

Transformer* transformer_create(
    int vocab_size,
    int model_dim,
    int num_heads,
    int num_layers,
    int ff_hidden_dim,
    bool requires_grad
) {
    Transformer* transformer = (Transformer*)malloc(sizeof(Transformer));
    if (!transformer) return NULL;

    transformer->vocab_size = vocab_size;
    transformer->model_dim = model_dim;

    // 创建词嵌入
    transformer->src_embed = token_embedding_create(vocab_size, model_dim, requires_grad);
    transformer->tgt_embed = token_embedding_create(vocab_size, model_dim, requires_grad);
    
    // 创建位置编码
    transformer->pos_enc = positional_encoding_create(model_dim, 1024); // 最大位置1024
    
    // 创建编码器和解码器堆栈
    transformer->encoder = encoder_create(num_layers, model_dim, num_heads, ff_hidden_dim, requires_grad);
    transformer->decoder = decoder_create(num_layers, model_dim, num_heads, ff_hidden_dim, requires_grad);

    // 创建输出线性层
    transformer->linear_weight = (float*)malloc(vocab_size * model_dim * sizeof(float));
    transformer->linear_bias = (float*)malloc(vocab_size * sizeof(float));

    // 初始化输出层参数
    float scale = sqrt(2.0f / model_dim);
    for (int i = 0; i < vocab_size * model_dim; i++) {
        transformer->linear_weight[i] = ((float)rand() / RAND_MAX * 2 - 1) * scale;
    }
    memset(transformer->linear_bias, 0, vocab_size * sizeof(float));

    // 检查所有组件是否创建成功
    if (!transformer->src_embed || !transformer->tgt_embed || 
        !transformer->pos_enc || !transformer->encoder || 
        !transformer->decoder || !transformer->linear_weight || 
        !transformer->linear_bias) {
        transformer_free(transformer);
        return NULL;
    }

    return transformer;
}

void transformer_free(Transformer* transformer) {
    if (transformer) {
        if (transformer->src_embed) token_embedding_free(transformer->src_embed);
        if (transformer->tgt_embed) token_embedding_free(transformer->tgt_embed);
        if (transformer->pos_enc) positional_encoding_free(transformer->pos_enc);
        if (transformer->encoder) encoder_free(transformer->encoder);
        if (transformer->decoder) decoder_free(transformer->decoder);
        free(transformer->linear_weight);
        free(transformer->linear_bias);
        free(transformer);
    }
}

void transformer_forward(
    Transformer* transformer,
    int* src_tokens,      // [batch_size, src_len]
    int* tgt_tokens,      // [batch_size, tgt_len]
    int batch_size,
    int src_len,
    int tgt_len,
    float* output        // [batch_size, tgt_len, vocab_size]
) {
    // 分配临时缓冲区
    int src_embed_size = batch_size * src_len * transformer->model_dim;
    int tgt_embed_size = batch_size * tgt_len * transformer->model_dim;
    
    float* src_embedded = (float*)malloc(src_embed_size * sizeof(float));
    float* tgt_embedded = (float*)malloc(tgt_embed_size * sizeof(float));
    float* encoder_output = (float*)malloc(src_embed_size * sizeof(float));
    float* decoder_output = (float*)malloc(tgt_embed_size * sizeof(float));
    float* causal_mask = create_causal_mask(tgt_len);

    if (!src_embedded || !tgt_embedded || !encoder_output || 
        !decoder_output || !causal_mask) {
        free(src_embedded);
        free(tgt_embedded);
        free(encoder_output);
        free(decoder_output);
        free(causal_mask);
        return;
    }

    // 1. 源语言词嵌入和位置编码
    token_embedding_forward(transformer->src_embed, src_tokens, batch_size, src_len, src_embedded);
    positional_encoding_add(transformer->pos_enc, src_embedded, batch_size, src_len);

    // 2. 目标语言词嵌入和位置编码
    token_embedding_forward(transformer->tgt_embed, tgt_tokens, batch_size, tgt_len, tgt_embedded);
    positional_encoding_add(transformer->pos_enc, tgt_embedded, batch_size, tgt_len);

    // 3. 编码器前向传播
    encoder_forward(transformer->encoder, src_embedded, batch_size, src_len, encoder_output);

    // 4. 解码器前向传播
    decoder_forward(
        transformer->decoder,
        tgt_embedded,
        encoder_output,
        causal_mask,
        batch_size,
        tgt_len,
        src_len,
        decoder_output
    );

    // 5. 输出线性层
    for (int b = 0; b < batch_size; b++) {
        for (int t = 0; t < tgt_len; t++) {
            for (int v = 0; v < transformer->vocab_size; v++) {
                float sum = transformer->linear_bias[v];
                for (int d = 0; d < transformer->model_dim; d++) {
                    sum += decoder_output[(b * tgt_len + t) * transformer->model_dim + d] *
                           transformer->linear_weight[v * transformer->model_dim + d];
                }
                output[(b * tgt_len + t) * transformer->vocab_size + v] = sum;
            }
        }
    }

    // 释放临时缓冲区
    free(src_embedded);
    free(tgt_embedded);
    free(encoder_output);
    free(decoder_output);
    free(causal_mask);
}

// 辅助函数：采样下一个token
static int sample_next_token(float* logits, int vocab_size, float temperature) {
    // 应用temperature
    float max_logit = -INFINITY;
    for (int i = 0; i < vocab_size; i++) {
        logits[i] /= temperature;
        if (logits[i] > max_logit) max_logit = logits[i];
    }

    // 计算softmax
    float sum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        logits[i] = exp(logits[i] - max_logit);
        sum += logits[i];
    }
    for (int i = 0; i < vocab_size; i++) {
        logits[i] /= sum;
    }

    // 采样
    float r = (float)rand() / RAND_MAX;
    float cumsum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        cumsum += logits[i];
        if (r <= cumsum) return i;
    }
    return vocab_size - 1;
}

void transformer_generate(
    Transformer* transformer,
    int* src_tokens,      // [batch_size, src_len]
    int batch_size,
    int src_len,
    int max_len,
    float temperature,
    int* output_tokens    // [batch_size, max_len]
) {
    // 编码源序列
    float* src_embedded = (float*)malloc(batch_size * src_len * transformer->model_dim * sizeof(float));
    float* encoder_output = (float*)malloc(batch_size * src_len * transformer->model_dim * sizeof(float));
    
    // 编码器前向传播
    token_embedding_forward(transformer->src_embed, src_tokens, batch_size, src_len, src_embedded);
    positional_encoding_add(transformer->pos_enc, src_embedded, batch_size, src_len);
    encoder_forward(transformer->encoder, src_embedded, batch_size, src_len, encoder_output);

    // 逐token生成
    for (int t = 0; t < max_len; t++) {
        float* tgt_embedded = (float*)malloc(batch_size * (t + 1) * transformer->model_dim * sizeof(float));
        float* decoder_output = (float*)malloc(batch_size * (t + 1) * transformer->model_dim * sizeof(float));
        float* causal_mask = create_causal_mask(t + 1);
        float* logits = (float*)malloc(transformer->vocab_size * sizeof(float));

        // 解码器前向传播
        token_embedding_forward(transformer->tgt_embed, output_tokens, batch_size, t + 1, tgt_embedded);
        positional_encoding_add(transformer->pos_enc, tgt_embedded, batch_size, t + 1);
        
        decoder_forward(
            transformer->decoder,
            tgt_embedded,
            encoder_output,
            causal_mask,
            batch_size,
            t + 1,
            src_len,
            decoder_output
        );

        // 为每个批次生成下一个token
        for (int b = 0; b < batch_size; b++) {
            // 计算最后一个位置的logits
            for (int v = 0; v < transformer->vocab_size; v++) {
                float sum = transformer->linear_bias[v];
                for (int d = 0; d < transformer->model_dim; d++) {
                    sum += decoder_output[(b * (t + 1) + t) * transformer->model_dim + d] *
                           transformer->linear_weight[v * transformer->model_dim + d];
                }
                logits[v] = sum;
            }
            
            // 采样下一个token
            output_tokens[b * max_len + t] = sample_next_token(logits, transformer->vocab_size, temperature);
        }

        free(tgt_embedded);
        free(decoder_output);
        free(causal_mask);
        free(logits);
    }

    free(src_embedded);
    free(encoder_output);
}