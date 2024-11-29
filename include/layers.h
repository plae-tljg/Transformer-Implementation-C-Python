#ifndef LAYERS_H
#define LAYERS_H
#include "attention.h"
#include <stdbool.h>

typedef struct EncoderLayer EncoderLayer;
typedef struct DecoderLayer DecoderLayer;
typedef struct PositionalEncoding PositionalEncoding;
typedef struct TokenEmbedding TokenEmbedding;
typedef struct LayerNorm LayerNorm;
typedef struct FeedForward FeedForward;
typedef struct Encoder Encoder;
typedef struct Decoder Decoder;
typedef struct Transformer Transformer;

// 编码器层结构
struct EncoderLayer {
    MultiHeadAttention* self_attention;
    LayerNorm* norm1;
    LayerNorm* norm2;
    FeedForward* feed_forward;
    int model_dim;
};

// 解码器层结构
struct DecoderLayer {
    MultiHeadAttention* self_attention;
    MultiHeadAttention* cross_attention;
    LayerNorm* norm1;
    LayerNorm* norm2;
    LayerNorm* norm3;
    FeedForward* feed_forward;
    int model_dim;
};

struct LayerNorm {
    int normalized_shape;  // 需要归一化的维度大小
    float epsilon;        // 用于数值稳定性的小常数
    
    // 可学习参数
    float* gamma;         // 缩放参数
    float* beta;          // 偏移参数
    
    bool requires_grad;   // 是否需要梯度
};

struct FeedForward {
    int input_dim;          // 输入维度
    int hidden_dim;         // 隐藏层维度
    float* w1;             // 第一个线性变换的权重 [hidden_dim, input_dim]
    float* b1;             // 第一个线性变换的偏置 [hidden_dim]
    float* w2;             // 第二个线性变换的权重 [input_dim, hidden_dim]
    float* b2;             // 第二个线性变换的偏置 [input_dim]
    bool requires_grad;     // 是否需要梯度
};

// 创建层归一化
LayerNorm* layer_norm_create(int normalized_shape, float epsilon, bool requires_grad);

// 释放层归一化
void layer_norm_free(LayerNorm* ln);

// 前向传播
void layer_norm_forward(
    LayerNorm* ln,
    float* input,      // [batch_size, normalized_shape]
    int batch_size,
    float* output     // [batch_size, normalized_shape]
);

// 创建前馈网络
FeedForward* feed_forward_create(int input_dim, int hidden_dim, bool requires_grad);

// 释放前馈网络
void feed_forward_free(FeedForward* ff);

// 前向传播
void feed_forward_forward(
    FeedForward* ff,
    float* input,       // [batch_size, input_dim]
    int batch_size,
    float* output      // [batch_size, input_dim]
);

// 创建编码器层
EncoderLayer* encoder_layer_create(int model_dim, int num_heads, int ff_hidden_dim, bool requires_grad);

// 释放编码器层
void encoder_layer_free(EncoderLayer* encoder);

// 编码器层前向传播
void encoder_layer_forward(
    EncoderLayer* encoder,
    float* input,        // [batch_size, seq_len, model_dim]
    int batch_size,
    int seq_len,
    float* output       // [batch_size, seq_len, model_dim]
);

// 完整编码器结构体
struct Encoder {
    EncoderLayer** layers;    // 编码器层数组
    int num_layers;           // 层数
    int model_dim;            // 模型维度
};

// 完整解码器结构体
struct Decoder {
    DecoderLayer** layers;    // 解码器层数组
    int num_layers;           // 层数
    int model_dim;            // 模型维度
};

// 编码器堆栈函数
Encoder* encoder_create(int num_layers, int model_dim, int num_heads, int ff_hidden_dim, bool requires_grad);
void encoder_free(Encoder* encoder);
void encoder_forward(
    Encoder* encoder,
    float* input,        // [batch_size, src_len, model_dim]
    int batch_size,
    int src_len,
    float* output       // [batch_size, src_len, model_dim]
);

// 解码器堆栈函数
Decoder* decoder_create(int num_layers, int model_dim, int num_heads, int ff_hidden_dim, bool requires_grad);
void decoder_free(Decoder* decoder);
void decoder_forward(
    Decoder* decoder,
    float* input,          // [batch_size, tgt_len, model_dim]
    float* encoder_output, // [batch_size, src_len, model_dim]
    float* tgt_mask,      // [batch_size, tgt_len, tgt_len]
    int batch_size,
    int tgt_len,
    int src_len,
    float* output         // [batch_size, tgt_len, model_dim]
);

// Transformer 结构体
struct Transformer {
    TokenEmbedding* src_embed;    // 源语言词嵌入
    TokenEmbedding* tgt_embed;    // 目标语言词嵌入
    PositionalEncoding* pos_enc;  // 位置编码
    Encoder* encoder;             // 编码器堆栈
    Decoder* decoder;             // 解码器堆栈
    float* linear_weight;         // 输出线性层权重
    float* linear_bias;           // 输出线性层偏置
    int model_dim;                // 模型维度
    int vocab_size;               // 词表大小
};

// Transformer 函数声明
Transformer* transformer_create(
    int vocab_size,
    int model_dim,
    int num_heads,
    int num_layers,
    int ff_hidden_dim,
    bool requires_grad
);

void transformer_free(Transformer* transformer);

// 前向传播（训练模式）
void transformer_forward(
    Transformer* transformer,
    int* src_tokens,      // [batch_size, src_len]
    int* tgt_tokens,      // [batch_size, tgt_len]
    int batch_size,
    int src_len,
    int tgt_len,
    float* output        // [batch_size, tgt_len, vocab_size]
);

// 推理模式（自回归生成）
void transformer_generate(
    Transformer* transformer,
    int* src_tokens,      // [batch_size, src_len]
    int batch_size,
    int src_len,
    int max_len,         // 最大生成长度
    float temperature,   // 采样温度
    int* output_tokens   // [batch_size, max_len]
);

#endif

