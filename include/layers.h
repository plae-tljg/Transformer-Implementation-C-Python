#ifndef LAYERS_H
#define LAYERS_H

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

// 位置编码结构
struct PositionalEncoding {
    int max_seq_length;
    int encoding_dim;
    float* encodings;
};

// Token嵌入结构
struct TokenEmbedding {
    int vocab_size;
    int embedding_dim;
    float* weights;
    float* gradients;
    bool requires_grad;
};

#endif

