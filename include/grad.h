#ifndef GRAD_H
#define GRAD_H

#include "layers.h"
#include "training.h"

typedef struct TokenEmbeddingGrad TokenEmbeddingGrad;
typedef struct TransformerGrad TransformerGrad;
typedef struct TrainingState TrainingState;
typedef struct FeedForwardGrad FeedForwardGrad;
typedef struct LayerNormGrad LayerNormGrad;
typedef struct MultiHeadAttentionGrad MultiHeadAttentionGrad;
typedef struct EncoderLayerGrad EncoderLayerGrad;
typedef struct DecoderLayerGrad DecoderLayerGrad;

// 词嵌入的梯度
struct TokenEmbeddingGrad {
    float* grad_embedding;  // [vocab_size, embedding_dim]
    float* grad_position_embedding;  // [max_seq_len, embedding_dim] (如果使用可学习的位置编码)
};

// 前馈网络的梯度
struct FeedForwardGrad {
    float* grad_w1;  // [hidden_dim, input_dim]
    float* grad_b1;  // [hidden_dim]
    float* grad_w2;  // [input_dim, hidden_dim]
    float* grad_b2;  // [input_dim]
};

// Layer Norm的梯度
struct LayerNormGrad {
    float* grad_gamma;  // [normalized_shape]
    float* grad_beta;   // [normalized_shape]
};

// 多头注意力的梯度
struct MultiHeadAttentionGrad {
    float* grad_W_q;  // [model_dim, model_dim]
    float* grad_W_k;  // [model_dim, model_dim]
    float* grad_W_v;  // [model_dim, model_dim]
    float* grad_W_o;  // [model_dim, model_dim]
};

// 编码器层的梯度
struct EncoderLayerGrad {
    MultiHeadAttentionGrad* self_attention_grad;
    LayerNormGrad* norm1_grad;
    LayerNormGrad* norm2_grad;
    FeedForwardGrad* feed_forward_grad;
};

// 解码器层的梯度
struct DecoderLayerGrad {
    MultiHeadAttentionGrad* self_attention_grad;
    MultiHeadAttentionGrad* cross_attention_grad;
    LayerNormGrad* norm1_grad;
    LayerNormGrad* norm2_grad;
    LayerNormGrad* norm3_grad;
    FeedForwardGrad* feed_forward_grad;
};

// Transformer的梯度
struct TransformerGrad {
    float* grad_src_embed;     // [vocab_size, model_dim]
    float* grad_tgt_embed;     // [vocab_size, model_dim]
    float* grad_linear_weight; // [vocab_size, model_dim]
    float* grad_linear_bias;   // [vocab_size]
    EncoderLayerGrad** encoder_layer_grads;  // [num_layers]
    DecoderLayerGrad** decoder_layer_grads;  // [num_layers]
};

// 训练一个批次
float train_batch(
    Transformer* model,
    int* src_tokens,      // [batch_size, src_len]
    int* tgt_tokens,      // [batch_size, tgt_len]
    int batch_size,
    int src_len,
    int tgt_len,
    TrainingConfig* config,
    OptimizerState* optimizer,
    TransformerGrad* grad
);

// 训练一个epoch
void train_epoch(
    Transformer* model,
    int* train_data,     // [num_samples, max_len]
    int num_samples,
    int max_len,
    TrainingConfig* config,
    OptimizerState* optimizer
);

// 添加函数声明
void update_embedding_params(
    TokenEmbedding* emb,
    TokenEmbeddingGrad* grad,
    OptimizerState* opt,
    float lr,
    float weight_decay
);

void update_encoder_layer_params(
    EncoderLayer* layer,
    EncoderLayerGrad* grad,
    OptimizerState* opt,
    float lr,
    float weight_decay
);

// 创建和释放梯度结构的函数
TransformerGrad* create_transformer_grad(Transformer* model);
void free_transformer_grad(TransformerGrad* grad);

// 清零梯度
void zero_transformer_grad(TransformerGrad* grad);

// 训练状态结构
struct TrainingState {
    TransformerGrad* gradients;
    float* optimizer_state;  // Adam优化器状态
    bool is_training;
};

// 创建训练状态
TrainingState* create_training_state(Transformer* model);
void free_training_state(TrainingState* state);

// 创建和释放词嵌入梯度的函数
TokenEmbeddingGrad* create_token_embedding_grad(int vocab_size, int embedding_dim, bool has_position_embedding);
void free_token_embedding_grad(TokenEmbeddingGrad* grad);


#endif // GRAD_H