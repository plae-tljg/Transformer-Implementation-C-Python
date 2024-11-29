#ifndef TYPES_H
#define TYPES_H

#include <stdbool.h>

// 前向声明所有类型
typedef struct ModelConfig ModelConfig;
typedef struct TransformerModel TransformerModel;
typedef struct EncoderLayer EncoderLayer;
typedef struct DecoderLayer DecoderLayer;
typedef struct SelfAttention SelfAttention;
typedef struct MultiHeadAttention MultiHeadAttention;
typedef struct FeedForward FeedForward;
typedef struct LayerNorm LayerNorm;
typedef struct PositionalEncoding PositionalEncoding;
typedef struct TokenEmbedding TokenEmbedding;

// 层归一化结构
struct LayerNorm {
    int dim;                // 归一化维度
    float epsilon;          // 数值稳定性参数
    bool requires_grad;     // 是否需要梯度
    
    float* gamma;           // 缩放参数
    float* beta;            // 偏移参数
    float* gamma_grad;      // gamma的梯度
    float* beta_grad;       // beta的梯度
    
    // 缓存用于反向传播
    float* mean_cache;      // 均值缓存
    float* var_cache;       // 方差缓存
    float* norm_cache;      // 归一化后的值缓存
    float* input_cache;     // 输入缓存
};



// 前馈神经网络结构
struct FeedForward {
    int input_dim;
    int hidden_dim;
    bool requires_grad;
    
    // 权重和偏置
    float* weight1;  // input_dim x hidden_dim
    float* weight2;  // hidden_dim x input_dim
    float* bias1;    // hidden_dim
    float* bias2;    // input_dim
    
    // 梯度
    float* weight1_gradients;
    float* weight2_gradients;
    float* bias1_gradients;
    float* bias2_gradients;
    
    // 缓存用于反向传播
    float* hidden_layer_output;
    float* activation_output;
    float* input_cache;
};


#pragma region Attention Structures

// 自注意力结构
struct SelfAttention {
    int head_dim;
    int num_heads;
    
    // 权重
    float* query_weights;
    float* key_weights;
    float* value_weights;
    float* output_weights;
    
    // 偏置
    float* query_bias;
    float* key_bias;
    float* value_bias;
    float* output_bias;
    
    // 梯度
    float* query_gradients;
    float* key_gradients;
    float* value_gradients;
    float* output_gradients;
    
    // 偏置梯度
    float* query_bias_gradients;
    float* key_bias_gradients;
    float* value_bias_gradients;
    float* output_bias_gradients;
    
    bool requires_grad;
};

// 多头注意力结构
struct MultiHeadAttention {
    int num_heads;
    int model_dim;
    int head_dim;
    
    // 注意力头
    SelfAttention** attention_heads;
    
    // 输出层
    float* output_weights;
    float* output_bias;
    
    // 梯度
    float* output_weights_gradients;
    float* output_bias_gradients;
    float* attention_weights_gradients;
    
    // 缓存用于反向传播
    float* input_cache;
    float* attention_scores_cache;
    float* attention_output_cache;
    
    bool requires_grad;
};
#pragma endregion

#pragma region Model_and_Training_Structures

// 模型配置结构
struct ModelConfig {
    int model_dim;
    int num_heads;
    int vocab_size;
    int max_seq_length;
    int num_layers;
    float learning_rate;
    float dropout_rate;
    bool use_bias;
    int warmup_steps;
    float weight_decay;
    bool use_layer_norm;
    int seed;
};

// Transformer模型结构
struct TransformerModel {
    ModelConfig* config;              // 模型配置
    int model_dim;                    // 模型维度
    int max_seq_length;              // 最大序列长度
    int num_encoder_layers;          // 编码器层数
    int num_decoder_layers;          // 解码器层数
    EncoderLayer** encoder_layers;  // 编码器层数组
    DecoderLayer** decoder_layers;  // 解码器层数组
};

// 训练配置结构
struct TrainingConfig { 
    float learning_rate;        // 学习率
    int batch_size;            // 批次大小
    int max_epochs;            // 训练轮数
    float dropout_rate;        // Dropout率
    float gradient_clip_value; // 梯度裁剪值
    int warmup_steps;          // 预热步数
    bool use_gradient_clipping; // 是否使用梯度裁剪
    int save_interval;         // 保存间隔
    char* checkpoint_dir;      // 检查点目录
    int vocab_size;            // 词汇表大小
    int max_seq_length;        // 最大序列长度
    int model_dim;            // 模型维度
    int num_heads;            // 注意力头数
    int num_layers;           // 层数
    bool use_bias;            // 是否使用偏置
    float weight_decay;       // 权重衰减
    int seed;                 // 随机种子
};
#pragma endregion

#pragma region Embeddings
// 位置编码结构
struct PositionalEncoding {
    float* encodings;          // 位置编码矩阵
    int max_seq_length;        // 最大序列长度
    int encoding_dim;          // 编码维度
};

struct TokenEmbedding {
    float* embedding_matrix;    // 嵌入矩阵
    float* embedding_gradients; // 嵌入梯度
    int vocab_size;            // 词汇表大小
    int embedding_dim;         // 嵌入维度
    bool requires_grad;        // 是否需要梯度
};
#pragma endregion

#endif // TYPES_H 