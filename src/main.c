#include "transformer.h"
#include "tensor_type.h"
#include "01attention_mask.h"
#include <stdio.h>

int main() {
    // 配置参数
    int batch_size = 2;
    int enc_seq_len = 10;
    int dec_seq_len = 8;
    int model_dim = 512;
    int num_heads = 8;
    int num_layers = 6;
    int ff_dim = 2048;
    float dropout_prob = 0.1;
    
    // 创建输入张量
    int enc_shape[] = {batch_size, enc_seq_len, model_dim};
    int dec_shape[] = {batch_size, dec_seq_len, model_dim};
    
    Tensor* encoder_input = tensor_create(enc_shape, 3);
    Tensor* decoder_input = tensor_create(dec_shape, 3);
    Tensor* output = tensor_create(dec_shape, 3);

    // 创建注意力掩码
    int enc_mask_shape[] = {batch_size, num_heads, enc_seq_len, enc_seq_len};
    int dec_mask_shape[] = {batch_size, num_heads, dec_seq_len, dec_seq_len};
    int cross_mask_shape[] = {batch_size, num_heads, dec_seq_len, enc_seq_len};
    
    // // 创建encoder的padding掩码
    // AttentionMask* enc_mask = pad_mask_create(encoder_input, encoder_input, num_heads, 0);
    
    // // 创建decoder的因果掩码和padding掩码组合
    // AttentionMask* dec_mask = create_trg_mask(dec_mask_shape, 4, NULL);
    
    // // 创建cross attention的padding掩码
    // AttentionMask* cross_mask = pad_mask_create(decoder_input, encoder_input, num_heads, 0);
    
    // // 创建transformer
    // Transformer* transformer = transformer_create(num_layers, num_heads, model_dim,
    //                                            ff_dim, dropout_prob);
    // if (!transformer) {
    //     printf("Failed to create transformer\n");
    //     return -1;
    // }
    
    // // 初始化输入数据
    // // ... 这里需要设置实际的输入数据 ...
    
    // // 执行前向传播
    // if (!transformer_forward(transformer, encoder_input, decoder_input, output,
    //                        enc_mask, dec_mask, cross_mask)) {
    //     printf("Forward pass failed\n");
    //     return -1;
    // }
    
    // // 处理输出结果
    // // ... 这里处理输出数据 ...
    
    // // 释放资源
    // transformer_free(transformer);
    // tensor_free(encoder_input);
    // tensor_free(decoder_input);
    // tensor_free(output);
    // attention_mask_free(enc_mask);
    // attention_mask_free(dec_mask);
    // attention_mask_free(cross_mask);
    
    // return 0;
}