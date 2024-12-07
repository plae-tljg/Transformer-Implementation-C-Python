#include <stdlib.h>
#include <stdio.h>
#include "01attention_mask.h"
#include "tensor_logic.h"
#include "tensor_trio.h"

// 创建注意力掩码, used when in Q@K in multiattention
AttentionMask* pad_mask_create(Tensor* q, Tensor* k, int num_heads, int pad_token_id) {
    if (!q || !k) {
        fprintf(stderr, "Input tensors Q and K cannot be NULL\n");
        return NULL;
    }

    int batch_size = q->shape[0];
    int q_seq_len = q->shape[1];
    int k_seq_len = k->shape[1];

    AttentionMask* mask = (AttentionMask*)malloc(sizeof(AttentionMask));
    if (!mask) {
        fprintf(stderr, "Failed to allocate memory for attention mask\n");
        return NULL;
    }

    mask->seq_length = k_seq_len;
    
    // 创建4D掩码张量 [batch_size, num_heads, q_seq_len, k_seq_len]
    int mask_shape[] = {batch_size, num_heads, q_seq_len, k_seq_len}; 
    mask->mask = tensor_create(mask_shape, 4);
    if (!mask->mask) {
        fprintf(stderr, "Failed to allocate memory for mask tensor\n");
        free(mask);
        return NULL;
    }

    // 初始化掩码
    tensor_create_pad_mask(mask->mask, batch_size, q_seq_len, k, pad_token_id);

    return mask;
}

// 创建因果掩码 - 用于decoder的自注意力掩码
AttentionMask* create_causal_mask(int* shape, int num_dims) {
    // 检查维度 - 应该是3维 [batch_size, seq_len, seq_len]
    if (num_dims != 3) {
        fprintf(stderr, "因果掩码需要3个维度 [batch_size, seq_len, seq_len]\n");
        return NULL;
    }

    AttentionMask* mask = (AttentionMask*)malloc(sizeof(AttentionMask));
    if (!mask) {
        fprintf(stderr, "为因果掩码分配内存失败\n");
        return NULL;
    }

    mask->seq_length = shape[1]; // seq_len
    mask->mask = tensor_create(shape, num_dims);
    if (!mask->mask) {
        fprintf(stderr, "为掩码张量分配内存失败\n");
        free(mask);
        return NULL;
    }

    // 创建下三角矩阵掩码
    if (!tensor_create_causal_mask(mask->mask)) {
        tensor_free(mask->mask);
        free(mask);
        return NULL;
    }

    return mask;
}

// 创建目标掩码 - 用于decoder的自注意力掩码,考虑padding
AttentionMask* create_trg_mask(int* shape, int num_dims) {
    // 创建因果掩码
    AttentionMask* causal_mask = create_causal_mask(shape, num_dims);
    if (!causal_mask) return NULL;

    // 创建padding掩码
    Tensor* padding_mask = tensor_create(shape, num_dims);
    if (!padding_mask) {
        attention_mask_free(causal_mask);
        return NULL;
    }

    // 初始化padding掩码
    if (!tensor_create_pad_mask(padding_mask, shape[0], shape[1], NULL, 0)) {
        tensor_free(padding_mask);
        attention_mask_free(causal_mask);
        return NULL;
    }

    // 将padding mask与因果掩码结合
    bool success = tensor_and(causal_mask->mask, padding_mask, causal_mask->mask);
    if (!success) {
        tensor_free(padding_mask);
        attention_mask_free(causal_mask);
        return NULL;
    }

    tensor_free(padding_mask);
    return causal_mask;
}



// 释放注意力掩码
void attention_mask_free(AttentionMask* mask) {
    if (!mask) return;
    tensor_free(mask->mask);
    free(mask);
} 

// 应用注意力掩码到注意力分数上
bool apply_attention_mask(
    const Tensor* scores,  // [batch_size, num_heads, seq_len_q, seq_len_k]
    const AttentionMask* mask,  // [batch_size, num_heads, seq_len_q, seq_len_k]
    Tensor* output  // [batch_size, num_heads, seq_len_q, seq_len_k]
) {
    if (!scores || !mask || !output) {
        fprintf(stderr, "输入参数不能为空\n");
        return false;
    }

    // 检查维度
    if (scores->num_dims != 4 || output->num_dims != 4) {
        fprintf(stderr, "scores和output必须是4维张量\n");
        return false;
    }

    // 检查形状匹配
    if (scores->shape[2] != mask->mask->shape[0] || 
        scores->shape[3] != mask->mask->shape[1]) {
        fprintf(stderr, "scores和mask的序列长度不匹配\n");
        return false;
    }

    // 应用掩码
    const float MASKING_VALUE = -1e9f;
    bool success = tensor_apply_mask(scores, mask->mask, output, MASKING_VALUE);

    return success;
}