#include "training.h"
#include "grad.h"
#include "model.h"
#include "layers.h"
#include <stdlib.h>
#include <math.h>

float train_batch(
    Transformer* model,
    int* src_tokens,
    int* tgt_tokens,
    int batch_size,
    int src_len,
    int tgt_len,
    TrainingConfig* config,
    OptimizerState* optimizer,
    TransformerGrad* grad
) {
    // 1. 前向传播
    float* encoder_output = (float*)malloc(batch_size * src_len * model->model_dim * sizeof(float));
    float* decoder_output = (float*)malloc(batch_size * tgt_len * model->model_dim * sizeof(float));
    float* output = (float*)malloc(batch_size * tgt_len * model->vocab_size * sizeof(float));
    
    transformer_forward(
        model, 
        src_tokens, 
        tgt_tokens, 
        batch_size, 
        src_len, 
        tgt_len, 
        encoder_output,
        decoder_output,
        output
    );
    
    // 2. 计算损失和损失梯度
    float* loss_grad = (float*)malloc(batch_size * tgt_len * model->vocab_size * sizeof(float));
    float total_loss = 0.0f;
    
    for (int b = 0; b < batch_size; b++) {
        for (int t = 0; t < tgt_len - 1; t++) {
            int target = tgt_tokens[b * tgt_len + t + 1];
            
            // 计算交叉熵损失
            float* logits = &output[(b * tgt_len + t) * model->vocab_size];
            float max_logit = logits[0];
            for (int v = 1; v < model->vocab_size; v++) {
                if (logits[v] > max_logit) max_logit = logits[v];
            }
            
            float sum_exp = 0.0f;
            for (int v = 0; v < model->vocab_size; v++) {
                sum_exp += expf(logits[v] - max_logit);
            }
            float log_sum_exp = logf(sum_exp) + max_logit;
            total_loss -= logits[target] - log_sum_exp;
            
            // 计算softmax梯度
            for (int v = 0; v < model->vocab_size; v++) {
                float prob = expf(logits[v] - log_sum_exp);
                loss_grad[(b * tgt_len + t) * model->vocab_size + v] = prob;
            }
            loss_grad[(b * tgt_len + t) * model->vocab_size + target] -= 1.0f;
        }
    }
    
    // 3. 反向传播
    transformer_backward(
        model,
        loss_grad,
        encoder_output,
        decoder_output,
        batch_size,
        src_len,
        tgt_len,
        grad
    );
    
    // 4. 应用梯度更新参数
    apply_gradients(
        model,
        grad,
        optimizer,
        config->learning_rate,
        config->weight_decay
    );
    
    // 释放内存
    free(encoder_output);
    free(decoder_output);
    free(output);
    free(loss_grad);
    
    return total_loss / (batch_size * (tgt_len - 1));
}

void train_epoch(
    Transformer* model,
    int* src_data,      // [num_samples, src_len]
    int* tgt_data,      // [num_samples, tgt_len]
    int num_samples,
    int src_len,
    int tgt_len,
    TrainingConfig* config,
    OptimizerState* optimizer
) {
    // 创建梯度结构
    TransformerGrad* grad = create_transformer_grad(model);
    
    // 计算每个epoch的批次数
    int num_batches = (num_samples + config->batch_size - 1) / config->batch_size;
    
    for (int batch = 0; batch < num_batches; batch++) {
        // 准备当前批次的数据
        int current_batch_size = (batch == num_batches - 1) ? 
            (num_samples - batch * config->batch_size) : config->batch_size;
            
        // 获取当前批次的源语言和目标语言数据
        int* batch_src = &src_data[batch * config->batch_size * src_len];
        int* batch_tgt = &tgt_data[batch * config->batch_size * tgt_len];
        
        // 训练一个批次
        train_batch(
            model,
            batch_src,
            batch_tgt,
            current_batch_size,
            src_len,
            tgt_len,
            config,
            optimizer,
            grad
        );
        
        // 清零梯度，准备下一个批次
        zero_transformer_grad(grad);
        
        // 打印训练进度
        if (batch % 100 == 0) {
            printf("Batch %d/%d completed\n", batch, num_batches);
        }
    }
    
    // 释放梯度结构
    free_transformer_grad(grad);
}
