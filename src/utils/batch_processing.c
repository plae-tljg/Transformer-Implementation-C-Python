#include "training.h"
#include <stdlib.h>

// 添加 min 函数的实现
static inline int min(int a, int b) {
    return (a < b) ? a : b;
}

typedef struct {
    float** data;
    float** targets;
    int batch_size;
    int current_idx;
    int total_samples;
} BatchIterator;

BatchIterator* create_batch_iterator(float** data, float** targets,
                                   int total_samples, int batch_size) {
    BatchIterator* iterator = malloc(sizeof(BatchIterator));
    iterator->data = data;
    iterator->targets = targets;
    iterator->batch_size = batch_size;
    iterator->total_samples = total_samples;
    iterator->current_idx = 0;
    return iterator;
}

int get_next_batch(BatchIterator* iterator, float** batch_data, 
                  float** batch_targets) {
    if (iterator->current_idx >= iterator->total_samples) {
        return 0;
    }
    
    int samples_remaining = iterator->total_samples - iterator->current_idx;
    int current_batch_size = min(iterator->batch_size, samples_remaining);
    
    // 复制批次数据
    for (int i = 0; i < current_batch_size; i++) {
        batch_data[i] = iterator->data[iterator->current_idx + i];
        batch_targets[i] = iterator->targets[iterator->current_idx + i];
    }
    
    iterator->current_idx += current_batch_size;
    return current_batch_size;
} 