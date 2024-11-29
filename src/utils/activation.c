#include "utils.h"
#include <math.h>
#include <float.h>

void softmax(float* x, int size) {
    if (!x || size <= 0) return;
    
    // 找到最大值以防止数值溢出
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    
    // 计算exp并求和
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    
    // 归一化
    if (sum > FLT_EPSILON) {
        for (int i = 0; i < size; i++) {
            x[i] /= sum;
        }
    }
} 