#include "training.h"
#include "types.h"
#include <stdlib.h>
#include <string.h>

float* compute_loss_gradient(float* output, float* target, int size) {
    if (!output || !target || size <= 0) return NULL;
    
    float* gradient = malloc(size * sizeof(float));
    if (!gradient) return NULL;
    
    for (int i = 0; i < size; i++) {
        gradient[i] = 2.0f * (output[i] - target[i]);
    }
    
    return gradient;
}

float compute_loss(float* predictions, float* targets, int size) {
    if (!predictions || !targets || size <= 0) return 0.0f;
    
    float loss = 0.0f;
    for (int i = 0; i < size; i++) {
        float diff = predictions[i] - targets[i];
        loss += diff * diff;
    }
    
    return loss / size;
} 