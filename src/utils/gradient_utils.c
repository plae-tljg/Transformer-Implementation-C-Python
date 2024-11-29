#include "../../include/utils.h"
#include "../../include/embeddings.h"
#include <math.h>

float compute_gradient_norm(float* gradients, int size) {
    if (!gradients) return 0.0f;
    
    float norm = 0.0f;
    for (int i = 0; i < size; i++) {
        norm += gradients[i] * gradients[i];
    }
    return norm;
}

void scale_gradients(float* gradients, int size, float scale) {
    if (!gradients) return;
    
    for (int i = 0; i < size; i++) {
        gradients[i] *= scale;
    }
}

void update_embedding_gradients(TokenEmbedding* embedding, float* grad) {
    if (!embedding || !grad) return;
    
    for (int i = 0; i < embedding->vocab_size * embedding->embedding_dim; i++) {
        embedding->embedding_gradients[i] += grad[i];
    }
} 