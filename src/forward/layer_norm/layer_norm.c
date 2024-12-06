#include "layer_norm.h"
#include <stdlib.h>
#include <math.h>

LayerNorm* layer_norm_create(int normalized_shape, float eps) {
    LayerNorm* ln = (LayerNorm*)malloc(sizeof(LayerNorm));
    ln->eps = eps;
    
    // 创建gamma和beta参数，初始化gamma为1，beta为0
    int shape[] = {normalized_shape};
    ln->gamma = tensor_create(shape, 1);
    ln->beta = tensor_create(shape, 1);
    
    // 初始化gamma为1
    for (int i = 0; i < normalized_shape; i++) {
        ln->gamma->data[i] = 1.0f;
    }
    // 初始化beta为0
    for (int i = 0; i < normalized_shape; i++) {
        ln->beta->data[i] = 0.0f;
    }
    
    ln->normalized_dim = normalized_shape;
    return ln;
}

void layer_norm_free(LayerNorm* ln) {
    if (ln) {
        tensor_free(ln->gamma);
        tensor_free(ln->beta);
        free(ln);
    }
}

Tensor* layer_norm_forward(LayerNorm* ln, Tensor* input) {
    Tensor* output = tensor_create(input->shape, input->num_dims);
    
    layer_norm_forward_3d(
        input->data,
        output->data,
        ln->gamma->data,
        ln->beta->data,
        input->shape[0],  // batch_size
        input->shape[1],  // seq_len
        ln->normalized_dim,
        ln->eps
    );
    
    return output;
}
