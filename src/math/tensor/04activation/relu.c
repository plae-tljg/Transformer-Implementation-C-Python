#include "relu.h"
#include <stdio.h>

void relu_forward(Tensor* input, Tensor* output) {
    size_t size = calculate_total_size(input->shape, input->num_dims);
    for (int i = 0; i < size; i++) {
        output->data[i] = input->data[i] > 0 ? input->data[i] : 0;
    }
}