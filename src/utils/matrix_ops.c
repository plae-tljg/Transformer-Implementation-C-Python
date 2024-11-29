#include "utils.h"
#include <stdlib.h>
#include <math.h>

float* matrix_multiply(float* a, float* b, int m, int n, int p) {
    if (!a || !b) return NULL;
    
    float* result = calloc(m * p, sizeof(float));
    if (!result) return NULL;
    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            float sum = 0;
            for (int k = 0; k < n; k++) {
                sum += a[i * n + k] * b[k * p + j];
            }
            result[i * p + j] = sum;
        }
    }
    
    return result;
}

// void softmax(float* x, int size) {
//     if (!x || size <= 0) return;
    
//     float max_val = x[0];
//     for (int i = 1; i < size; i++) {
//         if (x[i] > max_val) max_val = x[i];
//     }
    
//     float sum = 0.0f;
//     for (int i = 0; i < size; i++) {
//         x[i] = expf(x[i] - max_val);
//         sum += x[i];
//     }
    
//     if (sum != 0) {
//         for (int i = 0; i < size; i++) {
//             x[i] /= sum;
//         }
//     }
// }

float gelu(float x) {
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coef = 0.044715f;
    return 0.5f * x * (1.0f + tanhf(sqrt_2_over_pi * (x + coef * x * x * x)));
}

float* matrix_transpose(float* matrix, int rows, int cols) {
    if (!matrix) return NULL;
    
    float* result = malloc(rows * cols * sizeof(float));
    if (!result) return NULL;
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[j * rows + i] = matrix[i * cols + j];
        }
    }
    
    return result;
}

void matrix_add_inplace(float* a, float* b, int size) {
    if (!a || !b || size <= 0) return;
    
    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        a[i] += b[i];
    }
}

void matrix_scale(float* matrix, float scalar, int size) {
    if (!matrix) return;
    
    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        matrix[i] *= scalar;
    }
}

float* matrix_multiply_transpose(float* a, float* b, int m, int k, int n) {
    float* result = malloc(m * n * sizeof(float));
    if (!result) return NULL;
    
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int l = 0; l < k; l++) {
                sum += a[i * k + l] * b[j * k + l];  // b is transposed
            }
            result[i * n + j] = sum;
        }
    }
    
    return result;
}

float* matrix_multiply_transpose2(float* a, float* b, int m, int k, int n) {
    float* result = malloc(m * n * sizeof(float));
    if (!result) return NULL;
    
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int l = 0; l < k; l++) {
                sum += a[l * m + i] * b[l * n + j];  // a is transposed
            }
            result[i * n + j] = sum;
        }
    }
    
    return result;
} 