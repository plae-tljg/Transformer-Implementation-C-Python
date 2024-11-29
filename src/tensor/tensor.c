#include "tensor.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <stdio.h>

Tensor* tensor_create(int* dims, int ndims, bool is_param){
    Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
    if (tensor == NULL) {
        fprintf(stderr, "Failed to allocate memory for tensor\n");
        exit(EXIT_FAILURE);
    }
    tensor->dims = (int*)malloc(ndims * sizeof(int));
    tensor->ndims = ndims;
    tensor->is_param = is_param;
    memcpy(tensor->dims, dims, ndims * sizeof(int));
    
    // 计算总元素数并为data分配内存
    int total_elements = 1;
    for (int i = 0; i < ndims; i++) {
        total_elements *= dims[i];
    }
    tensor->data = (float*)malloc(total_elements * sizeof(float));
    if (tensor->data == NULL) {
        fprintf(stderr, "Failed to allocate memory for tensor data\n");
        free(tensor->dims);
        free(tensor);
        exit(EXIT_FAILURE);
    }
    
    return tensor;
}

// Function to insert data into the tensor
void tensor_insert(Tensor* tensor, float* data) {
    if (tensor == NULL || data == NULL) {
        fprintf(stderr, "Error: tensor or data is NULL in tensor_insert\n");
        return;
    }

    int total_elements = 1;
    for (int i = 0; i < tensor->ndims; i++) {
        total_elements *= tensor->dims[i];
    }

    memcpy(tensor->data, data, total_elements * sizeof(float));
}

// 将print_recursive函数移到tensor_print外部
void print_recursive(Tensor* tensor, int* indices, int* strides, int depth) {
    if (depth == tensor->ndims) {
        int index = 0;
        for (int i = 0; i < tensor->ndims; i++) {
            index += indices[i] * strides[i];
        }
        printf("%.2f ", tensor->data[index]);
        return;
    }

    for (int i = 0; i < tensor->dims[depth]; i++) {
        if (i == 0) {
            for (int j = 0; j < depth; j++) printf("  ");
            printf("[\n");
        }
        
        indices[depth] = i;
        print_recursive(tensor, indices, strides, depth + 1);
        
        if (i == tensor->dims[depth] - 1) {
            printf("\n");
            for (int j = 0; j < depth; j++) printf("  ");
            printf("]");
            if (depth > 0) printf(",");
            printf("\n");
        } else if (depth == tensor->ndims - 1) {
            printf(" ");
        }
    }
}

void tensor_print(Tensor* tensor) {
    if (tensor == NULL) {
        fprintf(stderr, "Error: tensor is NULL in tensor_print\n");
        return;
    }

    printf("Tensor data (shape: [");
    for (int i = 0; i < tensor->ndims; i++) {
        printf("%d%s", tensor->dims[i], i < tensor->ndims - 1 ? ", " : "");
    }
    printf("]):\n");

    int* strides = (int*)malloc(tensor->ndims * sizeof(int));
    strides[tensor->ndims - 1] = 1;
    for (int i = tensor->ndims - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * tensor->dims[i + 1];
    }

    int* indices = (int*)calloc(tensor->ndims, sizeof(int));
    
    print_recursive(tensor, indices, strides, 0);
    
    free(strides);
    free(indices);
}

void tensor_free(Tensor* tensor) {
    // 如果张量为空则直接返回
    if (!tensor) {
        return;
    }
    
    // 释放数据内存
    if (tensor->data) {
        free(tensor->data);
    }
    
    // 释放维度数组
    if (tensor->dims) {
        free(tensor->dims);
    }
    
    // ndims和is_param是基本类型,不需要释放
    
    // 释放张量结构体本身
    free(tensor);
}

// 检查两个张量的维度是否兼容
static bool check_dimensions_match(Tensor* a, Tensor* b) {
    if (a->ndims != b->ndims) return false;
    for (int i = 0; i < a->ndims; i++) {
        if (a->dims[i] != b->dims[i]) return false;
    }
    return true;
}

// 计算张量的总元素数
static int get_total_elements(Tensor* t) {
    int total = 1;
    for (int i = 0; i < t->ndims; i++) {
        total *= t->dims[i];
    }
    return total;
}

Tensor* tensor_add(Tensor* a, Tensor* b) {
    if (!a || !b) {
        fprintf(stderr, "Error: NULL tensor in tensor_add\n");
        return NULL;
    }
    
    if (!check_dimensions_match(a, b)) {
        fprintf(stderr, "Error: Dimension mismatch in tensor_add\n");
        return NULL;
    }
    
    Tensor* result = tensor_create(a->dims, a->ndims, false);
    int total_elements = get_total_elements(a);
    
    for (int i = 0; i < total_elements; i++) {
        result->data[i] = a->data[i] + b->data[i];
    }
    
    return result;
}

Tensor* tensor_sub(Tensor* a, Tensor* b) {
    if (!a || !b) {
        fprintf(stderr, "Error: NULL tensor in tensor_sub\n");
        return NULL;
    }
    
    if (!check_dimensions_match(a, b)) {
        fprintf(stderr, "Error: Dimension mismatch in tensor_sub\n");
        return NULL;
    }
    
    Tensor* result = tensor_create(a->dims, a->ndims, false);
    int total_elements = get_total_elements(a);
    
    for (int i = 0; i < total_elements; i++) {
        result->data[i] = a->data[i] - b->data[i];
    }
    
    return result;
}

Tensor* tensor_mul(Tensor* a, Tensor* b) {
    if (!a || !b) {
        fprintf(stderr, "Error: NULL tensor in tensor_mul\n");
        return NULL;
    }
    
    if (!check_dimensions_match(a, b)) {
        fprintf(stderr, "Error: Dimension mismatch in tensor_mul\n");
        return NULL;
    }
    
    Tensor* result = tensor_create(a->dims, a->ndims, false);
    int total_elements = get_total_elements(a);
    
    for (int i = 0; i < total_elements; i++) {
        result->data[i] = a->data[i] * b->data[i];
    }
    
    return result;
}

void tensor_fill(Tensor* a, float value) {
    if (!a) {
        fprintf(stderr, "Error: NULL tensor in tensor_fill\n");
        return;
    }
    
    int total_elements = get_total_elements(a);
    for (int i = 0; i < total_elements; i++) {
        a->data[i] = value;
    }
}

bool tensor_equal(Tensor* a, Tensor* b) {
    if (!a || !b) return false;
    if (!check_dimensions_match(a, b)) return false;
    
    int total_elements = get_total_elements(a);
    for (int i = 0; i < total_elements; i++) {
        if (a->data[i] != b->data[i]) return false;
    }
    
    return true;
}

Tensor* tensor_relu(Tensor* a) {
    if (!a) {
        fprintf(stderr, "Error: NULL tensor in tensor_relu\n");
        return NULL;
    }
    
    Tensor* result = tensor_create(a->dims, a->ndims, false);
    int total_elements = get_total_elements(a);
    
    for (int i = 0; i < total_elements; i++) {
        result->data[i] = a->data[i] > 0 ? a->data[i] : 0;
    }
    
    return result;
}

Tensor* tensor_sigmoid(Tensor* a) {
    if (!a) {
        fprintf(stderr, "Error: NULL tensor in tensor_sigmoid\n");
        return NULL;
    }
    
    Tensor* result = tensor_create(a->dims, a->ndims, false);
    int total_elements = get_total_elements(a);
    
    for (int i = 0; i < total_elements; i++) {
        result->data[i] = 1.0f / (1.0f + expf(-a->data[i]));
    }
    
    return result;
}

void tensor_random_uniform(Tensor* a, float min, float max) {
    if (!a) {
        fprintf(stderr, "Error: NULL tensor in tensor_random_uniform\n");
        return;
    }
    
    int total_elements = get_total_elements(a);
    for (int i = 0; i < total_elements; i++) {
        float random = (float)rand() / RAND_MAX;  // 生成0到1之间的随机数
        a->data[i] = min + random * (max - min);  // 将随机数映射到[min, max]范围
    }
}

// 检查两个张量是否可以进行矩阵乘法
static bool check_matmul_dimensions(Tensor* a, Tensor* b) {
    // 确保两个张量至少是2维的
    if (a->ndims < 2 || b->ndims < 2) return false;
    
    // 检查最后两个维度是否满足矩阵乘法条件
    // a的最后一个维度必须等于b的倒数第二个维度
    return a->dims[a->ndims-1] == b->dims[b->ndims-2];
}

// 矩阵乘法实现
Tensor* tensor_matmul(Tensor* a, Tensor* b) {
    if (!a || !b) {
        fprintf(stderr, "Error: NULL tensor in tensor_matmul\n");
        return NULL;
    }
    
    if (!check_matmul_dimensions(a, b)) {
        fprintf(stderr, "Error: Invalid dimensions for matrix multiplication\n");
        return NULL;
    }
    
    // 获取矩阵维度
    int m = a->dims[a->ndims-2];  // a的行数
    int k = a->dims[a->ndims-1];  // a的列数 = b的行数
    int n = b->dims[b->ndims-1];  // b的列数
    
    // 创建结果张量
    // 结果维度将是 [..., m, n]
    int* result_dims = (int*)malloc(a->ndims * sizeof(int));
    memcpy(result_dims, a->dims, (a->ndims-2) * sizeof(int));
    result_dims[a->ndims-2] = m;
    result_dims[a->ndims-1] = n;
    
    Tensor* result = tensor_create(result_dims, a->ndims, false);
    free(result_dims);
    
    // 计算batch size（如果有的话）
    int batch_size = 1;
    for (int i = 0; i < a->ndims-2; i++) {
        batch_size *= a->dims[i];
    }
    
    // 对每个batch执行矩阵乘法
    for (int batch = 0; batch < batch_size; batch++) {
        int offset = batch * m * n;
        // 标准矩阵乘法 C = A * B
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                float sum = 0.0f;
                for (int p = 0; p < k; p++) {
                    sum += a->data[batch * m * k + i * k + p] * 
                          b->data[batch * k * n + p * n + j];
                }
                result->data[offset + i * n + j] = sum;
            }
        }
    }
    
    return result;
}

int main() {
    int dims[] = {2, 3, 4};  // Example: 2x3x4 tensor
    int ndims = 3;
    Tensor* my_tensor = tensor_create(dims, ndims, true);
    printf("创建张量: 维度=[%d, %d, %d], 参数=%s\n", 
           dims[0], dims[1], dims[2],
           my_tensor->is_param ? "true" : "false");
           
    if (my_tensor != NULL) {
        float my_data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0}; // Data to insert
        tensor_insert(my_tensor, my_data);

        tensor_print(my_tensor);  // Print the tensor

        tensor_free(my_tensor);
    }

    // 测试矩阵乘法
    printf("\n测试矩阵乘法:\n");
    
    // 创建两个矩阵 A(2x3) 和 B(3x2)
    int dims_a[] = {2, 3};
    int dims_b[] = {3, 2}; 
    Tensor* A = tensor_create(dims_a, 2, false);
    Tensor* B = tensor_create(dims_b, 2, false);
    
    // 初始化矩阵A的数据
    float data_a[] = {1.0, 2.0, 3.0,
                      4.0, 5.0, 6.0};
    tensor_insert(A, data_a);
    
    // 初始化矩阵B的数据
    float data_b[] = {1.0, 2.0,
                      3.0, 4.0,
                      5.0, 6.0};
    tensor_insert(B, data_b);
    
    printf("矩阵 A:\n");
    tensor_print(A);
    printf("矩阵 B:\n");
    tensor_print(B);
    
    // 执行矩阵乘法
    Tensor* C = tensor_matmul(A, B);
    printf("矩阵乘法结果 C = A * B:\n");
    tensor_print(C);
    
    // 释放内存
    tensor_free(A);
    tensor_free(B);
    tensor_free(C);
    
    // 测试批量矩阵乘法
    printf("\n测试批量矩阵乘法:\n");
    
    // 创建两个批量矩阵 batch_A(2x2x3) 和 batch_B(2x3x2)
    int batch_dims_a[] = {2, 2, 3};
    int batch_dims_b[] = {2, 3, 2};
    Tensor* batch_A = tensor_create(batch_dims_a, 3, false);
    Tensor* batch_B = tensor_create(batch_dims_b, 3, false);
    
    // 初始化批量矩阵A的数据
    float batch_data_a[] = {1.0, 2.0, 3.0,
                           4.0, 5.0, 6.0,
                           7.0, 8.0, 9.0,
                           10.0, 11.0, 12.0};
    tensor_insert(batch_A, batch_data_a);
    
    // 初始化批量矩阵B的数据
    float batch_data_b[] = {1.0, 2.0,
                           3.0, 4.0,
                           5.0, 6.0,
                           7.0, 8.0,
                           9.0, 10.0,
                           11.0, 12.0};
    tensor_insert(batch_B, batch_data_b);
    
    printf("批量矩阵 A:\n");
    tensor_print(batch_A);
    printf("批量矩阵 B:\n");
    tensor_print(batch_B);
    
    // 执行批量矩阵乘法
    Tensor* batch_C = tensor_matmul(batch_A, batch_B);
    printf("批量矩阵乘法结果 C = A * B:\n");
    tensor_print(batch_C);
    
    // 释放内存
    tensor_free(batch_A);
    tensor_free(batch_B);
    tensor_free(batch_C);

    return 0;
}