#include "position_lookup.h"
#include <math.h>

// 新增：位置编码计算函数
bool compute_positional_encodings(
    Tensor* encodings,
    int batch_size,
    int max_seq_length,
    int encoding_dim
) {
    if (!encodings) return false;

    for (int batch = 0; batch < batch_size; batch++) {
        for (int pos = 0; pos < max_seq_length; pos++) {
            for (int i = 0; i < encoding_dim; i += 2) {
                float angle = pos / powf(10000.0f, (float)i / encoding_dim);
                int idx = (batch * max_seq_length * encoding_dim) + (pos * encoding_dim) + i;
                encodings->data[idx] = sinf(angle);    // this part modify later to use sin and cos lookup table from model file
                if (i + 1 < encoding_dim) {
                    encodings->data[idx + 1] = cosf(angle);
                }
            }
        }
    }
    return true;
}