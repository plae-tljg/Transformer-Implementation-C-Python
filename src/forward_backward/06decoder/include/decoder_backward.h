#ifndef DECODER_BACKWARD_H
#define DECODER_BACKWARD_H

#include "decoder.h"

bool decoder_backward(
    Decoder* decoder,
    Tensor* grad_output,           
    Tensor* encoder_output,        
    Tensor* grad_encoder_output,   
    Tensor* grad_input,            
    AttentionMask* self_mask,
    AttentionMask* cross_mask
);

#endif
