#ifndef FEED_FORWARD_BACKWARD_H
#define FEED_FORWARD_BACKWARD_H

#include "feed_forward.h"

bool feed_forward_backward(
    FeedForward* ff,
    Tensor* grad_output,
    Tensor* grad_input
);

#endif