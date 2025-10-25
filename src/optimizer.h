#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "utils.h"
#include "layer.h"

typedef struct
{
    mat_t lr, momentum;
    /* Separate settings for activation parameter updates */
    mat_t act_lr;        /* learning rate for activation params (base) */
    mat_t act_momentum;  /* momentum for activation params */
    mat_t act_grad_clip; /* L2-norm clip for activation param grads */
} SGD;

void sgd_update(Layer *l, SGD *opt); // v = mom * v - lr * grad; param += v

#endif