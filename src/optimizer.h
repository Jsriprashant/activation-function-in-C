#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "utils.h"
#include "layer.h"

typedef struct
{
    mat_t lr, momentum;
} SGD;

void sgd_update(Layer *l, SGD *opt); // v = mom * v - lr * grad; param += v

#endif