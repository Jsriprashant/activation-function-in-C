#ifndef LAYER_H
#define LAYER_H

#include "activations.h"
#include "utils.h"

typedef struct
{
    Matrix W, b, grad_W, grad_b, v_W, v_b, x_cache;
    Matrix v_act; // optimizer velocity for activation params (1 x n_params)
    Matrix act_lr; // per-activation-parameter learning rate multipliers (1 x n_params), default ones
    Activation act;
    int in_dim, out_dim;
} Layer;

// Init layer: in_dim -> out_dim, act_type
Layer init_layer(int in, int out, ActType t, ActInitStrategy strat);

// Free
void free_layer(Layer *l);

// Forward: x (batch x in) -> out (batch x out)
void layer_forward(Layer *l, Matrix x, Matrix out);

// Backward: delta_out (batch x out) -> delta_in (batch x in); update grads
void layer_backward(Layer *l, Matrix delta_out, Matrix delta_in);

#endif