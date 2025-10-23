#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include "utils.h"

typedef enum
{
    PRELU,
    POLY_CUBIC,
    PIECEWISE,
    SWISH,
    FIXED_RELU,
    FIXED_SIG
} ActType;

typedef struct
{
    ActType type;
    int n_params;
    mat_t *params;   // Learnable coeffs
    mat_t *grad_act; // Grad accum for params
    Matrix z;        // Pre-act (for backprop)
    Matrix out;      // Post-act
} Activation;

// Init
Activation init_act(ActType t, int dim); // dim for alloc
void free_act(Activation *a);

// Forward: in -> out
void act_forward(Activation *a, Matrix in);

// Backward: delta_out -> delta_in, update act grads (via a->grad_act)
void act_backward(Activation *a, Matrix delta_out, Matrix delta_z);

// Reg term (for loss)
mat_t act_reg(Activation *a, mat_t lambda);

// Helpers to query params
mat_t *act_get_params(Activation *a);
int act_get_nparams(Activation *a);

#endif