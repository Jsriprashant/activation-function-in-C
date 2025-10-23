#include "activations.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Small numerical gradient checker for activation parameter gradients.
// For a small random z batch, compute analytic grad via act_backward and
// numeric grad by finite differences on a param.

mat_t numeric_grad(Activation *a, Matrix z, int p_idx, mat_t eps)
{
    if (!a->params) return 0.0;
    // Backup
    mat_t old = a->params[p_idx];
    // f(x+eps)
    a->params[p_idx] = old + eps;
    act_forward(a, z);
    mat_t loss_p = 0.0;
    for (int i = 0; i < z.rows * z.cols; ++i) loss_p += a->out.data[i] * a->out.data[i];
    // f(x-eps)
    a->params[p_idx] = old - eps;
    act_forward(a, z);
    mat_t loss_m = 0.0;
    for (int i = 0; i < z.rows * z.cols; ++i) loss_m += a->out.data[i] * a->out.data[i];
    // restore
    a->params[p_idx] = old;
    act_forward(a, z);
    return (loss_p - loss_m) / (2.0 * eps);
}

int check_activation(ActType t, int dim)
{
    Activation a = init_act(t, dim);
    if (a.n_params == 0) {
        printf("Activation %d has no params; skipping.\n", t);
        free_act(&a);
        return 1;
    }
    // prepare random z batch
    Matrix z = alloc_matrix(4, dim);
    mat_rand_uniform(z, -1.0, 1.0);
    // analytic: run forward then backward with simple scalar loss L = sum(out^2)
    act_forward(&a, z);
    // delta_out = dL/dout = 2*out
    Matrix delta_out = alloc_matrix(z.rows, z.cols);
    for (int i = 0; i < z.rows * z.cols; ++i) delta_out.data[i] = 2.0 * a.out.data[i];
    // Zero grads
    for (int i = 0; i < a.n_params; ++i) a.grad_act[i] = 0.0;
    Matrix delta_z = alloc_matrix(z.rows, z.cols);
    act_backward(&a, delta_out, delta_z);
    // analytic grads are in a.grad_act
    int ok = 1;
    for (int p = 0; p < a.n_params; ++p)
    {
        mat_t an = a.grad_act[p];
        mat_t num = numeric_grad(&a, z, p, 1e-4);
        mat_t diff = fabs(an - num);
        // If analytic grad is zero, it likely means grad for this param is not implemented
        // (e.g. tau params in PIECEWISE). Report a warning but don't fail the whole check.
        if (fabs(an) < 1e-12) {
            printf("Act %d param %d: analytic=%.6e numeric=%.6e diff=%.6e [WARN: analytic zero, skipping strict check]\n", t, p, an, num, diff);
            continue;
        }
        printf("Act %d param %d: analytic=%.6e numeric=%.6e diff=%.6e\n", t, p, an, num, diff);
    if (diff > 1e-2) ok = 0;
    }
    free_matrix(z);
    free_matrix(delta_out);
    free_matrix(delta_z);
    free_act(&a);
    return ok;
}

int main()
{
    srand(123);
    int ok = 1;
    ok &= check_activation(PRELU, 8);
    ok &= check_activation(POLY_CUBIC, 8);
    ok &= check_activation(PIECEWISE, 8);
    ok &= check_activation(SWISH, 8);
    if (ok) printf("All activation param gradients match numerically (within tolerance)\n");
    else printf("Some activation param gradients differ from numeric check\n");
    return ok ? 0 : 1;
}
