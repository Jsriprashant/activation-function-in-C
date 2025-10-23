#include "optimizer.h"
#include "layer.h"

void sgd_update(Layer *l, SGD *opt)
{
    mat_t lr = opt->lr;
    mat_t mom = opt->momentum;

    // Update W with momentum
    for (int i = 0; i < l->W.rows * l->W.cols; ++i)
    {
        l->v_W.data[i] = mom * l->v_W.data[i] - lr * l->grad_W.data[i];
        l->W.data[i] += l->v_W.data[i];
        l->grad_W.data[i] = 0.0; // Reset grad
    }
    // Update b
    for (int i = 0; i < l->b.rows * l->b.cols; ++i)
    {
        l->v_b.data[i] = mom * l->v_b.data[i] - lr * l->grad_b.data[i];
        l->b.data[i] += l->v_b.data[i];
        l->grad_b.data[i] = 0.0;
    }
    // Update act params (simple SGD, no mom)
    for (int i = 0; i < l->act.n_params; ++i)
    {
        mat_t g = l->act.grad_act[i];
        l->act.grad_act[i] = 0.0; // Reset
        l->act.params[i] -= lr * g;
        l->act.params[i] = fmin(10.0, fmax(-10.0, l->act.params[i])); // Bound
    }
}