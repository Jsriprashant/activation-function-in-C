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
    /* Momentum update for activation params (use v_act buffer). */
    if (l->act.n_params > 0)
    {
        /* Ensure v_act size matches n_params (defensive) */
        if (l->v_act.rows * l->v_act.cols < l->act.n_params)
        {
            free_matrix(l->v_act);
            l->v_act = alloc_matrix(1, l->act.n_params);
            mat_scale(l->v_act, 0.0);
            fprintf(stderr, "[DEBUG] Allocated v_act (size=%d) for layer (in=%d out=%d)\n", l->act.n_params, l->in_dim, l->out_dim);
        }

        Matrix gview = {l->act.n_params, 1, l->act.grad_act};
        mat_t gnorm = mat_l2_norm(gview);
        mat_t max_g = 1.0; // threshold (tunable)
        if (gnorm > max_g)
        {
            mat_t scale = max_g / gnorm;
            for (int i = 0; i < l->act.n_params; ++i)
                l->act.grad_act[i] *= scale;
            fprintf(stderr, "[DEBUG] Clipped act.grad norm from %.6f to %.6f for layer (in=%d out=%d)\n", gnorm, max_g, l->in_dim, l->out_dim);
        }

        for (int i = 0; i < l->act.n_params; ++i)
        {
            mat_t g = l->act.grad_act[i];
            /* momentum update: v = mom * v - lr * g; param += v */
            l->v_act.data[i] = mom * l->v_act.data[i] - lr * g;
            l->act.params[i] += l->v_act.data[i];
            l->act.grad_act[i] = 0.0; // reset
            l->act.params[i] = fmin(10.0, fmax(-10.0, l->act.params[i])); // Bound
        }

        /* With exponent-cumulative parameterization for PIECEWISE taus, explicit ordering enforcement is unnecessary. */
    }
}