#include "layer.h"
#include <math.h> // fmax, etc.

Layer init_layer(int in, int out, ActType t, ActInitStrategy strat)
{
    Layer l;
    l.W = alloc_matrix(in, out); // W: in x out
    l.b = alloc_matrix(1, out);  // b: 1 x out
    l.grad_W = alloc_matrix(in, out); // grad_W: in x out
    l.grad_b = alloc_matrix(1, out);  // grad_b: 1 x out
    l.v_W = alloc_matrix(in, out); // v_W: in x out
    l.v_b = alloc_matrix(1, out);  // v_b: 1 x out
    l.v_act = alloc_matrix(1, 1);      // v_act placeholder (will be resized below)
    l.x_cache = alloc_matrix(1024, in);  // x_cache: max_batch x in (increased)
    l.act = init_act(t, out, strat);
    l.in_dim = in;
    l.out_dim = out;
    /* initialize act_lr to empty (will be allocated if n_params > 0) */
    l.act_lr.rows = 0; l.act_lr.cols = 0; l.act_lr.data = NULL;
    mat_rand_xavier(l.W, in);  // Fan-in for W
    mat_rand_xavier(l.b, out); // Fan-out approx for b
    mat_scale(l.b, 0.0);       // Bias zero-init
    mat_scale(l.grad_W, 0.0);  // Zero grads
    mat_scale(l.grad_b, 0.0);
    mat_scale(l.v_W, 0.0); // Zero velocities
    mat_scale(l.v_b, 0.0);
    /* Allocate velocity buffer for activation params if needed */
    if (l.act.n_params > 0)
    {
        free_matrix(l.v_act);
        l.v_act = alloc_matrix(1, l.act.n_params);
        mat_scale(l.v_act, 0.0);
        /* Per-parameter learning rate multipliers: default to 1.0 (no scaling) */
        l.act_lr = alloc_matrix(1, l.act.n_params);
        for (int _i = 0; _i < l.act.n_params; ++_i)
            l.act_lr.data[_i] = 1.0;
    }
    return l;
}

void free_layer(Layer *l)
{
    free_matrix(l->W);
    free_matrix(l->b);
    free_matrix(l->grad_W);
    free_matrix(l->grad_b);
    free_matrix(l->v_W);
    free_matrix(l->v_b);
    free_matrix(l->v_act);
    free_matrix(l->act_lr);
    free_matrix(l->x_cache);
    free_act(&l->act);
}

void layer_forward(Layer *l, Matrix x, Matrix out)
{
    int batch = x.rows;
    if (batch > l->x_cache.rows)
    {
        free_matrix(l->x_cache);
        l->x_cache = alloc_matrix(batch, x.cols);
    }
    copy_matrix(l->x_cache, x); // Store full batch x into cache
    Matrix z = alloc_matrix(batch, l->out_dim);
    matmul(x, l->W, z);    // x (batch x in) @ W (in x out) -> z (batch x out)
    mat_add_bias(z, l->b); // + b broadcast
    // Resize act if needed: assume act.z/out alloc max, copy z to act.z (first batch rows)
    // For simplicity, assume act_forward handles resize or max batch
    act_forward(&l->act, z);
    copy_matrix(out, l->act.out);
    free_matrix(z);
}

void layer_backward(Layer *l, Matrix delta_out, Matrix delta_in)
{
    int batch = delta_out.rows;
    Matrix delta_z = alloc_matrix(batch, l->out_dim);
    act_backward(&l->act, delta_out, delta_z);

    // grad_b = mean(delta_z, axis=0)
    for (int j = 0; j < l->out_dim; ++j)
    {
        mat_t sum = 0.0;
        for (int bb = 0; bb < batch; ++bb)
        {
            sum += delta_z.data[bb * l->out_dim + j];
        }
        l->grad_b.data[j] += sum / batch;
    }

    // grad_W = (x^T @ delta_z) / batch
    // Transpose only the active top 'batch' rows of x_cache
    Matrix xcache_view = {batch, l->x_cache.cols, l->x_cache.data};
    Matrix xt = alloc_matrix(l->in_dim, batch);
    mat_transpose(xcache_view, xt); // x^T (in x batch)
    Matrix outer_temp = alloc_matrix(l->in_dim, l->out_dim);
    matmul(xt, delta_z, outer_temp); // (in x batch) @ (batch x out) -> in x out
    mat_scale(outer_temp, 1.0 / batch);
    add_matrix(l->grad_W, outer_temp); // Accum +=

    // delta_in = delta_z @ W^T
    Matrix wt = alloc_matrix(l->out_dim, l->in_dim);
    mat_transpose(l->W, wt);       // W^T (out x in)
    matmul(delta_z, wt, delta_in); // (batch x out) @ (out x in) -> batch x in

    // Cleanup (delta_z last)
    free_matrix(xt);
    free_matrix(outer_temp);
    free_matrix(wt);
    free_matrix(delta_z);
}