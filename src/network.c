#include "network.h"
#include <math.h> // INFINITY, log, exp, fmax
#include "optimizer.h"

Network init_net(int input_dim, int *arch, int n_arch, ActType *acts)
{
    Network net = {n_arch - 1, malloc((n_arch - 1) * sizeof(Layer)), input_dim};
    for (int i = 0; i < net.n_layers; ++i)
    {
        net.layers[i] = init_layer(arch[i], arch[i + 1], acts[i]);
    }
    return net;
}

void free_net(Network *net)
{
    for (int i = 0; i < net->n_layers; ++i)
        free_layer(&net->layers[i]);
    free(net->layers);
}

// Forward full net (outs optional array of Matrix*)
static void net_forward(Network *net, Matrix x, Matrix *outs)
{
    int batch = x.rows;
    Matrix curr = alloc_matrix(batch, net->input_dim);
    copy_matrix(curr, x);
    for (int i = 0; i < net->n_layers; ++i)
    {
        Matrix next = alloc_matrix(batch, net->layers[i].out_dim);
        layer_forward(&net->layers[i], curr, next);
        if (outs)
            copy_matrix(outs[i], next); // Copy if needed
        free_matrix(curr);
        curr = next;
    }
    free_matrix(curr);
}

// Back full
static void net_backward(Network *net, Matrix delta_out, Matrix y, int is_ce)
{
    int batch = delta_out.rows;
    Matrix curr_delta = alloc_matrix(batch, net->layers[net->n_layers - 1].out_dim);
    copy_matrix(curr_delta, delta_out);
    for (int i = net->n_layers - 1; i >= 0; --i)
    {
        Matrix prev_delta = alloc_matrix(batch, net->layers[i].in_dim);
        layer_backward(&net->layers[i], curr_delta, prev_delta);
        free_matrix(curr_delta);
        curr_delta = prev_delta;
    }
    free_matrix(curr_delta);
}

mat_t train_step(Network *net, Matrix x, Matrix y, SGD *opt, int is_ce)
{
    int batch = x.rows;
    int out_dim = net->layers[net->n_layers - 1].out_dim;
    // Forward
    Matrix out = alloc_matrix(batch, out_dim);
    net_forward(net, x, NULL);
    copy_matrix(out, net->layers[net->n_layers - 1].act.out);
    // Loss + delta_out
    mat_t loss = 0.0;
    Matrix delta_out = alloc_matrix(batch, out_dim);
    
    if (is_ce)
    {
        // Softmax + CE; assume y.cols=1, y.data[b] = class idx (0 to out_dim-1)
        for (int b = 0; b < batch; ++b)
        {
            
            int y_idx = (int)y.data[b];
            // Softmax (stable)
            mat_t maxo = -INFINITY;
            for (int j = 0; j < out_dim; ++j)
                maxo = fmax(maxo, out.data[b * out_dim + j]);
            mat_t sum_exp = 0.0;
            for (int j = 0; j < out_dim; ++j)
            {
                mat_t exp_val = exp(out.data[b * out_dim + j] - maxo);
                sum_exp += exp_val;
                out.data[b * out_dim + j] = exp_val; // Temp
            }
            for (int j = 0; j < out_dim; ++j)
                out.data[b * out_dim + j] /= sum_exp;
            loss -= log(out.data[b * out_dim + y_idx] + 1e-8);
            // Delta = softmax - one_hot(y)
            for (int j = 0; j < out_dim; ++j)
                delta_out.data[b * out_dim + j] = out.data[b * out_dim + j] - (j == y_idx ? 1.0 : 0.0);
        }
        loss /= batch;
    }
    else
    { // MSE
        for (int b = 0; b < batch; ++b)
        {
            for (int j = 0; j < out_dim; ++j)
            {
                mat_t target = (y.cols == 1) ? y.data[b] : y.data[b * y.cols + j];
                
                mat_t d = out.data[b * out_dim + j] - target;
                loss += d * d;
                delta_out.data[b * out_dim + j] = d;
            }
        }
        loss /= (batch * out_dim);
    }

    // Reg (on acts only)
    mat_t reg = 0.0;
    for (int i = 0; i < net->n_layers; ++i)
        reg += act_reg(&net->layers[i].act, 1e-4);
    loss += reg;
    

    // Backprop
    net_backward(net, delta_out, y, is_ce);

    // Clip grads (per layer W/b; acts bounded separately)
    for (int i = 0; i < net->n_layers; ++i)
    {
        mat_clip_grad(net->layers[i].grad_W, 1.0);
        mat_clip_grad(net->layers[i].grad_b, 1.0);
    }
    

    // Update all layers
    for (int i = 0; i < net->n_layers; ++i)
        sgd_update(&net->layers[i], opt);
    

    free_matrix(out);
    free_matrix(delta_out);
    if (isnan(loss) || isinf(loss))
    {
        fprintf(stderr, "Invalid loss in train_step: %f\n", loss);
        exit(1);
    }
    return loss;
}

mat_t eval_acc(Network *net, Matrix x, Matrix y)
{
    int batch = x.rows;
    int out_dim = net->layers[net->n_layers - 1].out_dim;
    Matrix out = alloc_matrix(batch, out_dim);
    net_forward(net, x, NULL);
    copy_matrix(out, net->layers[net->n_layers - 1].act.out);
    int correct = 0;
    for (int b = 0; b < batch; ++b)
    {
        int pred = 0;
        if (out_dim == 1 && y.cols == 1)
        {
            /* Binary classification case: interpret single output as a score/logit.
               If the layer activation is already a sigmoid-like output (FIXED_SIG or SWISH),
               treat it as a probability; otherwise apply sigmoid to convert logits to prob.
               Then threshold at 0.5. */
            ActType last_act = net->layers[net->n_layers - 1].act.type;
            mat_t score = out.data[b * out_dim + 0];
            if (last_act != FIXED_SIG && last_act != SWISH)
                score = sigmoid(score);
            pred = (score >= 0.5) ? 1 : 0;
            int y_true = (int)y.data[b];
            if (pred == y_true)
                ++correct;
        }
        else
        {
            mat_t maxp = out.data[b * out_dim + 0];
            for (int j = 1; j < out_dim; ++j)
            {
                if (out.data[b * out_dim + j] > maxp)
                {
                    maxp = out.data[b * out_dim + j];
                    pred = j;
                }
            }
            int y_true = (y.cols == 1) ? (int)y.data[b] : 0; // Assume cols=1 idx
            if (pred == y_true)
                ++correct;
        }
    }
    free_matrix(out);
    return (mat_t)correct / batch;
}