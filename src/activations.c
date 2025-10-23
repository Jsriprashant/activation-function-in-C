#include "activations.h"

Activation init_act(ActType t, int dim)
{
    Activation a;
    a.type = t;
    a.n_params = 0;
    a.params = NULL;
    a.grad_act = NULL;
    a.z = alloc_matrix(1024, dim); // Preallocate for common case
    a.out = alloc_matrix(1024, dim);
    switch (t)
    {
    case PRELU:
    case SWISH:
        a.n_params = 1;
        break;
    case POLY_CUBIC:
        a.n_params = 4;
        break;
    case PIECEWISE:
        a.n_params = 7;
        break;
    default:
        a.n_params = 0;
    }
    if (a.n_params > 0)
    {
        a.params = malloc(a.n_params * sizeof(mat_t));
        a.grad_act = calloc(a.n_params, sizeof(mat_t)); // Zero grads
        Matrix tmp = {1, a.n_params, a.params};
        mat_rand_uniform(tmp, -0.1, 0.1);
        // ... (your inits)
    }
    return a;
}

void free_act(Activation *a) {
    free_matrix(a->z);
    free_matrix(a->out);
    free(a->params);
    free(a->grad_act);  // Free grads
}

void act_forward(Activation *a, Matrix in)
{
    // Ensure buffers are large enough for this batch
    if (in.rows > a->z.rows || in.cols != a->z.cols) {
        free_matrix(a->z);
        free_matrix(a->out);
        a->z = alloc_matrix(in.rows, in.cols);
        a->out = alloc_matrix(in.rows, in.cols);
    }
    copy_matrix(a->z, in); // copy top rows
    int n = in.rows * in.cols; // elements in current batch
    switch (a->type)
    {
    case PRELU:
    {
        mat_t alpha = a->params[0];
        for (int i = 0; i < n; ++i)
        {
            mat_t z = a->z.data[i];
            a->out.data[i] = (z >= 0 ? z : alpha * z);
        }
        break;
    }
    case POLY_CUBIC:
    {
        mat_t a0 = a->params[0], a1 = a->params[1], a2 = a->params[2], a3 = a->params[3];
        for (int i = 0; i < n; ++i)
        {
            mat_t z = a->z.data[i], z2 = z * z, z3 = z2 * z;
            a->out.data[i] = a0 + a1 * z + a2 * z2 + a3 * z3;
        }
        break;
    }
    case PIECEWISE:
    {
        mat_t taus[3] = {a->params[0], a->params[1], a->params[2]};
        mat_t slopes[4] = {a->params[3], a->params[4], a->params[5], a->params[6]};
        mat_t B = 5.0; // Bound
        for (int i = 0; i < n; ++i)
        {
            mat_t z = fmin(B, fmax(-B, a->z.data[i])); // Clip
            int seg = 0;
            if (z > taus[0])
                seg = 1;
            if (z > taus[1])
                seg = 2;
            if (z > taus[2])
                seg = 3;
            // Simple affine per seg; assume continuity via init (no c_k explicit)
            a->out.data[i] = slopes[seg] * z + (seg > 0 ? slopes[seg - 1] * taus[seg - 1] - slopes[seg] * taus[seg - 1] : 0); // Approx continuity
        }
        break;
    }
    case SWISH:
    {
        mat_t beta = a->params[0];
        for (int i = 0; i < n; ++i)
        {
            mat_t z = a->z.data[i], bz = beta * z;
            a->out.data[i] = z * sigmoid(bz);
        }
        break;
    }
    case FIXED_RELU:
    {
        for (int i = 0; i < n; ++i)
            a->out.data[i] = fmax(0, a->z.data[i]);
        break;
    }
    case FIXED_SIG:
    {
        for (int i = 0; i < n; ++i)
            a->out.data[i] = sigmoid(a->z.data[i]);
        break;
    }
    }
    // Caller will copy a->out into the layer output buffer.
}

void act_backward(Activation *a, Matrix delta_out, Matrix delta_z)
{
    int n = delta_out.rows * delta_out.cols;
    // First, delta_z = delta_out * df/dz
    switch (a->type)
    {
    case PRELU:
    {
        mat_t alpha = a->params[0];
        for (int i = 0; i < n; ++i)
        {
            mat_t z = a->z.data[i];
            mat_t dfdz = (z >= 0 ? 1.0 : alpha);
            delta_z.data[i] = delta_out.data[i] * dfdz;
            // Grad alpha: accum (temp in params as grad; reset post-update)
            a->grad_act[0] += delta_out.data[i] * z * (z < 0 ? 1.0 : 0.0);
        }
        break;
    }
    case POLY_CUBIC:
    {
        mat_t a1 = a->params[1], a2 = a->params[2], a3 = a->params[3];
        for (int i = 0; i < n; ++i)
        {
            mat_t z = a->z.data[i], z2 = z * z;
            mat_t dfdz = a1 + 2 * a2 * z + 3 * a3 * z2;
            delta_z.data[i] = delta_out.data[i] * dfdz;
            // Grads: ∂L/∂a_i += delta_out * z^i
            a->grad_act[0] += delta_out.data[i] * 1.0;
            a->grad_act[1] += delta_out.data[i] * z;
            a->grad_act[2] += delta_out.data[i] * z2;
            a->grad_act[3] += delta_out.data[i] * z2 * z;
        }
        break;
    }
    case PIECEWISE:
    {
        // Simplified: dfdz = slope in seg; ∂/∂s_k = delta_out * (z - tau_k) I(seg k); ∂/∂tau approx 0 (or finite diff, but skip for simplicity)
        mat_t taus[3] = {a->params[0], a->params[1], a->params[2]};
        mat_t slopes[4] = {a->params[3], a->params[4], a->params[5], a->params[6]};
        for (int i = 0; i < n; ++i)
        {
            mat_t z = a->z.data[i];
            int seg = 0;
            if (z > taus[0])
                seg = 1;
            if (z > taus[1])
                seg = 2;
            if (z > taus[2])
                seg = 3;
            delta_z.data[i] = delta_out.data[i] * slopes[seg];
            // Grad s_seg += delta_out * z (approx, ignoring tau)
            a->grad_act[3 + seg] += delta_out.data[i] * z;
            // Tau grads: complex, set to 0 for now (extension: impl chain)
        }
        break;
    }
    case SWISH:
    {
        mat_t beta = a->params[0];
        for (int i = 0; i < n; ++i)
        {
            mat_t z = a->z.data[i];
            mat_t bz = beta * z;
            mat_t s = sigmoid(bz);
            mat_t s_prime = s * (1 - s);
            // df/dz = s + z * beta * s'
            mat_t dfdz = s + z * beta * s_prime;
            delta_z.data[i] = delta_out.data[i] * dfdz;
            // df/dbeta = z * ds/dbeta = z * (z * s') = z^2 * s'
            mat_t dfdb = z * z * s_prime;
            a->grad_act[0] += delta_out.data[i] * dfdb;
        }
        break;
    }
    case FIXED_RELU:
    {
        for (int i = 0; i < n; ++i)
        {
            mat_t z = a->z.data[i];
            delta_z.data[i] = delta_out.data[i] * (z > 0 ? 1.0 : 0.0);
        }
        break;
    }
    case FIXED_SIG:
    {
        for (int i = 0; i < n; ++i)
        {
            delta_z.data[i] = delta_out.data[i] * sigmoid_deriv(a->z.data[i]);
        }
        break;
    }
    }
    // Note: a->grad_act now holds accumulated gradients for params; optimizer will apply updates.
}

// Helper: return pointer to params and count
mat_t *act_get_params(Activation *a) { return a->params; }

int act_get_nparams(Activation *a) { return a->n_params; }

mat_t act_reg(Activation *a, mat_t lambda)
{
    if (!a->n_params)
        return 0;
    mat_t reg = 0;
    for (int i = 0; i < a->n_params; ++i)
        reg += a->params[i] * a->params[i];
    reg *= lambda / 2;
    // Linearity penalty
    if (a->type == POLY_CUBIC)
    {
        reg += lambda * (a->params[2] * a->params[2] + a->params[3] * a->params[3]);
    }
    else if (a->type == PRELU)
    {
        mat_t alpha = a->params[0];
        reg += lambda * (alpha - 1) * (alpha - 1); // Penalize near-linear
    }
    // Similar for others...
    return reg;
}