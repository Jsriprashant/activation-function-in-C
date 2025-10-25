#include "activations.h"
#include "config.h"

Activation init_act(ActType t, int dim, ActInitStrategy strat)
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
        /* Type-specific initialization strategies. Support multiple
           strategies via 'strat' so experiments can compare inits.
        */
        /* First set sensible defaults (identity-like), then apply strategy tweaks. */
        for (int i = 0; i < a.n_params; ++i) a.params[i] = 0.0;
        switch (t)
        {
        case PRELU:
            /* Single slope alpha: small positive (leaky-style) */
            a.params[0] = 0.25; /* common PReLU init */
            break;
        case SWISH:
            /* Beta param: start near 1 (standard swish) */
            a.params[0] = 1.0;
            break;
        case POLY_CUBIC:
            /* a0 + a1*x + a2*x^2 + a3*x^3
               Start near identity: a0=0, a1=1, higher terms small */
            a.params[0] = 0.0; /* a0 */
            a.params[1] = 1.0; /* a1 */
            a.params[2] = 0.0; /* a2 */
            a.params[3] = 0.0; /* a3 */
            break;
        case PIECEWISE:
            /* params: tau0,tau1,tau2, s0,s1,s2,s3
               Choose breakpoints roughly spaced and slopes near 1. */
            if (a.n_params >= 7)
            {
                a.params[0] = -1.0; /* tau0 */
                a.params[1] = 0.0;  /* tau1 */
                a.params[2] = 1.0;  /* tau2 */
                a.params[3] = 1.0;  /* s0 */
                a.params[4] = 1.0;  /* s1 */
                a.params[5] = 1.0;  /* s2 */
                a.params[6] = 1.0;  /* s3 */
            }
            break;
        default:
            /* For unexpected types, leave zeros (or later apply strategy) */
            break;
        }

        /* Apply initialization strategy overrides */
        if (strat == ACT_INIT_NOISY)
        {
            Matrix tmp = {1, a.n_params, a.params};
            mat_rand_uniform(tmp, -0.01, 0.01);
        }
        else if (strat == ACT_INIT_RANDOM_SMALL)
        {
            Matrix tmp = {1, a.n_params, a.params};
            mat_rand_uniform(tmp, -0.05, 0.05);
        }
        else if (strat == ACT_INIT_IDENTITY)
        {
            /* keep identity-like defaults already set above */
        }
        /* Debug: print initial params for visibility. For PIECEWISE also print derived taus. */
        fprintf(stderr, "[INIT_ACT] type=%d n_params=%d params=", t, a.n_params);
        for (int i = 0; i < a.n_params; ++i)
            fprintf(stderr, " %.6f", a.params[i]);
        if (t == PIECEWISE && a.n_params >= 7)
        {
            mat_t p0 = a.params[0], p1 = a.params[1], p2 = a.params[2];
            mat_t tau0 = p0, tau1 = p0 + exp(p1), tau2 = tau1 + exp(p2);
            fprintf(stderr, " | derived_taus= %.6f %.6f %.6f", tau0, tau1, tau2);
        }
        fprintf(stderr, "\n");
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
        /* Parameterization: params[0] = tau0_raw, params[1] = log(delta1), params[2] = log(delta2)
           Derived taus: tau0 = p0; tau1 = p0 + exp(p1); tau2 = tau1 + exp(p2)
           This guarantees tau0 < tau1 < tau2 (strictly) and keeps learnable raw params.
        */
        mat_t p0 = a->params[0];
        mat_t p1 = a->params[1];
        mat_t p2 = a->params[2];
        mat_t tau0 = p0;
        mat_t tau1 = p0 + exp(p1);
        mat_t tau2 = tau1 + exp(p2);
        mat_t taus[3] = {tau0, tau1, tau2};
        mat_t slopes[4] = {a->params[3], a->params[4], a->params[5], a->params[6]};
        mat_t B = ACT_Z_CLIP_B; // Bound for z (configurable)
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
            /* continuity constant c_seg = sum_{m=0}^{seg-1} (s_m - s_{m+1}) * tau_m */
            mat_t c = 0.0;
            for (int m = 0; m < seg; ++m)
                c += (slopes[m] - slopes[m + 1]) * taus[m];
            a->out.data[i] = slopes[seg] * z + c;
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
        /* Using parameterization where params[0]=p0 (tau0), params[1]=log(delta1), params[2]=log(delta2).
           Derived taus: tau0 = p0; tau1 = p0 + exp(p1); tau2 = tau1 + exp(p2).
           We compute df/dtau_m as before and then map to d/dp via chain rule.
        */
        mat_t p0 = a->params[0];
        mat_t p1 = a->params[1];
        mat_t p2 = a->params[2];
        mat_t tau0 = p0;
        mat_t tau1 = p0 + exp(p1);
        mat_t tau2 = tau1 + exp(p2);
        mat_t taus[3] = {tau0, tau1, tau2};
        mat_t slopes[4] = {a->params[3], a->params[4], a->params[5], a->params[6]};
        /* Accumulators for grads wrt taus (to be mapped to p afterwards) */
        mat_t grad_tau[3] = {0.0, 0.0, 0.0};
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
            /* df/dz */
            delta_z.data[i] = delta_out.data[i] * slopes[seg];
            /* df/dtau_m = (m < seg) ? (s_m - s_{m+1}) : 0 - accumulate into grad_tau */
            for (int m = 0; m < 3; ++m)
            {
                if (m < seg)
                    grad_tau[m] += delta_out.data[i] * (slopes[m] - slopes[m + 1]);
            }
            /* grads for slopes: for k in [0..3] */
            for (int k = 0; k < 4; ++k)
            {
                mat_t contrib = 0.0;
                if (k == seg)
                    contrib += z;
                mat_t left = 0.0, right = 0.0;
                if (k < seg)
                    left = taus[k];
                if ((k - 1) >= 0 && (k - 1) < seg)
                    right = taus[k - 1];
                contrib += (left - right);
                a->grad_act[3 + k] += delta_out.data[i] * contrib;
            }
        }
        /* Map grad_tau -> grad w.r.t params p0,p1,p2 via chain rule:
           tau0 = p0
           tau1 = p0 + exp(p1)
           tau2 = p0 + exp(p1) + exp(p2)
           therefore:
             dL/dp0 = dL/dtau0 + dL/dtau1 + dL/dtau2
             dL/dp1 = exp(p1) * (dL/dtau1 + dL/dtau2)
             dL/dp2 = exp(p2) * (dL/dtau2)
        */
        a->grad_act[0] += grad_tau[0] + grad_tau[1] + grad_tau[2];
        a->grad_act[1] += exp(p1) * (grad_tau[1] + grad_tau[2]);
        a->grad_act[2] += exp(p2) * (grad_tau[2]);
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