#include "network.h"
#include "data.h"
#include "optimizer.h"
#include "utils.h"

int main() {
    srand_seed(42);  // Seed 0
    int arch[] = {2, 4, 1};
    /* Use POLY_CUBIC for hidden layer and SIG for output */
    ActType acts[] = {POLY_CUBIC, FIXED_SIG};
    Network net = init_net(2, arch, 3, acts);
    SGD opt = {0.01, 0.9};

    Matrix X, Y;

    /* Generate spiral dataset and prepare results CSV */
    gen_spirals(&X, &Y);

    char logf[256];
    sprintf(logf, "experiments/results/spirals_poly_%d.csv", 42);
    int total_params = 0;
    for (int i = 0; i < net.n_layers; ++i) total_params += act_get_nparams(&net.layers[i].act);
    const char **names = NULL;
    if (total_params > 0)
    {
        names = malloc(total_params * sizeof(char *));
        int idx = 0;
        for (int i = 0; i < net.n_layers; ++i)
        {
            Activation *a = &net.layers[i].act;
            const char *atype = "unknown";
            switch (a->type) { case PRELU: atype = "prelu"; break; case POLY_CUBIC: atype = "poly"; break; case PIECEWISE: atype = "piecewise"; break; case SWISH: atype = "swish"; break; case FIXED_RELU: atype = "relu"; break; case FIXED_SIG: atype = "sig"; break; }
            for (int j = 0; j < act_get_nparams(a); ++j)
            {
                char *buf = malloc(64);
                sprintf(buf, "l%d_%s_p%d", i, atype, j);
                names[idx++] = buf;
            }
        }
    }
    log_csv_header(logf, total_params, names);
    if (names) { for (int i = 0; i < total_params; ++i) free((void*)names[i]); free(names); }

    for (int e = 0; e < 100; ++e) {
        mat_t loss = train_step(&net, X, Y, &opt, 0);  // MSE
        mat_t acc = eval_acc(&net, X, Y);
        int tp = 0;
        for (int i = 0; i < net.n_layers; ++i) tp += act_get_nparams(&net.layers[i].act);
        mat_t *params = NULL;
        if (tp > 0) {
            params = malloc(tp * sizeof(mat_t));
            int idx = 0;
            for (int i = 0; i < net.n_layers; ++i) {
                Activation *a = &net.layers[i].act;
                int np = act_get_nparams(a);
                mat_t *p = act_get_params(a);
                for (int j = 0; j < np; ++j) params[idx++] = p[j];
            }
        }
        log_csv(logf, e, loss, acc, tp, params);
        if (params) free(params);
        if (e % 10 == 0) printf("Epoch %d: loss=%.4f acc=%.2f\n", e, loss, acc);
        if (acc > 0.95) break;
    }

    free_net(&net);
    free_matrix(X); free_matrix(Y);
    return 0;
}