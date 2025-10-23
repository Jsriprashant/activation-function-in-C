#include "network.h"
#include "data.h"
#include "optimizer.h"
#include "utils.h"
#include <string.h>

int main()
{
    srand_seed(42); // Seed 0
    printf("Starting main\n");
     int arch[] = {2, 4, 1};
     /* Provide one activation per dense layer: hidden and output.
         Use a sigmoid-like final activation for binary output. */
     ActType acts[] = {FIXED_SIG, FIXED_SIG};
     Network net = init_net(2, arch, 3, acts);

    printf("Network initialized\n");
    SGD opt = {0.01, 0.9};

    Matrix X, Y;
    gen_xor(&X, &Y);

    printf("XOR data generated: X=%dx%d, Y=%dx%d\n", X.rows, X.cols, Y.rows, Y.cols);
    char logf[256];
    sprintf(logf, "experiments/results/xor_poly_%d.csv", 42);
    FILE *lf = fopen(logf, "w");
    if (!lf)
    {
        fprintf(stderr, "Failed to open %s: %s\n", logf, strerror(errno));
        exit(1);
    }
    fprintf(lf, "epoch,loss,acc");
    int total_params = 0;
    for (int i = 0; i < net.n_layers; ++i)
        total_params += act_get_nparams(&net.layers[i].act);
    for (int i = 0; i < total_params; ++i)
        fprintf(lf, ",p%d", i);
    fprintf(lf, "\n");
    fclose(lf);

    for (int e = 0; e < 100; ++e)
    {
        mat_t loss = train_step(&net, X, Y, &opt, 0); // MSE
        mat_t acc = eval_acc(&net, X, Y);
        // Gather activation params
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
        // if (e % 10 == 0)
        //     printf("Epoch %d: loss=%.4f acc=%.2f\n", e, loss, acc);
        // if (acc > 0.95)
        //     break;
    }
    printf("Training complete\n");
    free_net(&net);
    free_matrix(X);
    free_matrix(Y);
    printf("Resources freed\n");
    return 0;
}