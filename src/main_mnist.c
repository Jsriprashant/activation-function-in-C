#include "network.h"
#include "data.h"
#include "optimizer.h"
#include "utils.h"

int main()
{
    srand_seed(42);                             // Seed 0 (first of 5: 42-46)
    int arch[] = {784, 256, 128, 10};           // MNIST: 28x28=784 in, 10 classes out
     /* Provide one activation per dense layer (hidden1, hidden2, output).
         Previously only two were provided which caused the final layer's
         Activation.type to be uninitialized and led to unpredictable outputs
         and inflated accuracy. */
     ActType acts[] = {POLY_CUBIC, POLY_CUBIC, POLY_CUBIC};
     Network net = init_net(784, arch, 4, acts); // 3 layers: 784->256->128->10
    SGD opt = {0.01, 0.9};                      // lr=0.01, momentum=0.9

    // Load train data
    Matrix X_train, Y_train;
    if (!load_data("data/mnist_train.bin", &X_train, &Y_train))
    {
        fprintf(stderr, "Failed to load mnist_train.bin\n");
        return 1;
    }

    // Load test data
    Matrix X_test, Y_test;
    if (!load_data("data/mnist_test.bin", &X_test, &Y_test))
    {
        fprintf(stderr, "Failed to load mnist_test.bin\n");
        free_matrix(X_train);
        free_matrix(Y_train);
        return 1;
    }

    // Logging setup
    char logf[256];
    sprintf(logf, "experiments/results/mnist_poly_%d.csv", 42);
    FILE *lf = fopen(logf, "w");
    if (!lf)
    {
        fprintf(stderr, "Failed to open log file\n");
        free_matrix(X_train);
        free_matrix(Y_train);
        free_matrix(X_test);
        free_matrix(Y_test);
        free_net(&net);
        return 1;
    }
    // Build header with activation params (layer-wise)
    fprintf(lf, "epoch,loss,acc");
    // Sum total activation params across layers
    int total_params = 0;
    for (int i = 0; i < net.n_layers; ++i)
        total_params += act_get_nparams(&net.layers[i].act);
    for (int i = 0; i < total_params; ++i)
        fprintf(lf, ",p%d", i);
    fprintf(lf, "\n");
    fclose(lf);

    // Batch training
    int batch_size = 32;
    int n_samples = X_train.rows;                              // 10k subset
    int n_batches = (n_samples + batch_size - 1) / batch_size; // Ceiling
    // Reduced epochs for quick test runs
    for (int e = 0; e < 5; ++e)
    {
        mat_t epoch_loss = 0.0;
        mat_t epoch_acc = 0.0;
        for (int b = 0; b < n_batches; ++b)
        {
            // Extract batch
            int start = b * batch_size;
            int end = start + batch_size < n_samples ? start + batch_size : n_samples;
            int curr_batch_size = end - start;

            // Submatrix: Manual slicing
            Matrix X_batch = {curr_batch_size, X_train.cols, X_train.data + start * X_train.cols};
            Matrix Y_batch = {curr_batch_size, Y_train.cols, Y_train.data + start * Y_train.cols};

            // Train step (CE=1 for multi-class)
            mat_t loss = train_step(&net, X_batch, Y_batch, &opt, 1);
            epoch_loss += loss * curr_batch_size; // Weighted sum
            mat_t acc = eval_acc(&net, X_batch, Y_batch);
            epoch_acc += acc * curr_batch_size;
        }
        epoch_loss /= n_samples;
        epoch_acc /= n_samples;

        // Log
        // Gather activation params into one array
        int tp = 0;
        for (int i = 0; i < net.n_layers; ++i)
            tp += act_get_nparams(&net.layers[i].act);
        mat_t *params = NULL;
        if (tp > 0) {
            params = malloc(tp * sizeof(mat_t));
            int idx = 0;
            for (int i = 0; i < net.n_layers; ++i) {
                Activation *a = &net.layers[i].act;
                int np = act_get_nparams(a);
                mat_t *p = act_get_params(a);
                for (int j = 0; j < np; ++j)
                    params[idx++] = p[j];
            }
        }
        log_csv(logf, e, epoch_loss, epoch_acc, tp, params);
        if (params) free(params);
        if (e % 10 == 0)
        {
            printf("Epoch %d: loss=%.4f train_acc=%.4f\n", e, epoch_loss, epoch_acc);
        }
        // Early stop
        if (epoch_acc > 0.95)
        {
            printf("Early stopping at epoch %d (train_acc=%.4f)\n", e, epoch_acc);
            break;
        }
    }

    // Evaluate on test set
    mat_t test_acc = eval_acc(&net, X_test, Y_test);
    printf("Final test accuracy: %.4f\n", test_acc);

    // Cleanup
    free_net(&net);
    free_matrix(X_train);
    free_matrix(Y_train);
    free_matrix(X_test);
    free_matrix(Y_test);
    return 0;
}