#include "data.h"

int load_data(const char *fname, Matrix *X, Matrix *Y)
{
    FILE *f = fopen(fname, "rb");
    if (!f)
        return 0;
    int n, in_d, out_d;
    fread(&n, sizeof(int), 1, f);
    fread(&in_d, sizeof(int), 1, f);
    fread(&out_d, sizeof(int), 1, f);
    *X = alloc_matrix(n, in_d);
    *Y = alloc_matrix(n, out_d);
    fread(X->data, sizeof(mat_t), n * in_d, f);
    fread(Y->data, sizeof(mat_t), n * out_d, f);
    fclose(f);
    return 1;
}

void gen_xor(Matrix *X, Matrix *Y)
{
    *X = alloc_matrix(4, 2);
    *Y = alloc_matrix(4, 1);
    X->data[0] = 0;
    X->data[1] = 0;
    Y->data[0] = 0;
    X->data[2] = 0;
    X->data[3] = 1;
    Y->data[1] = 1;
    X->data[4] = 1;
    X->data[5] = 0;
    Y->data[2] = 1;
    X->data[6] = 1;
    X->data[7] = 1;
    Y->data[3] = 0;
}

void gen_spirals(Matrix *X, Matrix *Y)
{
    int per_class = 100;
    int n = per_class * 2;
    *X = alloc_matrix(n, 2);
    *Y = alloc_matrix(n, 1);
    for (int i = 0; i < per_class; ++i)
    {
        double r = (double)i / per_class * 5.0;
    double t = 1.75 * (double)i / per_class * 3.141592653589793;
        // class 0
        (*X).data[i * 2 + 0] = r * cos(t) + ((double)rand() / RAND_MAX - 0.5) * 0.1;
        (*X).data[i * 2 + 1] = r * sin(t) + ((double)rand() / RAND_MAX - 0.5) * 0.1;
        (*Y).data[i] = 0.0;
        // class 1
    int j = i + per_class;
    double t2 = t + 3.141592653589793;
        (*X).data[j * 2 + 0] = r * cos(t2) + ((double)rand() / RAND_MAX - 0.5) * 0.1;
        (*X).data[j * 2 + 1] = r * sin(t2) + ((double)rand() / RAND_MAX - 0.5) * 0.1;
        (*Y).data[j] = 1.0;
    }
}