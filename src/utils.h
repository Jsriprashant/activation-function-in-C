#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

typedef double mat_t;
typedef struct
{
    int rows, cols;
    mat_t *data;
} Matrix;
Matrix alloc_matrix(int r, int c);
void free_matrix(Matrix m);
void copy_matrix(Matrix dst, Matrix src);
void matmul(Matrix a, Matrix b, Matrix out);
void mat_add_bias(Matrix x, Matrix b);
void mat_scale(Matrix m, mat_t s);
void mat_transpose(Matrix a, Matrix out);
void mat_outer(Matrix a, Matrix b, Matrix out);
void add_matrix(Matrix a, Matrix b);
mat_t mat_l2_norm(Matrix m);
void mat_clip_grad(Matrix m, mat_t max_norm);
void mat_rand_xavier(Matrix m, int fan_in);
void mat_rand_uniform(Matrix m, mat_t low, mat_t high);
mat_t sigmoid(mat_t x);
mat_t sigmoid_deriv(mat_t x);
// Write CSV row with optional activation parameters.
// params: pointer to array of mat_t of length n_params (may be NULL if n_params==0)
void log_csv(const char *fname, int epoch, mat_t loss, mat_t acc, int n_params, mat_t *params);
// Write CSV header with human-readable parameter names (names array of length n_params)
void log_csv_header(const char *fname, int n_params, const char **names);
void srand_seed(unsigned int seed);
#endif