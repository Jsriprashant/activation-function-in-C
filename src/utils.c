#include "utils.h"
#include <stdarg.h>
#include <string.h> // For memcpy
#include <math.h>   // For sqrt, exp, fmax, fmin

Matrix alloc_matrix(int r, int c)
{
    Matrix m = {r, c, malloc(r * c * sizeof(mat_t))};
    if (!m.data)
    {
        fprintf(stderr, "Alloc fail\n");
        exit(1);
    }
    return m;
}

void free_matrix(Matrix m)
{
    if (m.data)
        free(m.data);
}

void copy_matrix(Matrix dst, Matrix src)
{
    /* Allow copying when one matrix is a top/bottom padded buffer.
       If the number of columns differs we bail out (shape mismatch).
       Otherwise copy the minimum number of rows so callers can copy
       "src" into the top rows of "dst" (or vice-versa). */
    if (dst.cols != src.cols)
        return;
    int rows_to_copy = src.rows;
    if (dst.rows < src.rows)
        rows_to_copy = dst.rows;
    /* Copy contiguous block: rows_to_copy * cols elements */
    memcpy(dst.data, src.data, (size_t)rows_to_copy * src.cols * sizeof(mat_t));
}

void matmul(Matrix a, Matrix b, Matrix out)
{
    if (a.cols != b.rows || a.rows != out.rows || b.cols != out.cols)
        return;
    for (int i = 0; i < out.rows; ++i)
    {
        for (int j = 0; j < out.cols; ++j)
        {
            out.data[i * out.cols + j] = 0;
            for (int k = 0; k < a.cols; ++k)
            {
                out.data[i * out.cols + j] += a.data[i * a.cols + k] * b.data[k * b.cols + j];
            }
        }
    }
}

void mat_add_bias(Matrix x, Matrix b)
{
    if (x.cols != b.cols || b.rows != 1) // Enforce 1 x d
        return;
    for (int i = 0; i < x.rows; ++i)
    {
        for (int j = 0; j < x.cols; ++j)
        {
            x.data[i * x.cols + j] += b.data[j]; // b.data[0 to cols-1]
        }
    }
}

void mat_scale(Matrix m, mat_t s)
{
    for (int i = 0; i < m.rows * m.cols; ++i)
        m.data[i] *= s;
}

void mat_transpose(Matrix a, Matrix out)
{
    /* Allow transposing when 'a' has more rows than the active batch
       (e.g. x_cache with fixed max rows). We transpose only the first
       out.cols rows from 'a' into 'out'. Shapes must still be compatible
       in terms of columns. */
    if (a.cols != out.rows || out.cols > a.rows)
        return;
    int rows_to_transpose = out.cols; // number of rows (batch) in 'a' to transpose
    for (int i = 0; i < rows_to_transpose; ++i)
        for (int j = 0; j < a.cols; ++j)
            /* out element at (row=j, col=i) => index = row * out.cols + col */
            out.data[j * out.cols + i] = a.data[i * a.cols + j];
}

void mat_outer(Matrix a, Matrix b, Matrix out)
{ // a (1 x in), b (1 x out) -> in x out
    for (int i = 0; i < a.cols; ++i)
        for (int j = 0; j < b.cols; ++j)
            out.data[i * out.cols + j] = a.data[i] * b.data[j];
}

void add_matrix(Matrix a, Matrix b)
{
    if (a.rows != b.rows || a.cols != b.cols)
        return;
    for (int i = 0; i < a.rows * a.cols; ++i)
        a.data[i] += b.data[i];
}

mat_t mat_l2_norm(Matrix m)
{
    mat_t norm = 0;
    for (int i = 0; i < m.rows * m.cols; ++i)
        norm += m.data[i] * m.data[i];
    return sqrt(norm);
}

void mat_clip_grad(Matrix m, mat_t max_norm)
{
    mat_t norm = mat_l2_norm(m);
    if (norm > max_norm)
        mat_scale(m, max_norm / norm);
}

void mat_rand_xavier(Matrix m, int fan_in)
{
    mat_t bound = sqrt(6.0 / fan_in);
    for (int i = 0; i < m.rows * m.cols; ++i)
    {
        m.data[i] = (mat_t)rand() / RAND_MAX * 2 * bound - bound;
    }
}

void mat_rand_uniform(Matrix m, mat_t low, mat_t high)
{
    mat_t range = high - low;
    for (int i = 0; i < m.rows * m.cols; ++i)
    {
        m.data[i] = low + (mat_t)rand() / RAND_MAX * range;
    }
}

mat_t sigmoid(mat_t x)
{
    return 1.0 / (1.0 + exp(-fmax(-500, fmin(500, x)))); // Stable
}

mat_t sigmoid_deriv(mat_t x)
{
    mat_t s = sigmoid(x);
    return s * (1 - s);
}

void log_csv(const char *fname, int epoch, mat_t loss, mat_t acc, int n_params, mat_t *params)
{
    FILE *f = fopen(fname, "a");
    if (!f)
    {
        fprintf(stderr, "Failed to open %s: %s\n", fname, strerror(errno));
        return;
    }
    fprintf(f, "%d,%.6f,%.6f", epoch, loss, acc);
    if (n_params > 0 && params)
    {
        for (int i = 0; i < n_params; ++i)
            fprintf(f, ",%.6f", params[i]);
    }
    fprintf(f, "\n");
    fclose(f);
}

void srand_seed(unsigned int seed)
{
    srand(seed);
}

void log_csv_header(const char *fname, int n_params, const char **names)
{
    FILE *f = fopen(fname, "w");
    if (!f)
    {
        fprintf(stderr, "Failed to open %s for header: %s\n", fname, strerror(errno));
        return;
    }
    fprintf(f, "epoch,loss,acc");
    if (n_params > 0 && names)
    {
        for (int i = 0; i < n_params; ++i)
            fprintf(f, ",%s", names[i]);
    }
    fprintf(f, "\n");
    fclose(f);
}