#ifndef DATA_H
#define DATA_H

#include "utils.h"

// Load bin: header (int n_samples, in_dim, out_dim), then data
int load_data(const char *fname, Matrix *X, Matrix *Y);

// Gen XOR: 4 samples, 2in 1out
void gen_xor(Matrix *X, Matrix *Y);
// Generate simple 2-spiral dataset with n=100 per class (default: 200 samples)
void gen_spirals(Matrix *X, Matrix *Y);

#endif