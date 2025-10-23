#ifndef NETWORK_H
#define NETWORK_H

#include "layer.h"
#include "optimizer.h"

typedef struct
{
    int n_layers;
    Layer *layers;
    int input_dim;
} Network;

Network init_net(int input_dim, int *arch, int n_arch, ActType *acts); // arch[0]=input, arch[1]=hid1, ... acts for each post-dense

void free_net(Network *net);

mat_t train_step(Network *net, Matrix x, Matrix y, SGD *opt, int is_ce); // Forward, loss, back, update; return loss

mat_t eval_acc(Network *net, Matrix x, Matrix y); // Argmax out vs y

#endif