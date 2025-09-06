// main.c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "net.h"
#include "utils.h"

static void print_vec(const char *name, const float *v, int n) {
    printf("%s = [", name);
    for (int i = 0; i < n; ++i) printf("%s%.6f", (i ? ", " : ""), v[i]);
    printf("]\n");
}

int main(void) {
    // deterministic init so results are reproducible
    srand(1234);

    // 1) make a net
    Net net;
    init_net(&net);

    // 2) add one Linear layer: 4 -> 3 with bias
    const int in_features  = 4;
    const int out_features = 3;
    const int use_bias     = 1;

    if (!net.add_linear_layer(&net, in_features, out_features, use_bias)) {
        fprintf(stderr, "failed to add linear layer\n");
        return 1;
    }
    printf("layers after add = %zu\n", net.n_layers);

    // 3) prepare input x and output y, run forward
    float x[4] = { 1.0f, -2.0f, 0.5f, 3.0f };
    float y[3] = { 0 };

    Layer *L = &net.layers[0];
    if (!L->forward(L->ptr, x, y)) {
        fprintf(stderr, "forward failed\n");
        return 1;
    }
    print_vec("x", x, in_features);
    print_vec("y", y, out_features);

    // 4) quick capacity growth smoke test: add few more linear layers (small sizes)
    for (int k = 0; k < 5; ++k) {
        if (!net.add_linear_layer(&net, 2, 2, 0)) {
            fprintf(stderr, "failed to add layer #%d\n", k+2);
            return 1;
        }
    }
    printf("layers after growth = %zu (capacity=%zu)\n", net.n_layers, net.capacity);

    // 5) delete layers one by one (calls the right destructor)
    while (net.n_layers) {
        net.delete_last_layer(&net);
    }
    printf("layers after delete = %zu\n", net.n_layers);

    // free the array itself (net_reserve used realloc)
    free(net.layers);
    net.layers = NULL;
    net.capacity = 0;

    return 0;
}
