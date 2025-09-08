// main.c
#include "net.h"
#include "dataset.h"
#include "utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <time.h>

static int   N_SAMPLES   = 10;
static int   NUM_INPUTS  = 20;
static int   NUM_HIDDEN  = 21;
static int   NUM_OUTPUTS = 200;
static int   NUM_STEPS   = 25;
static float THRESHOLD_HIDDEN = 0.45f;
static float THRESHOLD_OUTPUT = 0.75f;
static float BETA_HIDDEN      = 0.8f;
static float BETA_OUTPUT      = 0.4f;

int main(void) {


    srand(time(NULL));

    float *ppg = malloc(sizeof *ppg * N_SAMPLES * NUM_INPUTS);
    gen_rnd_std_ppg_signals(ppg, N_SAMPLES * NUM_INPUTS);

    printf("First sample\n[%.6f", ppg[0]);
    for (int i = 1; i < NUM_INPUTS; ++i) printf(", %.6f", ppg[i]);
    printf("]\n");

    free(ppg);

    /*
    // data preparation

    // model creation
    int use_bias = 1; // Linear layer bias

    Net *net; // net --- freed.
    init_net(net);
    net->add_linear_layer(net, NUM_INPUTS, NUM_HIDDEN, use_bias); // net.fc1 --- freed
    net->add_leaky_layer(net, BETA_HIDDEN, THRESHOLD_HIDDEN);
    net->add_linear_layer(net, NUM_HIDDEN, NUM_OUTPUTS, use_bias);
    net->add_leaky_layer(net, BETA_OUTPUT, THRESHOLD_OUTPUT);


    net->del_net(net);
    free(net); // free net
    */

    return 0;
}
