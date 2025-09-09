// main.c
#include "net.h"
#include "dataset.h"
#include "spikegen.h"
#include "utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static const size_t N_SAMPLES    = 10;
static const size_t NUM_INPUTS   = 20;
static const size_t NUM_HIDDEN   = 21;
static const size_t NUM_OUTPUTS  = 200;
static const size_t NUM_STEPS    = 25;

static const float  THRESHOLD_HIDDEN = 0.45f;
static const float  THRESHOLD_OUTPUT = 0.75f;
static const float  BETA_HIDDEN      = 0.80f;
static const float  BETA_OUTPUT      = 0.40f;

int main(void) {
    srand((unsigned)time(NULL));

    // make some [S, N] inputs in [0,1]
    const size_t SN = N_SAMPLES * NUM_INPUTS;
    float *ppg = (float*)malloc(SN * sizeof *ppg);
    if (!ppg) { fprintf(stderr, "malloc ppg failed\n"); return 1; }
    gen_rnd_std_ppg_signals(ppg, SN); // fills SN floats in [0,1]

    // rate-encode to spikes [T, S, N] (time-major)
    const size_t TSN = NUM_STEPS * N_SAMPLES * NUM_INPUTS;
    float *spiking_ppg = (float*)malloc(TSN * sizeof *spiking_ppg);
    if (!spiking_ppg) { fprintf(stderr, "malloc spiking_ppg failed\n"); free(ppg); return 1; }
    spikegen_rate(ppg, spiking_ppg, NUM_STEPS, N_SAMPLES, NUM_INPUTS);
    free(ppg); // done with analog inputs

    // build the network
    Net net;
    init_net(&net);

    int use_bias = 1;
    if (!net.add_linear_layer(&net, (int)NUM_INPUTS, (int)NUM_HIDDEN, use_bias)) {
        fprintf(stderr, "fc1 add failed\n");
        goto CLEANUP_EARLY;
    }
    if (!net.add_leaky_layer(&net, BETA_HIDDEN, THRESHOLD_HIDDEN)) {
        fprintf(stderr, "lif1 add failed\n");
        goto CLEANUP_EARLY;
    }
    if (!net.add_linear_layer(&net, (int)NUM_HIDDEN, (int)NUM_OUTPUTS, use_bias)) {
        fprintf(stderr, "fc2 add failed\n");
        goto CLEANUP_EARLY;
    }
    if (!net.add_leaky_layer (&net, BETA_OUTPUT, THRESHOLD_OUTPUT)) {
        fprintf(stderr, "lif2 add failed\n");
        goto CLEANUP_EARLY;
    }

    // forward pass: out is [T, S, NUM_OUTPUTS] (time-major)
    const size_t TSO = NUM_STEPS * N_SAMPLES * NUM_OUTPUTS;
    float *spk_out = (float*)malloc(TSO * sizeof *spk_out);
    if (!spk_out) { fprintf(stderr, "malloc spk_out failed\n"); goto CLEANUP_EARLY; }

    if (!net_forward(&net, spiking_ppg, spk_out, NUM_STEPS, N_SAMPLES, NUM_INPUTS)) {
        fprintf(stderr, "net_forward failed\n");
        free(spk_out);
        goto CLEANUP_EARLY;
    }

    // quick sanity prints
    // print first timestep, first sample, first 12 outputs
    printf("spk_out[t=0, s=0, :12] = [");
    size_t O = NUM_OUTPUTS;
    for (size_t o = 0; o < 12 && o < O; ++o) {
        size_t idx = 0*(N_SAMPLES*O) + 0*O + o; // t=0, s=0
        printf("%s%.0f", (o? ", ":""), spk_out[idx]);
    }
    printf("]\n");

    // average firing rate over time for sample 0 (all outputs)
    double mean_rate = 0.0;
    for (size_t o = 0; o < O; ++o) {
        size_t count = 0;
        for (size_t t = 0; t < NUM_STEPS; ++t) {
            size_t idx = t*(N_SAMPLES*O) + 0*O + o; // sample 0
            count += (spk_out[idx] != 0.0f);
        }
        mean_rate += (double)count / (double)NUM_STEPS;
    }
    mean_rate /= (double)O;
    printf("Mean firing rate (sample 0 across %zu outputs): %.3f\n", O, (float)mean_rate);

    free(spk_out);

CLEANUP_EARLY:
    net.del_net(&net);
    free(spiking_ppg);
    return 0;
}
