#include "dataset.h"
#include "utils.h"

void gen_rnd_std_ppg_signals(float *ppg, size_t n_samples) {
    for (size_t i = 0; i < n_samples; ++i)
        ppg[i] = urand01();
}
