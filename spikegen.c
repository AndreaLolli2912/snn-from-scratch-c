#include "spikegen.h"
#include "utils.h"  // urand01()

static inline size_t idx_tsni(size_t t, size_t s, size_t n,
                              size_t S, size_t N) {
    return t*(S*N) + s*N + n;
}

void spikegen_rate(const float *in, float *out,
                   size_t T, size_t S, size_t N)
{
    for (size_t t = 0; t < T; ++t) {
        for (size_t s = 0; s < S; ++s) {
            const float *pin = in + s*N;     // sample s
            for (size_t i = 0; i < N; ++i) {
                float p = pin[i];            // already in [0,1]
                out[idx_tsni(t,s,i,S,N)] = (urand01() < p) ? 1.0f : 0.0f;
            }
        }
    }
}
